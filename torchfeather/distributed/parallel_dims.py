from dataclasses import dataclass, field

from loguru import logger
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from torchfeather.tools.device_utils import device_type

__all__ = ["ParallelDims"]


@dataclass
class ParallelDims:
    dp_replicate: int
    dp_shard: int
    cp: int
    tp: int
    pp: int
    ep: int
    etp: int
    world_size: int

    _meshes: dict[str, DeviceMesh] = field(default_factory=dict)
    _global_meshes: dict[str, DeviceMesh] = field(default_factory=dict)
    _world_mesh: DeviceMesh | None = None

    def __post_init__(self):
        self._validate()

    def _validate(self):
        dp_replicate, dp_shard, cp, tp, pp, ep, etp = (
            self.dp_replicate,
            self.dp_shard,
            self.cp,
            self.tp,
            self.pp,
            self.ep,
            self.etp,
        )
        for d in (dp_replicate, cp, tp, pp, ep, etp):
            assert d >= 1, d

        assert dp_shard == -1 or dp_shard >= 1, dp_shard
        if dp_shard < 0:
            self.dp_shard = dp_shard = self.world_size // (dp_replicate * cp * tp * pp)
        assert dp_shard >= 1, dp_shard

        assert dp_replicate * dp_shard * cp * tp * pp == self.world_size

        if ep > 1:
            assert etp == tp or etp == 1, (tp, etp)
            if etp == tp:
                # EP would borrow all cp and some dp_shard degree
                assert ep % cp == 0 and (dp_shard * cp) % ep == 0
            elif etp == 1:
                # EP would borrow all cp and tp and some dp_shard degree
                assert ep % (cp * tp) == 0 and (dp_shard * cp * tp) % ep == 0

    def _mesh_exist(self, name: str, degree: int) -> bool:
        if name == "fsdp":
            # Always keep fsdp mesh with real backend so fully_shard()
            # can apply MixedPrecisionPolicy even at degree 1.
            return True
        if name == "efsdp":
            # We always keep the efsdp if EP is larger than 1 because we need
            # FSDP wrapping to help the MoE layers do mixed precision training.
            return self.ep > 1
        return degree > 1

    def build_mesh(self) -> DeviceMesh:

        def unflatten_mesh(
            world_mesh: DeviceMesh,
            dim_names: tuple[str, ...],
            dim_degrees: tuple[int, ...],
        ):
            backend_override = {}
            for name, degree in zip(dim_names, dim_degrees, strict=True):
                if not self._mesh_exist(name, degree):
                    backend_override[name] = "fake"

            return world_mesh._unflatten(
                0,
                dim_degrees,
                dim_names,
                backend_override=backend_override,
            )

        logger.info(
            f"Building device mesh with parallelism: "
            f"pp={self.pp}, dp_replicate={self.dp_replicate}, dp_shard={self.dp_shard}, "
            f"cp={self.cp}, tp={self.tp}, ep={self.ep}, etp={self.etp}"
        )

        batch = self.dp_replicate * self.dp_shard
        fsdp = self.dp_shard * self.cp
        efsdp = fsdp * self.tp // (self.etp * self.ep)

        self._world_mesh = init_device_mesh(
            device_type, (self.world_size,), mesh_dim_names=("world",)
        )
        dataloading_mesh = unflatten_mesh(
            self._world_mesh,
            ("pp", "batch", "cp", "tp"),
            (self.pp, batch, self.cp, self.tp),
        )
        loss_mesh = dataloading_mesh["batch", "cp"]._flatten("loss_mesh")
        dense_mesh = unflatten_mesh(
            self._world_mesh,
            ("pp", "dp_replicate", "fsdp", "tp"),
            (self.pp, self.dp_replicate, fsdp, self.tp),
        )
        sparse_mesh = unflatten_mesh(
            self._world_mesh,
            # This basically means "EP" can borrow from dp_shard, cp and TP.
            # dp_shard * cp * tp = efsdp * ep * etp
            ("pp", "dp_replicate", "efsdp", "ep", "etp"),
            (self.pp, self.dp_replicate, efsdp, self.ep, self.etp),
        )

        self._global_meshes = {
            "dataloading": dataloading_mesh,
            "loss": loss_mesh,
            "dense": dense_mesh,
            "sparse": sparse_mesh,
        }

        self._meshes = {
            "pp": dataloading_mesh["pp"],
            "batch": dataloading_mesh["batch"],
            "loss": loss_mesh,
            "dp_replicate": dense_mesh["dp_replicate"],
            "fsdp": dense_mesh["fsdp"],
            "cp": dataloading_mesh["cp"],
            "tp": dataloading_mesh["tp"],
            "ep": sparse_mesh["ep"],
            "efsdp": sparse_mesh["efsdp"],
            "etp": sparse_mesh["etp"],
        }

        self._validate_meshes()

        logger.info(
            f"Successfully created meshes with active dimensions: "
            f"{list(self.get_all_one_dimensional_meshes().keys())}"
        )

        return self._world_mesh

    def _validate_meshes(self):
        expected_sizes = {
            "pp": self.pp,
            "batch": self.dp_replicate * self.dp_shard,
            "loss": self.dp_replicate * self.dp_shard * self.cp,
            "dp_replicate": self.dp_replicate,
            "fsdp": self.dp_shard * self.cp,
            "cp": self.cp,
            "tp": self.tp,
            "ep": self.ep,
            "efsdp": self.dp_shard * self.cp * self.tp // (self.etp * self.ep),
            "etp": self.etp,
        }

        for mesh_name, expected_size in expected_sizes.items():
            actual_size = self._meshes[mesh_name].size()
            assert actual_size == expected_size, (mesh_name, actual_size, expected_size)

    def get_optional_mesh(self, dims: str | list[str]) -> DeviceMesh | None:
        if not self._meshes:
            self.build_mesh()

        if isinstance(dims, str):
            dims = [dims]

        for mesh_name in dims:
            if mesh_name not in self._meshes:
                raise ValueError(
                    f"Invalid mesh dim: '{mesh_name}'. "
                    f"Valid dimensions are: {list(self._meshes.keys())}"
                )

        if any(not self._mesh_exist(dim, self._meshes[dim].size()) for dim in dims):
            return None

        if len(dims) == 1:
            return self._meshes[dims[0]]
        else:
            for global_mesh in self._global_meshes.values():
                assert global_mesh.mesh_dim_names is not None
                if not set(dims).issubset(set(global_mesh.mesh_dim_names)):
                    continue
                return global_mesh[tuple(dims)]
            raise ValueError(f"Invalid mesh name combinations {dims}.")

    def get_mesh(self, dims: str | list[str]) -> DeviceMesh:
        mesh = self.get_optional_mesh(dims)
        if mesh is None:
            enabled_str = (
                "enabled (size > 1)" if isinstance(dims, str) else "all enabled"
            )
            raise ValueError(
                f"Mesh '{dims}' is not available. "
                f"Ensure the corresponding parallelism dimension is {enabled_str}."
            )
        return mesh

    def get_all_one_dimensional_meshes(self) -> dict[str, DeviceMesh]:
        if not self._meshes:
            self.build_mesh()
        return {k: v for k, v in self._meshes.items() if v.ndim == 1 and v.size() > 1}

    @property
    def world_mesh(self) -> DeviceMesh:
        if self._world_mesh is None:
            self._world_mesh = self.build_mesh()
        return self._world_mesh

    @property
    def dp_enabled(self):
        return self.dp_replicate > 1 or self.dp_shard > 1

    @property
    def dp_replicate_enabled(self):
        return self.dp_replicate > 1

    @property
    def dp_shard_enabled(self):
        return self.dp_shard > 1

    @property
    def cp_enabled(self):
        return self.cp > 1

    @property
    def dp_cp_enabled(self):
        return self.dp_enabled or self.cp_enabled

    @property
    def fsdp_enabled(self):
        return self.dp_shard_enabled or self.cp_enabled

    @property
    def tp_enabled(self):
        return self.tp > 1

    @property
    def pp_enabled(self):
        return self.pp > 1

    @property
    def ep_enabled(self):
        return self.ep > 1

    @property
    def etp_enabled(self):
        return self.etp > 1

    @property
    def non_data_parallel_size(self):
        return self.cp * self.tp * self.pp

    @property
    def seq_len_divisor(self):
        # Sequence Parallel requires that seq_len be divisible by TP degree.
        # https://github.com/pytorch/torchtitan/pull/640#discussion_r1849481001

        # Context Parallel requires that seq_len be divisible by 2 * CP degree, when load balancing is enabled (by default).
        # https://github.com/pytorch/pytorch/blob/4f62dcc/torch/distributed/tensor/experimental/_attention.py#L1246
        return self.tp * (self.cp * 2)
