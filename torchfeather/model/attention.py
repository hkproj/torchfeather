import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor
from torch.nn.attention import SDPBackend, sdpa_kernel

__all__ = [
    "ScaledDotProductAttentionWrapper",
]


class ScaledDotProductAttentionWrapper(torch.nn.Module):
    """Wrapper around `F.scaled_dot_product_attention` to make it CP compatible.

    This wrapper is needed because `F.scaled_dot_product_attention` is not a torch.nn.Module, and thus cannot be applied with _ContextParallel.
    We need to wrap it into a torch.nn.Module.

    Note:
        The forward function must have q, k, v as the first three arguments to be compatible with _ContextParallel.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        scale: float | None = None,
    ) -> torch.Tensor:
        v_head_dim = v.shape[-1]
        q_head_dim = q.shape[-1]

        # When Tensor Parallel (TP) and Context Parallel (CP) are both active, q, k, v arrive as DTensors on the TP mesh.
        # We must convert them to local tensors so that CP's input hook can properly wrap them on the CP mesh for ring attention.
        # Without this, ring attention would incorrectly communicate on the TP process group instead of the CP process group.
        # This also avoids DTensor dispatch issues with F.pad (no registered sharding strategy for pad), which would change V's placements and cause the CP SDPA handler to fail with "inputs need to be redistributed".
        tp_spec = None
        if isinstance(q, DTensor):
            assert isinstance(k, DTensor) and isinstance(v, DTensor)
            tp_spec = (q.device_mesh, q.placements)
            q = q.to_local()
            k = k.to_local()
            v = v.to_local()

        if v_head_dim < q_head_dim:
            # Flash Attention requires Q, K, V to have the same head_dim, but MLA uses v_head_dim < qk_head_dim. Pad V with zeros so Flash Attention can be selected.
            # This is mathematically lossless:
            # softmax(QK^T/s) @ [V | 0] = [softmax(QK^T/s) @ V | 0], so trimming the output recovers the exact original result.
            v = F.pad(v, (0, q_head_dim - v_head_dim))

        with sdpa_kernel(
            [
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.CUDNN_ATTENTION,
            ],
            set_priority=True,
        ):
            out = F.scaled_dot_product_attention(q, k, v, scale=scale, is_causal=True)

        if out.shape[-1] != v_head_dim:
            out = out[..., :v_head_dim]

        # Re-wrap as DTensor on the TP mesh so that downstream layers (e.g., the output projection wo with RowwiseParallel) receive correctly sharded DTensors.
        if tp_spec is not None:
            out = DTensor.from_local(out, tp_spec[0], tp_spec[1], run_check=False)

        return out
