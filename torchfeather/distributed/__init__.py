from functools import partial

from torch import nn
from torch.distributed.tensor import (
    DeviceMesh,
    DTensor,
    Replicate,
    distribute_module,
)
from torch.distributed.tensor.parallel import ParallelStyle
from torch.distributed.tensor.placement_types import Placement

from torchfeather.distributed.parallel_dims import ParallelDims

__all__ = ["NoParallel", "ParallelDims"]


class NoParallel(ParallelStyle):
    def __init__(
        self,
        *,
        input_layout: Placement | None = None,
        output_layout: Placement | None = None,
        use_local_output: bool = True,
        output_grad_placements: tuple[Placement, ...] | None = None,
    ):
        super().__init__()
        self.input_layout = input_layout or Replicate()
        self.output_layout = output_layout or Replicate()
        self.desired_input_layout = Replicate()
        self.use_local_output = use_local_output
        self.output_grad_placements = output_grad_placements

    @staticmethod
    def _prepare_input_fn(input_layout, desired_input_layout, mod, inputs, device_mesh):
        # annotate module input placements/sharding with input_layouts
        input_tensor = inputs[0]
        if not isinstance(input_tensor, DTensor):
            input_tensor = DTensor.from_local(
                input_tensor, device_mesh, (input_layout,), run_check=False
            )

        if input_layout != desired_input_layout:
            input_tensor = input_tensor.redistribute(
                placements=(desired_input_layout,), async_op=True
            )
        return (input_tensor, *inputs[1:])

    @staticmethod
    def _prepare_output_fn(
        output_layout,
        use_local_output,
        output_grad_placements,
        mod,
        outputs,
        device_mesh,
    ):
        if outputs.placements != (output_layout,):
            outputs = outputs.redistribute(placements=(output_layout,), async_op=True)
        # back to local tensor
        if use_local_output:
            return outputs.to_local(grad_placements=output_grad_placements)
        return outputs

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            None,  # if this is None, the weights are replicated by default
            partial(
                self._prepare_input_fn, self.input_layout, self.desired_input_layout
            ),
            partial(
                self._prepare_output_fn,
                self.output_layout,
                self.use_local_output,
                self.output_grad_placements,
            ),
        )
