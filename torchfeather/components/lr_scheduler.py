import copy
import functools
import math
from collections.abc import Callable, Iterator
from typing import Any

from loguru import logger
from torch.distributed.checkpoint.stateful import Stateful
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

from torchfeather.components.optimizer import OptimizersContainer
from torchfeather.config import LRScheduler as LRSchedulerConfig

__all__ = [
    "LRSchedulersContainer",
    "build_lr_schedulers",
]


class LRSchedulersContainer(Stateful):
    schedulers: list[LRScheduler]

    def __init__(self, optimizers: OptimizersContainer, lr_lambda: Callable) -> None:
        assert len(optimizers) > 0, (
            "Must have at least one optimizer to create LRScheduler"
        )

        self.schedulers = [LambdaLR(optimizer, lr_lambda) for optimizer in optimizers]

    def __iter__(self) -> Iterator[LRScheduler]:
        return iter(self.schedulers)

    def __len__(self) -> int:
        return len(self.schedulers)

    def step(self) -> None:
        for scheduler in self.schedulers:
            scheduler.step()

    def state_dict(self) -> dict[str, Any]:
        # While there may be multiple schedulers, we only save the first one because the state_dict is the same for all. See the limitations section in the docstring.
        return self.schedulers[0].state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # Load the same state_dict for all schedulers. The key value we're concerned within ``LRScheduler.state_dict()`` is ``last_epoch``, which is an integer that is immutable. As long as ``training.steps`` and ``lr_scheduler.warmup_steps`` in ``job_config`` remain unchanged when resuming from a checkpoint, this approach is safe. We call ``copy()`` here to ensure extra safety.
        for scheduler in self.schedulers:
            scheduler.load_state_dict(copy.deepcopy(state_dict))


def build_lr_schedulers(
    optimizers: OptimizersContainer,
    lr_scheduler_config: LRSchedulerConfig,
    training_steps: int,
) -> LRSchedulersContainer:
    warmup_steps = int(lr_scheduler_config.warmup_steps)

    if warmup_steps > training_steps:
        logger.warning(
            f"Warmup steps ({warmup_steps}) exceed total training steps ({training_steps}). "
            f"Adjusting warmup steps to {training_steps}."
        )
        warmup_steps = training_steps

    if lr_scheduler_config.decay_ratio is not None:
        decay_steps = round(training_steps * lr_scheduler_config.decay_ratio)
        if warmup_steps + decay_steps > training_steps:
            logger.warning(
                f"Warmup ({warmup_steps}) + decay ({decay_steps}) steps exceed "
                f"total training steps ({training_steps}). "
                f"Adjusting decay steps to {training_steps - warmup_steps}."
            )
            decay_steps = training_steps - warmup_steps
    else:
        decay_steps = training_steps - warmup_steps
    # Add a virtual last step to prevent the learning rate from dropping to 0
    stable_steps = training_steps + 1 - warmup_steps - decay_steps
    lr_decay_type = lr_scheduler_config.decay_type
    min_lr_factor = lr_scheduler_config.min_lr_factor

    def linear_warmup_stable_decay(
        current_step: int,
        warmup_steps: int,
        stable_steps: int,
        decay_steps: int,
        lr_decay_type: str,
        min_lr_factor: float,
    ):
        warmup_stable_steps = warmup_steps + stable_steps
        if current_step < warmup_steps:
            # linear warmup
            # 0-indexed step, hence + 1 adjustments
            current_step += 1
            assert warmup_steps != 0, (
                "warmup_steps must not be zero to reach this branch"
            )
            curr_adjustment = float(current_step / warmup_steps)
        elif current_step < warmup_stable_steps:
            curr_adjustment = 1.0
        else:
            # 0-indexed step, hence + 1 adjustments
            current_step += 1
            assert decay_steps != 0, "decay_steps must not be zero to reach this branch"
            progress = float(current_step - warmup_stable_steps) / decay_steps

            if lr_decay_type == "linear":
                curr_adjustment = 1 - progress
            elif lr_decay_type == "sqrt":
                curr_adjustment = 1 - math.sqrt(progress)
            elif lr_decay_type == "cosine":
                curr_adjustment = 0.5 * (1.0 + math.cos(math.pi * progress))
            else:
                raise ValueError(f"Unknown lr_decay_type: {lr_decay_type}")
            curr_adjustment = min_lr_factor + (1 - min_lr_factor) * curr_adjustment
        return curr_adjustment

    lr_lambda = functools.partial(
        linear_warmup_stable_decay,
        warmup_steps=warmup_steps,
        stable_steps=stable_steps,
        decay_steps=decay_steps,
        lr_decay_type=lr_decay_type,
        min_lr_factor=min_lr_factor,
    )
    return LRSchedulersContainer(optimizers, lr_lambda)
