import enum
import functools
import os
import queue
import re
import shutil
import threading
import time
from concurrent.futures import Future
from typing import Any, cast

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from loguru import logger
from torch import nn
from torch.distributed.checkpoint.metadata import Metadata
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    set_model_state_dict,
)
from torch.distributed.checkpoint.state_dict_saver import async_save as dcp_async_save
from torch.distributed.checkpoint.state_dict_saver import save as dcp_save
from torch.distributed.checkpoint.stateful import Stateful

from torchfeather.components.dataloader import BaseDataLoader
from torchfeather.components.lr_scheduler import LRSchedulersContainer
from torchfeather.components.optimizer import OptimizersContainer
from torchfeather.config import Checkpoint as CheckpointConfig
from torchfeather.tools.utils import GarbageCollection

MODEL = "model"
OPTIMIZER = "optimizer"
LR_SCHEDULER = "lr_scheduler"
DATALOADER = "dataloader"

CHECKPOINT_FOLDER_FORMAT = r"step-(\d+)"


class AsyncMode(str, enum.Enum):
    DISABLED = "disabled"
    ASYNC = "async"


class ModelWrapper(Stateful):
    def __init__(self, model: nn.Module | list[nn.Module]) -> None:
        self.model = [model] if isinstance(model, nn.Module) else model
        self.cache_state_dict = self._get_state_dict()

    def _get_state_dict(self) -> dict[str, Any]:
        state_dict = {
            k: v for sd in map(get_model_state_dict, self.model) for k, v in sd.items()
        }
        return state_dict

    def state_dict(self) -> dict[str, Any]:
        return self.cache_state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        func = functools.partial(
            set_model_state_dict,
            model_state_dict=state_dict,
            options=StateDictOptions(strict=False),
        )
        list(map(func, self.model))
        # `set_model_state_dict()` does change the keys of the input state_dict, we will need to reinitialize the cache_state_dict.
        self.cache_state_dict = self._get_state_dict()


class TerminateSentinel:
    pass


def purge_thread(purge_queue: queue.Queue):
    try:
        while True:
            path = purge_queue.get()
            # This sentinel message indicates the thread should exit.
            if isinstance(path, TerminateSentinel):
                return
            assert isinstance(path, str), type(path)
            logger.info("Checkpointer is deleting {}.", path)
            begin = time.monotonic()
            shutil.rmtree(path, ignore_errors=True)
            logger.info(
                "Checkpointer deleted {} in {:.2f} seconds.",
                path,
                time.monotonic() - begin,
            )
    finally:
        logger.info("Destroying the purge thread.")


class CheckpointManager:
    def __init__(
        self,
        dataloader: BaseDataLoader | None,
        model_parts: list[nn.Module],
        optimizers: OptimizersContainer,
        lr_schedulers: LRSchedulersContainer,
        additional_states: dict[str, Any],
        checkpoint_config: CheckpointConfig,
        base_folder: str = "",
    ) -> None:
        self.config = checkpoint_config
        self.async_save_pg = None
        self.save_future = None
        self.purge_thread = None

        if not self.config.enable:
            return

        self.states = additional_states
        self.states.update(
            {
                MODEL: ModelWrapper(model_parts),
                OPTIMIZER: optimizers,
                DATALOADER: dataloader,
                LR_SCHEDULER: lr_schedulers,
            }
        )

        self.folder = os.path.join(base_folder, checkpoint_config.folder)

        # Async checkpoint related fields.
        if self.config.async_mode == AsyncMode.ASYNC:
            # Creates a new CPU process group for async checkpointing in which all ranks participate.
            self.async_save_pg = dist.new_group(backend="gloo")

        if self.config.keep_latest_k == 1:
            raise ValueError(
                "keep_latest_k=1 is not supported because the last checkpoint "
                "could be purged while a new one is being saved."
            )

        if self.config.keep_latest_k > 0:
            self.purge_queue = queue.Queue()
            self.purge_thread = threading.Thread(
                target=purge_thread, args=(self.purge_queue,), daemon=True
            )
            self.purge_thread.start()
        else:
            self.purge_thread = None

        self.save_future = None
        if self.config.async_mode == AsyncMode.DISABLED:
            self.async_mode = AsyncMode.DISABLED  # ty:ignore[unresolved-attribute]
        elif self.config.async_mode == AsyncMode.ASYNC:
            self.async_mode = AsyncMode.ASYNC  # ty:ignore[unresolved-attribute]
        else:
            raise NotImplementedError(
                f"Unsupported async checkpoint mode: {self.config.async_mode}"
            )

        logger.info(
            f"Checkpointing active. Checkpoints will be loaded from and saved to {self.folder}"
        )

    def __del__(self):
        self.close()

    def close(self):
        if self.config.enable:
            if self.save_future is not None:
                self._async_wait()
            if self.async_save_pg is not None:
                dist.destroy_process_group(self.async_save_pg)
                self.async_save_pg = None
            if self.purge_thread and self.purge_thread.is_alive():
                self.purge_queue.put(TerminateSentinel())
                self.purge_thread.join()

    @torch.no_grad()
    def dcp_save(
        self,
        state_dict: dict[str, Any],
        checkpoint_id: str,
        async_mode: AsyncMode,
        enable_garbage_collection: bool = False,
    ) -> Future[Any] | Metadata:
        ret = None

        if async_mode == AsyncMode.ASYNC:
            ret = cast(
                Future[Any],
                dcp_async_save(
                    state_dict,
                    checkpoint_id=checkpoint_id,
                    process_group=self.async_save_pg,
                ),
            )
        else:
            ret = dcp_save(
                state_dict,
                checkpoint_id=checkpoint_id,
            )

        if enable_garbage_collection:
            GarbageCollection.collect("GC collection invoked by checkpointer.")

        return ret

    def dcp_load(
        self,
        state_dict: dict[str, Any],
        checkpoint_id: str,
    ) -> None:
        dcp.load(state_dict, checkpoint_id=checkpoint_id)  # pyright: ignore[reportPrivateImportUsage]

        # TODO: Since we flatten the model states in state_dict, we need to manually call load_state_dict() for the model. Need to fix this.
        if MODEL in self.states:
            self.states[MODEL].load_state_dict(state_dict)

    @torch.no_grad()
    def save(self, curr_step: int, last_step: bool = False) -> None:
        if not self._should_save(curr_step, last_step):
            return

        begin = time.monotonic()
        logger.info("Saving the checkpoint (async if enabled).")
        checkpoint_id = self._create_checkpoint_id(curr_step)
        self._async_wait()
        # This GC is called for async checkpoint as it is useless to do GC right after async_save -- the CPU memory is not able to be freed until _async_wait()
        if last_step:
            self._save_last_step(curr_step)
            return

        states = self._flattened_model_states_sd()
        if self.async_mode == AsyncMode.ASYNC:  # ty:ignore[unresolved-attribute]
            GarbageCollection.collect("GC collection invoked by checkpointer.")
            self.save_future = self.dcp_save(
                states,
                checkpoint_id=checkpoint_id,
                async_mode=self.async_mode,  # ty:ignore[unresolved-attribute]
            )
            GarbageCollection.collect("GC collection invoked by checkpointer.")
        else:
            self.dcp_save(
                states,
                checkpoint_id=checkpoint_id,
                async_mode=AsyncMode.DISABLED,
                enable_garbage_collection=True,
            )
        self._purge_stale_checkpoints()

        logger.info(
            f"Finished saving the checkpoint in {time.monotonic() - begin:.2f} seconds."
        )

    @torch.no_grad()
    def load(self, step: int = -1) -> bool:
        if not self.config.enable:
            return False

        # if only load the model weights but not the optimizer states
        model_only = False

        if not os.path.exists(self.folder):
            if self.config.initial_load_path:
                # Load from the provided initial load path.
                checkpoint_id = self.config.initial_load_path
                if not os.path.isdir(checkpoint_id):
                    raise ValueError(
                        f"initial_load_path {self.config.initial_load_path} is not a valid directory."
                    )
            else:
                # No checkpoint to load.
                return False
        else:
            step = self._find_load_step() if step == -1 else step
            if step == -1:
                # No checkpoint to load.
                return False
            model_only = step == 0
            checkpoint_id = self._create_checkpoint_id(step)

            if not os.path.isdir(checkpoint_id):
                raise FileNotFoundError(
                    f"load_step={step} but checkpoint {checkpoint_id} is not found."
                )

        assert os.path.isdir(checkpoint_id), checkpoint_id
        logger.info(f"Loading the checkpoint from {checkpoint_id}.")
        begin = time.monotonic()
        states = self._states_to_load(model_only)
        self.dcp_load(
            states,
            checkpoint_id=checkpoint_id,
        )
        GarbageCollection.collect("GC collection for checkpoint loading.")
        logger.info(
            f"Finished loading the checkpoint in {time.monotonic() - begin:.2f} seconds."
        )
        return True

    def _find_load_step(self, folder: str = "") -> int:
        folder = folder if folder else self.folder
        pattern = CHECKPOINT_FOLDER_FORMAT
        step_counts = []

        if not os.path.isdir(folder):
            return -1

        for filename in os.listdir(folder):
            match = re.search(pattern, filename)
            dcp_metadata_probe = os.path.join(folder, filename, ".metadata")
            if match and os.path.isfile(dcp_metadata_probe):
                step_counts.append(int(match.group(1)))
        if not step_counts:
            return -1
        return max(step_counts)

    def _create_checkpoint_id(self, step: int, folder: str = "") -> str:
        folder = folder if folder else self.folder
        return os.path.join(folder, f"step-{step}")

    def _flattened_model_states_sd(
        self, state_dict: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        states = state_dict if state_dict is not None else self.states
        sd = {k: v for k, v in states.items() if k != MODEL}
        if MODEL in states:
            sd.update(states[MODEL].state_dict())
        return sd

    def _states_to_load(self, model_only: bool) -> dict[str, Any]:
        # For the first step, we will only load the model.
        if model_only:
            return self.states[MODEL].state_dict()

        states_to_load = {k: v for k, v in self.states.items()}
        states_to_load = self._flattened_model_states_sd(states_to_load)
        return states_to_load

    def _save_last_step(self, curr_step: int) -> None:
        logger.info(f"Saving a full checkpoint at last step, step {curr_step}.")
        states = self._flattened_model_states_sd()

        self.dcp_save(
            states,
            checkpoint_id=self._create_checkpoint_id(curr_step),
            async_mode=AsyncMode.DISABLED,
            enable_garbage_collection=True,
        )

    def _should_save(self, curr_step: int, last_step: bool = False) -> bool:
        if not self.config.enable:
            return False

        if curr_step == 1 and self.config.enable_first_step_checkpoint:
            return True

        if last_step:
            return True

        return curr_step % self.config.interval == 0

    def _async_wait(self) -> None:
        if self.async_mode == AsyncMode.ASYNC:  # ty:ignore[unresolved-attribute]
            if self.save_future is not None:
                cast(Future[Any], self.save_future).result()
                self.save_future = None
        elif self.save_future is not None:
            raise RuntimeError(
                "self.save_future is not None, but self.async_mode is not enabled."
            )

    def _purge_stale_checkpoints(self):
        if (
            self.config.keep_latest_k > 0
            and dist.get_rank() == 0
            and os.path.isdir(self.folder)
        ):
            discovered_checkpoints = []
            for filename in os.listdir(self.folder):
                match = re.search(CHECKPOINT_FOLDER_FORMAT, filename)
                if match:
                    path = os.path.join(self.folder, filename)
                    discovered_checkpoints.append((int(match.group(1)), path))

            discovered_checkpoints.sort()
            to_delete = discovered_checkpoints[: -1 * self.config.keep_latest_k]

            for _, path in to_delete:
                # Inform the purge thread to delete the checkpoint.
                assert self.purge_thread is not None
                self.purge_queue.put(path)
