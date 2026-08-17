from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DTensor


@dataclass
class MoEArgs:
    num_experts: int = 8
    num_shared_experts: int = 1

    # router
    score_func: Literal["softmax", "sigmoid"] = "sigmoid"
    route_norm: bool = False
    route_scale: float = 1.0
    score_before_experts: bool = True

    # token-choice
    top_k: int = 1
    load_balance_coeff: float | None = 1e-3


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Typical Llama-style Swiglu activation function
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def init_weights(
        self,
        init_std: float,
        buffer_device: torch.device | None = None,
    ):
        nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        for linear in (self.w2, self.w3):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)


def _run_experts_grouped_mm(
    w1: torch.Tensor,  # [num_experts, hidden_dim, dim]
    w2: torch.Tensor,  # [num_experts, dim, hidden_dim]
    w3: torch.Tensor,  # [num_experts, hidden_dim, dim]
    routed_input: torch.Tensor,  # [batch_size * seq_len * top_k, dim]
    num_tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    # num_tokens_per_expert = tensor([
    #     2., 3., 3., 2., 3., 3., 3., 4., 3., 2., 2., 2.
    # ])
    # offsets: [2, 5, 8, 10, 13, 16, 19, 23, 26, 28, 30, 32]
    offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)

    # NOTE: `routed_input` is already ordered! This means the first 2 rows are the ones assigned to expert 0.
    # In the case of pure TP, `routed_input` is extracted using `token_indices_experts_sorted` using torch.gather(x, ..., token_indices_experts_sorted).
    # In the case of EP+TP, `routed_input` comes from `ExpertParallel` which will do the all-to-all to gather all tokens from other ranks

    # `h` shape: [batch_size * seq_len * top_k, hidden_dim]
    h = F.silu(
        torch._grouped_mm(
            routed_input.bfloat16(), w1.bfloat16().transpose(-2, -1), offs=offsets
        )
    )

    # `h` shape: [batch_size * seq_len * top_k, hidden_dim]
    h = h * torch._grouped_mm(
        routed_input.bfloat16(), w3.bfloat16().transpose(-2, -1), offs=offsets
    )

    # `out` shape: [batch_size * seq_len * top_k, dim]
    out = torch._grouped_mm(h, w2.bfloat16().transpose(-2, -1), offs=offsets).type_as(
        routed_input
    )

    return out


class GroupedExperts(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_experts: int,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.w1 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))
        self.w2 = nn.Parameter(torch.empty(num_experts, dim, hidden_dim))
        self.w3 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))

    def forward(
        self,
        routed_input: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(self.w1, DTensor):
            assert isinstance(self.w2, DTensor) and isinstance(self.w3, DTensor)
            # Grouped MM operates on rank-local tensors, so extract each parameter's local shard before invoking the kernel.
            w1 = self.w1.to_local()
            w2 = self.w2.to_local()
            w3 = self.w3.to_local()
        else:
            w1 = self.w1
            w2 = self.w2
            w3 = self.w3

        return _run_experts_grouped_mm(w1, w2, w3, routed_input, num_tokens_per_expert)

    def init_weights(
        self,
        init_std: float,
        buffer_device: torch.device | None = None,
    ):
        nn.init.trunc_normal_(self.w1, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.w2, mean=0.0, std=init_std)
        nn.init.trunc_normal_(self.w3, mean=0.0, std=init_std)


class TokenChoiceTopKRouter(nn.Module):
    def __init__(
        self,
        dim: int,
        num_experts: int,
        top_k: int,
        score_func: Literal["softmax", "sigmoid"],
        route_norm: bool,
        route_scale: float,
    ):
        super().__init__()
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.num_experts = num_experts
        self.top_k = top_k
        self.score_func = score_func
        self.route_norm = route_norm
        self.route_scale = route_scale

    def forward(
        self, x: torch.Tensor, expert_bias: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (batch_size * seq_len, dim)
        # scores: (batch_size * seq_len, num_experts)
        with torch.autocast(device_type=x.device.type, dtype=torch.float32):
            scores = self.gate(x)

        if self.score_func == "sigmoid":
            scores = torch.sigmoid(scores)
        elif self.score_func == "softmax":
            scores = F.softmax(scores, dim=1)
        else:
            raise NotImplementedError(f"Unknown score function {self.score_func}")

        if expert_bias is not None:
            # NOTE: The expert_bias is only used for routing.
            # selected_experts_indices: (batch_size * seq_len, top_k)
            _, selected_experts_indices = torch.topk(
                scores + expert_bias, k=self.top_k, dim=1, sorted=False
            )
            # The gating value top_scores is still derived from the original scores.
            # top_scores: (batch_size * seq_len, top_k)
            top_scores = scores.gather(dim=1, index=selected_experts_indices)
        else:
            # top_scores: (batch_size * seq_len, top_k)
            top_scores, selected_experts_indices = torch.topk(
                scores, k=self.top_k, dim=1, sorted=False
            )

        if self.route_norm:
            # denominator: (batch_size * seq_len, 1)
            denominator = top_scores.sum(dim=-1, keepdim=True) + 1e-20
            # top_scores: (batch_size * seq_len, top_k)
            top_scores = top_scores / denominator
        top_scores = top_scores * self.route_scale

        # num_tokens_per_expert: (num_experts,)
        num_tokens_per_expert = torch.histc(
            selected_experts_indices.view(-1),
            bins=self.num_experts,
            min=0,
            max=self.num_experts,
        )

        return top_scores, selected_experts_indices, num_tokens_per_expert

    def init_weights(
        self,
        init_std: float,
        buffer_device: torch.device | None = None,
    ):
        nn.init.trunc_normal_(self.gate.weight, mean=0.0, std=init_std)


class TokenReorderer(nn.Module):
    def __init__(self, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(
        self,
        # Both tensors of shape (batch_size * seq_len, top_k) normally, OR (batch_size * seq_len // tp_degree, top_k) when EP borrows from TP (but without ETP)
        # This split is done by `ReordererSequenceParallel`
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Example for one rank's 8-token ReordererSequenceParallel shard from a 16-token TP/data input, with 12 experts and top_k=4:
        # selected_experts_indices = tensor([
        #     [0, 1, 4, 7],
        #     [1, 2, 5, 8],
        #     [2, 3, 6, 9],
        #     [3, 4, 7, 10],
        #     [4, 5, 8, 11],
        #     [5, 6, 9, 0],
        #     [6, 7, 10, 1],
        #     [7, 8, 11, 2],
        # ])
        # Flattened: [0, 1, 4, 7, 1, 2, 5, 8, 2, 3, 6, 9, 3, 4, 7, 10, 4, 5, 8, 11, 5, 6, 9, 0, 6, 7, 10, 1, 7, 8, 11, 2]

        # num_tokens_per_expert shape (num_experts,)
        # NOTE: the reason we need to recompute this is because the TokenReorderer may be running on a subset of tokens (as it gets wrapped with ReordererSequenceParallel).
        # num_tokens_per_expert = tensor([
        #     2., 3., 3., 2., 3., 3., 3., 4., 3., 2., 2., 2.
        # ])
        num_tokens_per_expert = torch.histc(
            selected_experts_indices.view(-1),
            bins=self.num_experts,
            min=0,
            max=self.num_experts,
        )

        # token_indices_experts_sorted shape (batch_size * seq_len * top_k,)
        # # token_indices_experts_sorted = tensor([
        #      0, 23,              # expert 0
        #      1,  4, 27,          # expert 1
        #      5,  8, 31,          # expert 2
        #      9, 12,              # expert 3
        #      2, 13, 16,          # expert 4
        #      6, 17, 20,          # expert 5
        #     10, 21, 24,          # expert 6
        #      3, 14, 25, 28,      # expert 7
        #      7, 18, 29,          # expert 8
        #     11, 22,              # expert 9
        #     15, 26,              # expert 10
        #     19, 30,              # expert 11
        # ])
        token_indices_experts_sorted = torch.argsort(
            selected_experts_indices.view(-1), stable=True
        )

        # top_scores: (batch_size * seq_len, top_k)
        # top_scores_experts_sorted: (batch_size * seq_len * top_k,)
        top_scores_experts_sorted = top_scores.view(-1)[token_indices_experts_sorted]

        # After dividing by top_k=4, flattened positions become token indices:
        # token_indices_experts_sorted = tensor([
        #     0, 5,             # expert 0
        #     0, 1, 6,          # expert 1
        #     1, 2, 7,          # expert 2
        #     2, 3,             # expert 3
        #     0, 3, 4,          # expert 4
        #     1, 4, 5,          # expert 5
        #     2, 5, 6,          # expert 6
        #     0, 3, 6, 7,       # expert 7
        #     1, 4, 7,          # expert 8
        #     2, 5,             # expert 9
        #     3, 6,             # expert 10
        #     4, 7,             # expert 11
        # ])
        # This means expert 0 appears in tokens 0 and 5 in the original flattened array above.
        token_indices_experts_sorted = token_indices_experts_sorted // self.top_k

        return (
            top_scores_experts_sorted,
            token_indices_experts_sorted,
            num_tokens_per_expert,
        )

    def init_weights(
        self,
        init_std: float,
        buffer_device: torch.device | None = None,
    ):
        pass


class MoE(nn.Module):
    def __init__(self, moe_args: MoEArgs, dim: int, hidden_dim: int):
        super().__init__()

        num_experts = moe_args.num_experts
        self.experts = GroupedExperts(
            dim=dim,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
        )
        self.router = TokenChoiceTopKRouter(
            dim=dim,
            num_experts=num_experts,
            top_k=moe_args.top_k,
            score_func=moe_args.score_func,
            route_norm=moe_args.route_norm,
            route_scale=moe_args.route_scale,
        )
        self.reorderer = TokenReorderer(num_experts=num_experts, top_k=moe_args.top_k)
        self.shared_experts = (
            FeedForward(dim=dim, hidden_dim=hidden_dim * moe_args.num_shared_experts)
            if moe_args.num_shared_experts > 0
            else None
        )
        self.score_before_experts = moe_args.score_before_experts

        # define fields for auxiliary-loss-free load balancing (https://arxiv.org/abs/2408.15664)
        # NOTE: tokens_per_expert is accumulated in the model forward pass.
        # expert_bias is updated outside the model in an optimizer step pre hook to work with gradient accumulation.
        self.load_balance_coeff = moe_args.load_balance_coeff
        if self.load_balance_coeff is not None:
            assert self.load_balance_coeff > 0.0
            self.register_buffer(
                "expert_bias",
                torch.zeros(num_experts, dtype=torch.float32),
                persistent=True,
            )
        else:
            self.expert_bias = None
        # tokens_per_expert will be used to track expert usage and to update the expert bias for load balancing
        self.register_buffer(
            "tokens_per_expert",
            torch.zeros(num_experts, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(x, DTensor):
            raise TypeError("A plain tensor is expected")

        batch_size, seq_len, dim = x.shape
        # (batch_size, seq_len, dim) -> (batch_size * seq_len, dim)
        x = x.view(-1, dim)

        (
            top_scores,  # (batch_size * seq_len, top_k,)
            selected_experts_indices,  # (batch_size * seq_len, top_k,)
            num_tokens_per_expert,  # (num_experts,)
        ) = self.router(x, self.expert_bias)

        # NOTE: Activation Checkpointing has the side effect of double counting tokens_per_expert: first in the forward pass, and then in the backward pass.
        # However, this has no effect on the expert bias update thanks to the torch.sign() operator.
        # This count is halved in optimiser.py
        # Technically we don't need "no_grad" here because `self.tokens_per_expert` has requires_grad=False by default (check constructor of torch.zeros)
        # But we keep it as it's easier to read and future-proof in case PyTorch API change.
        with torch.no_grad():
            self.tokens_per_expert.add_(num_tokens_per_expert)

        # NOTE: the reason we need to compute num_tokens_per_expert again is:
        #   1st computation in router is to update self.tokens_per_expert which would be the same across all TP ranks.
        #   2nd computation in reorderer is for the actual routing and experts computation which would be sharded over TP ranks if expert_tensor_parallel_degree==1.
        #   If tensor_paralllel_degree == expert_tensor_parallel_degree, they agree because no `ReordererSequenceParallel` is applied.
        (
            top_scores_experts_sorted,  # (batch_size * seq_len * top_k,) OR (batch_size * seq_len // tp_degree * top_k,) if `ReordererSequenceParallel` is applied
            token_indices_experts_sorted,  # (batch_size * seq_len * top_k,) OR (batch_size * seq_len // tp_degree * top_k,) if `ReordererSequenceParallel` is applied
            num_tokens_per_expert,  # (num_experts,)
        ) = self.reorderer(top_scores, selected_experts_indices)

        # shape (batch_size * seq_len * top_k, dim)
        token_indices_experts_sorted = token_indices_experts_sorted.reshape(
            -1, 1
        ).expand(-1, dim)

        # shape (batch_size * seq_len * top_k, dim)
        routed_input = torch.gather(x, dim=0, index=token_indices_experts_sorted)

        if self.score_before_experts:
            routed_input = (
                routed_input.to(torch.float32)
                * top_scores_experts_sorted.reshape(-1, 1)
            ).to(x.dtype)

        # shape (batch_size * seq_len * top_k, dim)
        routed_output = self.experts(routed_input, num_tokens_per_expert)

        # NOTE: we execute the shared expert before scoring the output of the routed expert to "implicitly" overlap the shared expert compute with token combine communication
        if self.shared_experts is not None:
            out = self.shared_experts(x)
        else:
            out = torch.zeros_like(x)

        if not self.score_before_experts:
            routed_output = (
                routed_output.to(torch.float32)
                * top_scores_experts_sorted.reshape(-1, 1)
            ).to(x.dtype)

        # Add output from routed experts to shared experts (if any)
        out = out.scatter_add(
            dim=0, index=token_indices_experts_sorted, src=routed_output
        )
        out = out.reshape(batch_size, seq_len, dim)
        return out

    def init_weights(
        self,
        init_std: float,
        buffer_device: torch.device | None = None,
    ):
        if buffer_device is None:
            raise ValueError(
                "buffer_device must be provided for MoE weight initialization"
            )
        self.experts.init_weights(init_std)
        self.router.init_weights(init_std)
        if self.shared_experts is not None:
            self.shared_experts.init_weights(init_std)

        with torch.device(buffer_device):
            self.tokens_per_expert = torch.zeros(
                self.experts.num_experts, dtype=torch.float32
            )
            if self.load_balance_coeff is not None:
                self.expert_bias = torch.zeros(
                    self.experts.num_experts, dtype=torch.float32
                )
