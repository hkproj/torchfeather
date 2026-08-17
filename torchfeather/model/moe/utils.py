import torch

from torchfeather.tools.utils import _round_up

from .kernels import generate_permute_indices

TOKEN_GROUP_ALIGN_SIZE_M = 8


def _permute(x, num_tokens_per_expert_group, ep_degree, num_local_experts):
    """
    Convert source-major routed rows into padded local-expert-major rows.

    The rank-0 example uses `ep_degree=4`, `num_local_experts=3`, and
    `TOKEN_GROUP_ALIGN_SIZE_M=8`. Rank 0 owns E0-E2 and receives 31 routed rows.
    """

    # x is the hidden-state output of the dispatch all-to-all. Its rows are grouped first by source rank,
    # then by local expert inside each source chunk. Each label stands for one hidden-state row of shape (dim,):
    # x.shape = (31, dim)
    # source 0: [E0: A0,A5 | E1: A0,A1,A6 | E2: A1,A2,A7] <-- the first 8 embeddings in `x` come from source 0, the first two of which should go to LOCAL expert 0 (expert index is not global, but happens to be global only on rank 0)
    # source 1: [E0: A11,A14 | E1: A12,A15 | E2: A8,A13] <-- the next 6 embeddings in `x` come from source 1
    # source 2: [E0: B1,B4,B7 | E1: B2,B5 | E2: B3,B6] <-- the next 7 embeddings in `x` come from source 2
    # source 3: [E0: B10,B13,B14 | E1: B8,B11,B14,B15 | E2: B9,B12,B15] <-- the next 10 embeddings in `x` come from source 3
    # num_tokens_per_expert_group.shape = (ep_degree * num_local_experts,) = (12,)
    # num_tokens_per_expert_group = [2,3,3 | 2,2,2 | 3,2,2 | 3,4,3]

    # Reserve enough space for every real row plus up to one alignment block per local expert
    # it's a conservative upper bound, even for those experts that won't have any tokens associated with them.
    # x_padded_per_expert is a scalar: 31 + 3 * 8 = 55.
    x_padded_per_expert = x.shape[0] + num_local_experts * TOKEN_GROUP_ALIGN_SIZE_M
    # padded_max_len is a scalar: _round_up(55, 8) = 56.
    padded_max_len = _round_up(x_padded_per_expert, TOKEN_GROUP_ALIGN_SIZE_M)
    with torch.no_grad():
        (
            permuted_indices,
            num_tokens_per_expert,
            _offsets,
        ) = generate_permute_indices(
            num_tokens_per_expert_group,
            num_local_experts,
            ep_degree,
            padded_max_len,
            TOKEN_GROUP_ALIGN_SIZE_M,
        )
        # generate_permute_indices returns:
        # permuted_indices.shape = (56,)
        # permuted_indices =
        #     E0 region: [0,1,8,9,14,15,16,21,22,23,-1,-1,-1,-1,-1,-1]
        #     E1 region: [2,3,4,10,11,17,18,24,25,26,27,-1,-1,-1,-1,-1]
        #     E2 region: [5,6,7,12,13,19,20,28,29,30,-1,-1,-1,-1,-1,-1]
        #     unused:    [-1,-1,-1,-1,-1,-1,-1,-1]
        # num_tokens_per_expert.shape = (num_local_experts,) = (3,)
        # num_tokens_per_expert = [16,16,16], the padded grouped-MM row count for E0, E1, and E2.
        # _offsets.shape = (3,), _offsets = [16,32,48]. Grouped MM consumes rows [0:48].

    # Append one all-zero row with index 31. Negative index -1 in permuted_indices (above) selects this padding row.
    x = torch.vstack((x, x.new_zeros(x.shape[-1])))
    # x.shape changes from (31, dim) to (32, dim).
    # Save this shape so _unpermute can reconstruct the 31 real source-major rows and then remove the zero row.
    input_shape = x.shape

    # Gather the source-major rows into padded local-expert-major order.
    x = x[permuted_indices, :]
    # x.shape = (56, dim)
    # E0 rows [0:16]:  [A0,A5,A11,A14,B1,B4,B7,B10,B13,B14, zero x 6]
    # E1 rows [16:32]: [A0,A1,A6,A12,A15,B2,B5,B8,B11,B14,B15, zero x 5]
    # E2 rows [32:48]: [A1,A2,A7,A8,A13,B3,B6,B9,B12,B15, zero x 6]
    # rows [48:56]: eight unused zero rows; num_tokens_per_expert offsets stop at row 48.

    return input_shape, x, permuted_indices, num_tokens_per_expert


def _unpermute(out, input_shape, permuted_indices):
    out_unpermuted = out.new_empty(input_shape)
    out_unpermuted[permuted_indices, :] = out
    out = out_unpermuted[:-1]
    return out
