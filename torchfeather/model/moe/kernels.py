import torch
import triton  # pyright: ignore[reportMissingImports]
import triton.language as tl  # pyright: ignore[reportMissingImports]

__all__ = ["generate_permute_indices"]


# parallelized kernel
@triton.jit
def _fill_indices_kernel(
    tokens_per_expert_group_ptr,
    start_index_values_ptr,
    write_offsets_ptr,
    output_ptr,
    experts_per_rank: tl.constexpr,
    num_ranks: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,  # Number of threads per block
):
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)

    # map programs (blocks) to the experts and loop (grid stride) if needed
    for expert_id in range(pid, experts_per_rank, num_programs):
        # read this experts write offset
        write_offset = tl.load(write_offsets_ptr + expert_id)

        for r in range(num_ranks):
            # index into tokens_per_expert_group array
            i = r * experts_per_rank + expert_id

            # load start index and number of tokens for this expert-rank pair
            start_index = tl.load(start_index_values_ptr + i)
            length = tl.load(tokens_per_expert_group_ptr + i)

            # each thread in block processes tokens in parallel
            offsets = tl.arange(0, BLOCK_SIZE)

            # tokens are processed in chunks of BLOCK_SIZE
            for chunk_start in range(0, length, BLOCK_SIZE):
                chunk_offsets = chunk_start + offsets

                # mask valid indices
                mask = chunk_offsets < length

                values = start_index + chunk_offsets

                # destination
                dest_indices = write_offset + chunk_offsets

                # store
                tl.store(output_ptr + dest_indices, values, mask=mask)

            # update write offset for next rank
            write_offset += length


# ==============
# wrapper
# ==============


def fill_indices_wrapper(
    tokens_per_expert_group: torch.Tensor,
    start_index_values: torch.Tensor,
    write_offsets: torch.Tensor,
    experts_per_rank: int,
    num_ranks: int,
    max_len: int,
    block_size: int = 128,
    max_blocks: int = 1024,  # cap on total number of blocks to launch
):
    # preallocate output
    permuted_indices = torch.full(
        (max_len,), -1, dtype=torch.int32, device=tokens_per_expert_group.device
    )

    # write offsets is per local expert...
    num_blocks = min(experts_per_rank, max_blocks)
    # grid = one block per expert unless capped and then we loop...
    grid = (num_blocks,)

    # launch kernel
    _fill_indices_kernel[grid](
        tokens_per_expert_group,
        start_index_values,
        write_offsets,
        permuted_indices,
        experts_per_rank,  # ty:ignore[invalid-argument-type]
        num_ranks,  # ty:ignore[invalid-argument-type]
        BLOCK_SIZE=block_size,  # ty:ignore[invalid-argument-type]
    )
    return permuted_indices


@torch.compiler.disable
def generate_permute_indices(
    tokens_per_expert_group: torch.Tensor,
    experts_per_rank: int,
    num_ranks: int,
    max_len: int,
    alignment: int,
):
    """
    Prepare permutation indices and aligned routed-row counts for each local expert.

    Args:
        tokens_per_expert_group: Source-major routed-row counts with shape `(num_ranks * experts_per_rank,)`.
        experts_per_rank: Number of local experts owned by this EP rank.
        num_ranks: Number of source ranks in the EP group.
        max_len: Allocated length of the output permutation vector.
        alignment: Minimum and alignment multiple for each returned element in `m_sizes`.

    Returns:
        permuted_indices: Shape `(max_len,)`. Indices that map source-major rows to padded local-expert-major rows.
        m_sizes: Shape `(experts_per_rank,)`. Aligned number of rows allocated to each local expert.
        m_offsets: Shape `(experts_per_rank,)`. Exclusive ending offset of each local expert's aligned rows.

    Explanatory details:
        The rank-0 example uses `num_ranks=4`, `experts_per_rank=3`, `max_len=56`, and `alignment=8`.
    """

    # Source-major counts: each three-value chunk contains E0, E1, and E2 counts from one source rank.
    # tokens_per_expert_group.shape = (num_ranks * experts_per_rank,) = (12,)
    # rank 0 tokens_per_expert_group = [2,3,3       |2,2,2      |3,2,2       |3,4,3]
    #                                   source 0     source 1    source 2     source 3

    # prefix sum to get start index of each expert (parallel scan kernel in future?)
    start_index_values = (
        torch.cumsum(tokens_per_expert_group, 0) - tokens_per_expert_group
    )
    # start_index_values.shape = (12,)
    # start_index_values = [0,2,5 | 8,10,12 | 14,17,19 | 21,24,28]
    # For example, source 2's E1 rows start at routed_input row 17 and have length 2, so they occupy rows [17,18].

    # total routed rows for each local expert (sum over source ranks)
    total_tokens_per_expert = tokens_per_expert_group.view(num_ranks, -1).sum(0)
    # tokens_per_expert_group.view(num_ranks, -1).shape = (4,3)
    # [[2,3,3],
    #  [2,2,2],
    #  [3,2,2],
    #  [3,4,3]]
    # total_tokens_per_expert.shape = (experts_per_rank,) = (3,)
    # total_tokens_per_expert = [10,11,10]

    # pad out empty experts to alignment requirement
    total_tokens_per_expert = torch.clamp_min(total_tokens_per_expert, alignment)
    # Each expert receives at least one alignment block. In this example every count is already greater than 8,
    # so total_tokens_per_expert remains [10,11,10] with shape (3,).

    # align the chunk sizes (ceiling division, basically round up)
    m_sizes = ((total_tokens_per_expert + alignment - 1) // alignment * alignment).to(
        torch.int32
    )
    # Round each expert's allocation up to a multiple of 8: [10,11,10] -> [16,16,16].
    # m_sizes.shape = (3,), m_sizes.dtype = torch.int32

    # additional prefix sum to get write offset of each expert in permuted_indices
    # write offsets is per local expert, not global
    m_offsets = torch.cumsum(m_sizes, 0)
    write_offsets = m_offsets - m_sizes
    # m_offsets.shape = write_offsets.shape = (3,)
    # m_offsets = [16,32,48]: exclusive ends of the E0, E1, and E2 aligned regions.
    # write_offsets = [0,16,32]: starting positions of those regions in permuted_indices.

    permuted_indices = fill_indices_wrapper(
        tokens_per_expert_group,
        start_index_values,
        write_offsets,
        experts_per_rank,
        num_ranks,
        max_len,
    )
    # The Triton kernel walks sources inside each local expert and writes the original source-major row indices into that expert's aligned output region. Unwritten entries remain -1:
    # permuted_indices.shape = (max_len,) = (56,)
    # E0 region [0:16]:  [0,1, 8,9, 14,15,16, 21,22,23, -1,-1,-1,-1,-1,-1]
    # E1 region [16:32]: [2,3,4, 10,11, 17,18, 24,25,26,27, -1,-1,-1,-1,-1]
    # E2 region [32:48]: [5,6,7, 12,13, 19,20, 28,29,30, -1,-1,-1,-1,-1,-1]
    # Extra capacity [48:56]: [-1,-1,-1,-1,-1,-1,-1,-1]
    # There are 31 valid row indices and 25 padding entries. `_permute` appends one zero row to `x`, so every -1 indexes that padding row.
    # The grouped-MM offsets end at m_offsets[-1] = 48, so the final 8 entries are unused capacity.

    return permuted_indices, m_sizes, m_offsets.to(torch.int32)
