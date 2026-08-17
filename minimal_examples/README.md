# Running the examples

Run `torchrun --standalone --nproc-per-node=2 ./minimal_examples/tp_swiglu.py`

## Ring attention

Run `uv run torchrun --standalone --nproc-per-node=4 ./minimal_examples/ring_attention.py`.

## GPipe pipeline parallelism

Run `uv run torchrun --standalone --nproc-per-node=8 ./minimal_examples/pp_gpipe.py`.

## Block matrix multiplication

Run `uv run python ./minimal_examples/block_matrix_multiply.py`.

If you encounter some IPV6 error due to c10d, use this:

```bash
torchrun \
    --nnodes=1 \
    --nproc-per-node=2 \
    --rdzv-backend=static \
    --master-addr=127.0.0.1 \
    --master-port=29500 \
    ./minimal_examples/tp_swiglu.py
```
