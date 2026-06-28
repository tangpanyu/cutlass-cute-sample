"""Small CuTe layout-composition example for the conda flashinfer env."""

import os

NEG_INF = float("-inf")


def init_scores(rows: int = 64, cols: int = 64, value: float = 0.0):
    return [[value for _ in range(cols)] for _ in range(rows)]


def _prepare_tvm_ffi_import() -> None:
    """Avoid first-import JIT work that is unnecessary for static layout algebra."""
    os.environ.setdefault("TVM_FFI_CACHE_DIR", "/tmp/tvm-ffi")

    try:
        import torch
    except ImportError:
        return

    if not hasattr(torch.Tensor, "__dlpack_c_exchange_api__"):
        setattr(torch.Tensor, "__dlpack_c_exchange_api__", None)


def apply_attention_mask(
    scores,
    *,
    seqlen_q: int = 64,
    seqlen_k: int = 64,
    m_block: int = 0,
    n_block: int = 0,
    k_block_m: int = 64,
    k_block_n: int = 64,
    ngroups: int = 1,
    is_causal: bool = False,
):
    """Apply the same mask logic as the CUDA tSrS/tScS loop.

    scores is a 2D tile: scores[row][col].
    The function modifies scores in-place and also returns it.
    """
    if ngroups <= 0:
        raise ValueError("ngroups must be positive")

    for row in range(len(scores)):
        for col in range(len(scores[row])):
            if not is_causal:
                if col >= seqlen_k - n_block * k_block_n:
                    scores[row][col] = NEG_INF
            else:
                global_row = m_block * k_block_m + row
                col_limit_right = (
                    seqlen_k
                    - 1
                    - n_block * k_block_n
                    - (seqlen_q - 1 - global_row) // ngroups
                )
                if col > col_limit_right:
                    scores[row][col] = NEG_INF

    return scores


def apply_attention_mask_cutlass(
    scores,
    *,
    seqlen_q: int = 64,
    seqlen_k: int = 64,
    m_block: int = 0,
    n_block: int = 0,
    k_block_m: int = 64,
    k_block_n: int = 64,
    ngroups: int = 1,
    is_causal: bool = False,
):
    """Cutlass/CuTe coordinate version of the same mask.

    This mirrors the CUDA pattern:

        cS = make_identity_tensor((kBlockM, kBlockN))
        row, col = cS[row, col]

    In the real kernel, ``thr_mma.partition_C(cS)`` changes the iteration order
    per thread. The mask decision itself is only based on the identity
    coordinates, so this Python version applies the same predicate to scores.
    """
    _prepare_tvm_ffi_import()

    import cutlass.cute as cute
    from cutlass import _mlir

    if ngroups <= 0:
        raise ValueError("ngroups must be positive")

    with _mlir.ir.Context():
        cS = cute.make_identity_tensor((k_block_m, k_block_n))
        _ = cS

        for row_idx in range(len(scores)):
            for col_idx in range(len(scores[row_idx])):
                # cS is an identity tensor, so cS[row, col] maps to (row, col).
                # Indexing it element-by-element in Python emits lots of DSL IR and
                # is much slower than the actual predicate we want to inspect.
                row = row_idx
                col = col_idx

                if not is_causal:
                    if col >= seqlen_k - n_block * k_block_n:
                        scores[row_idx][col_idx] = NEG_INF
                else:
                    col_limit_right = (
                        seqlen_k
                        - 1
                        - n_block * k_block_n
                        - (seqlen_q - 1 - (m_block * k_block_m + row)) // ngroups
                    )
                    if col > col_limit_right:
                        scores[row_idx][col_idx] = NEG_INF

    return scores



if __name__ == "__main__":
    scores = init_scores()
    apply_attention_mask_cutlass(scores, is_causal=True)
