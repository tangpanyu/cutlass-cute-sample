#include <cuda.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include "util.h"
#include "detail/cublaslt-gemm.h"
#include "detail/data.h"

#define PRINT_INFO
using namespace cute;

#define CUDA_CHECK(call)                                                   \
  do                                                                       \
  {                                                                        \
    cudaError_t err = call;                                                \
    if (err != cudaSuccess)                                                \
    {                                                                      \
      printf("CUDA error %s:%d: %s (%s)\n", __FILE__, __LINE__,            \
             cudaGetErrorString(err), #call);                              \
      exit(1);                                                             \
    }                                                                      \
  } while (0)

namespace config
{
  using namespace cute;
  // 32 不太行，64 byte 不够一个sector， 128byte 一个sector
  template <typename T_, int kTileM_ = 128, int kTileN_ = 128, int kTileK_ = 32,
            int kThreadblockSwizzleN_ = 1>
  struct GemmConfigV1
  {
    using T = T_;

    static constexpr int kTileM = kTileM_;
    static constexpr int kTileN = kTileN_;
    static constexpr int kTileK = kTileK_;
    static constexpr int kThreadblockSwizzleN = kThreadblockSwizzleN_;

    // shared memory layout
    using SmemLayoutAtom = decltype(composition(
        Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<8>{}, Int<kTileK>{}),
                    make_stride(Int<kTileK>{}, Int<1>{}))));
    using SmemLayoutA = decltype(tile_to_shape(SmemLayoutAtom{},
                                               make_shape(Int<kTileM>{}, Int<kTileK>{})));
    using SmemLayoutB = decltype(tile_to_shape(SmemLayoutAtom{},
                                               make_shape(Int<kTileN>{}, Int<kTileK>{})));

    // mma
    using mma_op = SM80_16x8x16_F16F16F16F16_TN;
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;
    using MMA = decltype(make_tiled_mma(mma_atom{},
                                        make_layout(Shape<_2, _2, _1>{}),
                                        make_layout(Shape<_1, _2, _1>{})));

    // copy from global memory to shared memory
    using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
    using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
    using g2s_copy_atom = Copy_Atom<g2s_copy_traits, T>;
    using G2SCopyA =
        decltype(make_tiled_copy(g2s_copy_atom{},
                                 make_layout(make_shape(Int<32>{}, Int<4>{}),
                                             make_stride(Int<4>{}, Int<1>{})),
                                 make_layout(make_shape(Int<1>{}, Int<8>{}))));
    using G2SCopyB = G2SCopyA;

    // copy from shared memory to register
    // use mma tiled ,so no tiled here
    using s2r_copy_op = SM75_U32x4_LDSM_N;
    using s2r_copy_traits = Copy_Traits<s2r_copy_op>;
    using s2r_copy_atom = Copy_Atom<s2r_copy_traits, T>;
    using S2RCopyAtomA = s2r_copy_atom;
    using S2RCopyAtomB = s2r_copy_atom;

    // C_shm is shared with A_shm and B_shm
    static constexpr int shm_size_AB =
        cute::cosize(SmemLayoutA{}) + cute::cosize(SmemLayoutB{});
    static constexpr int kShmSize =
        shm_size_AB * sizeof(T);
  };

  template <typename T_, int kTileM_ = 128, int kTileN_ = 128, int kTileK_ = 32,
            int kStage_ = 3, int kSmemLayoutCBatch_ = 4,
            int kThreadblockSwizzleN_ = 1>
  struct GemmConfig
  {
    using T = T_;

    static constexpr int kTileM = kTileM_;
    static constexpr int kTileN = kTileN_;
    static constexpr int kTileK = kTileK_;
    static constexpr int kStage = kStage_;

    // 这里的kSmemLayoutCBatch_表示一次经过kTileM和kTileN的SM80_16x8x16_F16F16F16F16_TNgemm结束后，会存在4个小矩阵也就是4次拷贝
    static constexpr int kSmemLayoutCBatch = kSmemLayoutCBatch_;
    static constexpr int kThreadblockSwizzleN = kThreadblockSwizzleN_;

    // shared memory layout
    using SmemLayoutAtom = decltype(composition(
        Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<8>{}, Int<kTileK>{}),
                    make_stride(Int<kTileK>{}, Int<1>{}))));
    using SmemLayoutA = decltype(tile_to_shape(SmemLayoutAtom{},
                                               make_shape(Int<kTileM>{}, Int<kTileK>{}, Int<kStage>{})));
    using SmemLayoutB = decltype(tile_to_shape(SmemLayoutAtom{},
                                               make_shape(Int<kTileN>{}, Int<kTileK>{}, Int<kStage>{})));

    // mma
    using mma_op = SM80_16x8x16_F16F16F16F16_TN;
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;
    using MMA = decltype(make_tiled_mma(mma_atom{},
                                        make_layout(Shape<_2, _2, _1>{}),
                                        make_layout(Shape<_1, _2, _1>{}))); // should obey TiledNumVal{} % AtomNumVal{} == Int<0>{}

    // copy from global memory to shared memory
    using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
    using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
    using g2s_copy_atom = Copy_Atom<g2s_copy_traits, T>;
    using G2SCopyA =
        decltype(make_tiled_copy(g2s_copy_atom{},
                                 make_layout(make_shape(Int<32>{}, Int<4>{}),
                                             make_stride(Int<4>{}, Int<1>{})),
                                 make_layout(make_shape(Int<1>{}, Int<8>{}))));
    using G2SCopyB = G2SCopyA;

    // copy from shared memory to register
    // use mma tiled ,so no tiled here
    using s2r_copy_op = SM75_U32x4_LDSM_N;
    using s2r_copy_traits = Copy_Traits<s2r_copy_op>;
    using s2r_copy_atom = Copy_Atom<s2r_copy_traits, T>;
    using S2RCopyAtomA = s2r_copy_atom;
    using S2RCopyAtomB = s2r_copy_atom;

    // (32, 32, 16) 是因为MMA中，根据MMA来计算，M=16x2, N=16x2x2，K=16,16就是MMA的基础shape,M和N是经过MMAAtom后的结果
    using MNK = typename MMA::TiledShape_MNK;
    using SmemLayoutAtomC = decltype(composition(
        Swizzle<3, 3, 3>{}, make_layout(make_shape(get<0>(MNK{}), get<1>(MNK{})),
                                        make_stride(get<1>(MNK{}), Int<1>{}))));
    using SmemLayoutC = decltype(tile_to_shape(
        SmemLayoutAtomC{},
        make_shape(get<0>(MNK{}), get<1>(MNK{}), Int<kSmemLayoutCBatch>{})));

    static_assert(size<0>(SmemLayoutA{}) * size<1>(SmemLayoutA{}) >=
                      size(SmemLayoutC{}),
                  "C shared memory request is large than A's one pipe");
    // copy from register to shared memory
    using R2SCopyAtomC = Copy_Atom<UniversalCopy<int>, T>;
    // copy from shared memory to global memory
    using S2GCopyAtomC = Copy_Atom<UniversalCopy<cute::uint128_t>, T>;
    using S2GCopyC =
        decltype(make_tiled_copy(S2GCopyAtomC{},
                                 make_layout(make_shape(Int<32>{}, Int<4>{}),
                                             make_stride(Int<4>{}, Int<1>{})),
                                 make_layout(make_shape(Int<1>{}, Int<8>{}))));

    // C_shm is shared with A_shm and B_shm
    static constexpr int shm_size_AB =
        cute::cosize(SmemLayoutA{}) + cute::cosize(SmemLayoutB{});
    static constexpr int shm_size_C = cute::cosize(SmemLayoutC{});
    static constexpr int kShmSize =
        cute::max(shm_size_AB, shm_size_C) * sizeof(T);
  };

} // namespace config

template <typename Config>
__host__ __device__ int get_swizzle_n(int tiles_n)
{
  if (Config::kThreadblockSwizzleN >= 8 && tiles_n >= 6)
  {
    return 8;
  }
  if (Config::kThreadblockSwizzleN >= 4 && tiles_n >= 3)
  {
    return 4;
  }
  if (Config::kThreadblockSwizzleN >= 2 && tiles_n >= 2)
  {
    return 2;
  }
  return 1;
}

template <typename Config>
__host__ dim3 get_swizzled_grid(int m, int n)
{
  int tiles_m = (m + Config::kTileM - 1) / Config::kTileM;
  int tiles_n = (n + Config::kTileN - 1) / Config::kTileN;
  int swizzle_n = get_swizzle_n<Config>(tiles_n);
  return dim3(tiles_m * swizzle_n, (tiles_n + swizzle_n - 1) / swizzle_n);
}

template <typename Config>
__device__ void get_swizzled_tile_coord(int m, int n, int &ix, int &iy)
{
  int tiles_m = (m + Config::kTileM - 1) / Config::kTileM;
  int tiles_n = (n + Config::kTileN - 1) / Config::kTileN;
  int swizzle_n = get_swizzle_n<Config>(tiles_n);

  iy = blockIdx.x / swizzle_n;
  ix = blockIdx.y * swizzle_n + ((blockIdx.x + iy) % swizzle_n);

  if (ix >= tiles_n || iy >= tiles_m)
  {
    ix = -1;
    iy = -1;
  }
}

double gemm_tflops(int m, int n, int k, float elapsed_ms)
{
  if (elapsed_ms <= 0.0f)
  {
    return 0.0;
  }
  return (2.0 * m * n * k) / (elapsed_ms * 1.0e9);
}

// apply shm
template <typename Config>
__global__ void
gemm_opt_shm(void *Dptr, const void *Aptr, const void *Bptr, int m, int n,
             int k)
{
  using T = typename Config::T;
  using SmemLayoutA = typename Config::SmemLayoutA;
  using SmemLayoutB = typename Config::SmemLayoutB;
  using TiledMMA = typename Config::MMA;

  using S2RCopyAtomA = typename Config::S2RCopyAtomA;
  using S2RCopyAtomB = typename Config::S2RCopyAtomB;
  using G2SCopyA = typename Config::G2SCopyA;
  using G2SCopyB = typename Config::G2SCopyB;

  constexpr int kTileM = Config::kTileM;
  constexpr int kTileN = Config::kTileN;
  constexpr int kTileK = Config::kTileK;

  extern __shared__ T shm_data[];

  T *Ashm = shm_data;
  T *Bshm = shm_data + cute::cosize(SmemLayoutA{});

  int idx = threadIdx.x;
  int ix, iy;
  get_swizzled_tile_coord<Config>(m, n, ix, iy);
  if (ix < 0)
  {
    return;
  }

  Tensor A = make_tensor(make_gmem_ptr((T *)Aptr), make_shape(m, k),
                         make_stride(k, Int<1>{})); // (M, K)
  Tensor B = make_tensor(make_gmem_ptr((T *)Bptr), make_shape(n, k),
                         make_stride(k, Int<1>{})); // (N, K)
  Tensor D = make_tensor(make_gmem_ptr((T *)Dptr), make_shape(m, n),
                         make_stride(n, Int<1>{})); // (M, N)
  // global memory
  Tensor gA = local_tile(A, make_tile(Int<kTileM>{}, Int<kTileK>{}),
                         make_coord(iy, _)); // (kTileM, kTileK, k)
  Tensor gB = local_tile(B, make_tile(Int<kTileN>{}, Int<kTileK>{}),
                         make_coord(ix, _)); // (kTileN, kTileK, k)
  Tensor gD = local_tile(D, make_tile(Int<kTileM>{}, Int<kTileN>{}),
                         make_coord(iy, ix)); // (kTileM, kTileN)

  // shared memory
  auto sA = make_tensor(make_smem_ptr(Ashm),
                        SmemLayoutA{}); // (kTileM, kTileK)
  auto sB = make_tensor(make_smem_ptr(Bshm),
                        SmemLayoutB{}); // (kTileN, kTileK)

  // register, use tiled_mma to partition register A/B/C
  TiledMMA tiled_mma;
  auto thr_mma = tiled_mma.get_slice(idx);
  auto tCrA = thr_mma.partition_fragment_A(gA(_, _, 0)); // (MMA, MMA_M, MMA_K)
  auto tCrB = thr_mma.partition_fragment_B(gB(_, _, 0)); // (MMA, MMA_N, MMA_K)
  auto tCrD = thr_mma.partition_fragment_C(gD);          // (MMA, MMA_M, MMA_N)

  auto tCgD = thr_mma.partition_C(gD); // (MMA, MMA_M, MMA_N)
  // fill zero for accumulator
  clear(tCrD);

  // from global memory to shared memory
  G2SCopyA g2s_tiled_copy_a;
  auto g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(idx);
  auto tAgA_copy = g2s_thr_copy_a.partition_S(gA); // (CPY, CPY_M, CPY_K, k)
  auto tAsA_copy =
      g2s_thr_copy_a.partition_D(sA); // (CPY, CPY_M, CPY_K)

  G2SCopyB g2s_tiled_copy_b;
  auto g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(idx);
  auto tBgB_copy = g2s_thr_copy_b.partition_S(gB); // (CPY, CPY_N, CPY_K, k)
  auto tBsB_copy =
      g2s_thr_copy_b.partition_D(sB); // (CPY, CPY_N, CPY_K)

  // from shared memory to register, use tiled_mma to generate tiled_copy
  auto s2r_tiled_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
  auto s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(idx);
  auto tAsA = s2r_thr_copy_a.partition_S(sA);     // (CPY, CPY_M, CPY_K)
  auto tCrA_view = s2r_thr_copy_a.retile_D(tCrA); // (CPY, CPY_M, CPY_K)

  auto s2r_tiled_copy_b = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
  auto s2r_thr_copy_b = s2r_tiled_copy_b.get_slice(idx);
  auto tBsB = s2r_thr_copy_b.partition_S(sB);     // (CPY, CPY_N, CPY_K)
  auto tCrB_view = s2r_thr_copy_b.retile_D(tCrB); // (CPY, CPY_N, CPY_K)

  // loop over k: i. load tile, ii. mma
  int ntile = k / kTileK;
#pragma unroll 1
  for (int itile = 0; itile < ntile; ++itile)
  {
    // copy  (CPY, CPY_M, CPY_K) , async
    cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, itile),
               tAsA_copy(_, _, _));
    cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, itile),
               tBsB_copy(_, _, _));
    cp_async_fence();

    cp_async_wait<0>();
    __syncthreads();

    int nk = size<2>(tCrA);
#pragma unroll
    for (int ik = 0; ik < nk; ++ik)
    {
      // copy  (CPY, CPY_M), sync
      cute::copy(s2r_tiled_copy_a, tAsA(_, _, ik),
                 tCrA_view(_, _, ik));
      // copy  (CPY, CPY_N)
      cute::copy(s2r_tiled_copy_b, tBsB(_, _, ik),
                 tCrB_view(_, _, ik));
      // (MMA, MMA_M) x (MMA, MMA_N) => (MMA, MMA_M, MMA_N)
      cute::gemm(tiled_mma, tCrD, tCrA(_, _, ik), tCrB(_, _, ik), tCrD);
    } // for ik
  } // itile

  // register to global memory
  cute::copy(tCrD, tCgD);
}

// one-level pipeline
// global write opt
template <typename Config>
__global__ void
gemm_opt_p1(void *Dptr, const void *Aptr, const void *Bptr, int m, int n,
            int k)
{
  using T = typename Config::T;
  using SmemLayoutA = typename Config::SmemLayoutA;
  using SmemLayoutB = typename Config::SmemLayoutB;
  using TiledMMA = typename Config::MMA;

  using S2RCopyAtomA = typename Config::S2RCopyAtomA;
  using S2RCopyAtomB = typename Config::S2RCopyAtomB;
  using G2SCopyA = typename Config::G2SCopyA;
  using G2SCopyB = typename Config::G2SCopyB;

  constexpr int kTileM = Config::kTileM;
  constexpr int kTileN = Config::kTileN;
  constexpr int kTileK = Config::kTileK;
  constexpr int kStage = Config::kStage;

  extern __shared__ T shm_data[];

  T *Ashm = shm_data;
  T *Bshm = shm_data + cute::cosize(SmemLayoutA{});

  int idx = threadIdx.x;
  int ix, iy;
  get_swizzled_tile_coord<Config>(m, n, ix, iy);
  if (ix < 0)
  {
    return;
  }

  Tensor A = make_tensor(make_gmem_ptr((T *)Aptr), make_shape(m, k),
                         make_stride(k, Int<1>{})); // (M, K)
  Tensor B = make_tensor(make_gmem_ptr((T *)Bptr), make_shape(n, k),
                         make_stride(k, Int<1>{})); // (N, K)
  Tensor D = make_tensor(make_gmem_ptr((T *)Dptr), make_shape(m, n),
                         make_stride(n, Int<1>{})); // (M, N)
  // global memory
  Tensor gA = local_tile(A, make_tile(Int<kTileM>{}, Int<kTileK>{}),
                         make_coord(iy, _)); // (kTileM, kTileK, k)
  Tensor gB = local_tile(B, make_tile(Int<kTileN>{}, Int<kTileK>{}),
                         make_coord(ix, _)); // (kTileN, kTileK, k)
  Tensor gD = local_tile(D, make_tile(Int<kTileM>{}, Int<kTileN>{}),
                         make_coord(iy, ix)); // (kTileM, kTileN)

  // shared memory
  auto sA = make_tensor(make_smem_ptr(Ashm),
                        SmemLayoutA{}); // (kTileM, kTileK, kStage)
  auto sB = make_tensor(make_smem_ptr(Bshm),
                        SmemLayoutB{}); // (kTileN, kTileK, kStage)

  // register, use tiled_mma to partition register A/B/C
  TiledMMA tiled_mma;
  auto thr_mma = tiled_mma.get_slice(idx);
  auto tCrA = thr_mma.partition_fragment_A(gA(_, _, 0)); // (MMA, MMA_M, MMA_K)
  auto tCrB = thr_mma.partition_fragment_B(gB(_, _, 0)); // (MMA, MMA_N, MMA_K)
  auto tCrD = thr_mma.partition_fragment_C(gD);          // (MMA, MMA_M, MMA_N)
  // fill zero for accumulator
  clear(tCrD);

  auto tCgD = thr_mma.partition_C(gD); // (MMA, MMA_M, MMA_N)

  // from global memory to shared memory
  G2SCopyA g2s_tiled_copy_a;
  auto g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(idx);
  auto tAgA_copy = g2s_thr_copy_a.partition_S(gA); // (CPY, CPY_M, CPY_K, k)
  auto tAsA_copy =
      g2s_thr_copy_a.partition_D(sA); // (CPY, CPY_M, CPY_K, kStage)

  G2SCopyB g2s_tiled_copy_b;
  auto g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(idx);
  auto tBgB_copy = g2s_thr_copy_b.partition_S(gB); // (CPY, CPY_N, CPY_K, k)
  auto tBsB_copy =
      g2s_thr_copy_b.partition_D(sB); // (CPY, CPY_N, CPY_K, kStage)

  // from shared memory to register, use tiled_mma to generate tiled_copy
  auto s2r_tiled_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
  auto s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(idx);
  auto tAsA = s2r_thr_copy_a.partition_S(sA);     // (CPY, CPY_M, CPY_K, kStage)
  auto tCrA_view = s2r_thr_copy_a.retile_D(tCrA); // (CPY, CPY_M, CPY_K)

  auto s2r_tiled_copy_b = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
  auto s2r_thr_copy_b = s2r_tiled_copy_b.get_slice(idx);
  auto tBsB = s2r_thr_copy_b.partition_S(sB);     // (CPY, CPY_N, CPY_K, kStage)
  auto tCrB_view = s2r_thr_copy_b.retile_D(tCrB); // (CPY, CPY_N, CPY_K)

  // global -> shm, [0, k / kTileK]
  int itile_to_read = 0;
  // shm -> register, [0, kStage-1]
  int ismem_read = 0;
  // global -> shm, [0, kStage-1]
  int ismem_write = 0;

  // submit kStage - 1 tile
  // gmem -> shm
#pragma unroll
  for (int istage = 0; istage < kStage - 1; ++istage)
  {
    // copy  (CPY, CPY_M, CPY_K), asynchronous, thread-level
    copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, istage),
         tAsA_copy(_, _, _, istage));
    copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, istage),
         tBsB_copy(_, _, _, istage));
    cp_async_fence();

    ++itile_to_read;
    ++ismem_write;
  }

  // wait one submitted gmem->smem done
  cp_async_wait<kStage - 2>();
  // wait all threads in one warp complete
  __syncthreads();

  // loop over k: i. load tile, ii. mma
  int ntile = k / kTileK;

#pragma unroll 1
  for (int itile = 0; itile < ntile; ++itile)
  {
    int nk = size<2>(tCrA);

#pragma unroll
    for (int ik = 0; ik < nk; ++ik)
    {
      // shm -> reg s[itile][ik] -> r[ik]
      // copy  (CPY, CPY_M), use in next iteration ,sync
      cute::copy(s2r_tiled_copy_a, tAsA(_, _, ik, ismem_read),
                 tCrA_view(_, _, ik));
      cute::copy(s2r_tiled_copy_b, tBsB(_, _, ik, ismem_read),
                 tCrB_view(_, _, ik));
      // (MMA, MMA_M) x (MMA, MMA_N) => (MMA, MMA_M, MMA_N)
      cute::gemm(tiled_mma, tCrD, tCrA(_, _, ik), tCrB(_, _, ik), tCrD);

      if (ik == nk - 1)
      {
        cp_async_wait<kStage - 2>();
        __syncthreads();

        ismem_read = (ismem_read + 1) % kStage;
      }

      if (ik == 0)
      {
        if (itile_to_read < ntile)
        {
          // copy  (CPY, CPY_M, CPY_K)
          cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, itile_to_read),
                     tAsA_copy(_, _, _, ismem_write));
          cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, itile_to_read),
                     tBsB_copy(_, _, _, ismem_write));

          ++itile_to_read;
          ismem_write = (ismem_write + 1) % kStage;
        }

        cp_async_fence();
      }

    } // for ik
  } // itile

  // register to global memory
  cute::copy(tCrD, tCgD);
}

// two-level pipelne
template <typename Config>
__global__ void
gemm_opt_p2(void *Dptr, const void *Aptr, const void *Bptr, int m, int n,
            int k)
{
  using T = typename Config::T;
  using SmemLayoutA = typename Config::SmemLayoutA;
  using SmemLayoutB = typename Config::SmemLayoutB;
  using SmemLayoutC = typename Config::SmemLayoutC;
  using TiledMMA = typename Config::MMA;

  using S2RCopyAtomA = typename Config::S2RCopyAtomA;
  using S2RCopyAtomB = typename Config::S2RCopyAtomB;
  using G2SCopyA = typename Config::G2SCopyA;
  using G2SCopyB = typename Config::G2SCopyB;
  using R2SCopyAtomC = typename Config::R2SCopyAtomC;
  using S2GCopyAtomC = typename Config::S2GCopyAtomC;
  using S2GCopyC = typename Config::S2GCopyC;

  constexpr int kTileM = Config::kTileM;
  constexpr int kTileN = Config::kTileN;
  constexpr int kTileK = Config::kTileK;
  constexpr int kStage = Config::kStage;

  // max(A+B, C)
  extern __shared__ T shm_data[];

  T *Ashm = shm_data;
  T *Bshm = shm_data + cute::cosize(SmemLayoutA{});

  int idx = threadIdx.x;
  int ix, iy;
  get_swizzled_tile_coord<Config>(m, n, ix, iy);
  if (ix < 0)
  {
    return;
  }

  Tensor A = make_tensor(make_gmem_ptr((T *)Aptr), make_shape(m, k),
                         make_stride(k, Int<1>{})); // (M, K)
  Tensor B = make_tensor(make_gmem_ptr((T *)Bptr), make_shape(n, k),
                         make_stride(k, Int<1>{})); // (N, K)
  Tensor D = make_tensor(make_gmem_ptr((T *)Dptr), make_shape(m, n),
                         make_stride(n, Int<1>{})); // (M, N)
  // global memory
  Tensor gA = local_tile(A, make_tile(Int<kTileM>{}, Int<kTileK>{}),
                         make_coord(iy, _)); // (kTileM, kTileK, k)
  Tensor gB = local_tile(B, make_tile(Int<kTileN>{}, Int<kTileK>{}),
                         make_coord(ix, _)); // (kTileN, kTileK, k)
  Tensor gD = local_tile(D, make_tile(Int<kTileM>{}, Int<kTileN>{}),
                         make_coord(iy, ix)); // (kTileM, kTileN)

  // shared memory
  auto sA = make_tensor(make_smem_ptr(Ashm),
                        SmemLayoutA{}); // (kTileM, kTileK, kStage)
  auto sB = make_tensor(make_smem_ptr(Bshm),
                        SmemLayoutB{}); // (kTileN, kTileK, kStage)

  // register, use tiled_mma to partition register A/B/C
  TiledMMA tiled_mma;
  auto thr_mma = tiled_mma.get_slice(idx);
  auto tCrA = thr_mma.partition_fragment_A(gA(_, _, 0)); // (MMA, MMA_M, MMA_K)
  auto tCrB = thr_mma.partition_fragment_B(gB(_, _, 0)); // (MMA, MMA_N, MMA_K)
  auto tCrD = thr_mma.partition_fragment_C(gD);          // (MMA, MMA_M, MMA_N)
  // fill zero for accumulator
  clear(tCrD);

  auto tCgD = thr_mma.partition_C(gD); // (MMA, MMA_M, MMA_N)

  // from global memory to shared memory
  G2SCopyA g2s_tiled_copy_a;
  auto g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(idx);
  auto tAgA_copy = g2s_thr_copy_a.partition_S(gA); // (CPY, CPY_M, CPY_K, k)
  auto tAsA_copy =
      g2s_thr_copy_a.partition_D(sA); // (CPY, CPY_M, CPY_K, kStage)

  G2SCopyB g2s_tiled_copy_b;
  auto g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(idx);
  auto tBgB_copy = g2s_thr_copy_b.partition_S(gB); // (CPY, CPY_N, CPY_K, k)
  auto tBsB_copy =
      g2s_thr_copy_b.partition_D(sB); // (CPY, CPY_N, CPY_K, kStage)

  // from shared memory to register, use tiled_mma to generate tiled_copy
  auto s2r_tiled_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
  auto s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(idx);
  auto tAsA = s2r_thr_copy_a.partition_S(sA);     // (CPY, CPY_M, CPY_K, kStage)
  auto tCrA_view = s2r_thr_copy_a.retile_D(tCrA); // (CPY, CPY_M, CPY_K)

  auto s2r_tiled_copy_b = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
  auto s2r_thr_copy_b = s2r_tiled_copy_b.get_slice(idx);
  auto tBsB = s2r_thr_copy_b.partition_S(sB);     // (CPY, CPY_N, CPY_K, kStage)
  auto tCrB_view = s2r_thr_copy_b.retile_D(tCrB); // (CPY, CPY_N, CPY_K)

  // global -> shm, [0, k / kTileK]
  int itile_to_read = 0;
  // shm -> register, [0, kStage-1]
  int ismem_read = 0;
  // global -> shm, [0, kStage-1]
  int ismem_write = 0;

  // submit kStage - 1 tile
  // gmem -> shm
#pragma unroll
  for (int istage = 0; istage < kStage - 1; ++istage)
  {
    // copy  (CPY, CPY_M, CPY_K), asynchronous, thread-level
    copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, istage),
         tAsA_copy(_, _, _, istage));
    copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, istage),
         tBsB_copy(_, _, _, istage));
    cp_async_fence();

    ++itile_to_read;
    ++ismem_write;
  }

  // wait one submitted gmem->smem done
  cp_async_wait<kStage - 2>();
  // wait all threads in one warp complete
  __syncthreads();

  int ik = 0;
  // smem -> reg
  // copy  (CPY, CPY_M) ,sync
  copy(s2r_tiled_copy_a, tAsA(_, _, ik, ismem_read), tCrA_view(_, _, ik));
  // copy  (CPY, CPY_N) ,sync
  copy(s2r_tiled_copy_b, tBsB(_, _, ik, ismem_read), tCrB_view(_, _, ik));

  // loop over k: i. load tile, ii. mma
  int ntile = k / kTileK;

#pragma unroll 1
  for (int itile = 0; itile < ntile; ++itile)
  {
    int nk = size<2>(tCrA);

#pragma unroll
    for (int ik = 0; ik < nk; ++ik)
    {
      int ik_next = (ik + 1) % nk;

      if (ik == nk - 1)
      {
        cp_async_wait<kStage - 2>();
        __syncthreads();

        ismem_read = (ismem_read + 1) % kStage;
      }

      // shm -> reg s[itile][ik + 1] -> r[ik + 1]
      // copy  (CPY, CPY_M), use in next iteration ,sync
      cute::copy(s2r_tiled_copy_a, tAsA(_, _, ik_next, ismem_read),
                 tCrA_view(_, _, ik_next));
      cute::copy(s2r_tiled_copy_b, tBsB(_, _, ik_next, ismem_read),
                 tCrB_view(_, _, ik_next));

      if (ik == 0)
      {
        if (itile_to_read < ntile)
        {
          // copy  (CPY, CPY_M, CPY_K)
          cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, itile_to_read),
                     tAsA_copy(_, _, _, ismem_write));
          cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, itile_to_read),
                     tBsB_copy(_, _, _, ismem_write));

          ++itile_to_read;
          ismem_write = (ismem_write + 1) % kStage;
        }

        cp_async_fence();
      }
      // instruction reordering
      // (MMA, MMA_M) x (MMA, MMA_N) => (MMA, MMA_M, MMA_N)
      cute::gemm(tiled_mma, tCrD, tCrA(_, _, ik), tCrB(_, _, ik), tCrD);
    } // for ik
  } // itile

  // register to global memory
  cute::copy(tCrD, tCgD);
}

// global write opt
template <typename Config>
__global__ void
gemm_opt_final(void *Dptr, const void *Aptr, const void *Bptr, int m, int n,
               int k)
{
  using T = typename Config::T;
  using SmemLayoutA = typename Config::SmemLayoutA;
  using SmemLayoutB = typename Config::SmemLayoutB;
  using SmemLayoutC = typename Config::SmemLayoutC;
  using TiledMMA = typename Config::MMA;

  using S2RCopyAtomA = typename Config::S2RCopyAtomA;
  using S2RCopyAtomB = typename Config::S2RCopyAtomB;
  using G2SCopyA = typename Config::G2SCopyA;
  using G2SCopyB = typename Config::G2SCopyB;
  using R2SCopyAtomC = typename Config::R2SCopyAtomC;
  using S2GCopyAtomC = typename Config::S2GCopyAtomC;
  using S2GCopyC = typename Config::S2GCopyC;

  constexpr int kTileM = Config::kTileM;
  constexpr int kTileN = Config::kTileN;
  constexpr int kTileK = Config::kTileK;
  constexpr int kStage = Config::kStage;

  // max(A+B, C)
  extern __shared__ T shm_data[];

  T *Ashm = shm_data;
  T *Bshm = shm_data + cute::cosize(SmemLayoutA{});

  int idx = threadIdx.x;
  int ix, iy;
  get_swizzled_tile_coord<Config>(m, n, ix, iy);
  if (ix < 0)
  {
    return;
  }

  Tensor A = make_tensor(make_gmem_ptr((T *)Aptr), make_shape(m, k),
                         make_stride(k, Int<1>{})); // (M, K)
  Tensor B = make_tensor(make_gmem_ptr((T *)Bptr), make_shape(n, k),
                         make_stride(k, Int<1>{})); // (N, K)
  Tensor D = make_tensor(make_gmem_ptr((T *)Dptr), make_shape(m, n),
                         make_stride(n, Int<1>{})); // (M, N)
  // global memory
  Tensor gA = local_tile(A, make_tile(Int<kTileM>{}, Int<kTileK>{}),
                         make_coord(iy, _)); // (kTileM, kTileK, k)
  Tensor gB = local_tile(B, make_tile(Int<kTileN>{}, Int<kTileK>{}),
                         make_coord(ix, _)); // (kTileN, kTileK, k)
  Tensor gD = local_tile(D, make_tile(Int<kTileM>{}, Int<kTileN>{}),
                         make_coord(iy, ix)); // (kTileM, kTileN)

  // shared memory
  auto sA = make_tensor(make_smem_ptr(Ashm),
                        SmemLayoutA{}); // (kTileM, kTileK, kStage)
  auto sB = make_tensor(make_smem_ptr(Bshm),
                        SmemLayoutB{}); // (kTileN, kTileK, kStage)

  // register, use tiled_mma to partition register A/B/C
  TiledMMA tiled_mma;
  auto thr_mma = tiled_mma.get_slice(idx);
  auto tCrA = thr_mma.partition_fragment_A(gA(_, _, 0)); // (MMA, MMA_M, MMA_K)
  auto tCrB = thr_mma.partition_fragment_B(gB(_, _, 0)); // (MMA, MMA_N, MMA_K)
  auto tCrD = thr_mma.partition_fragment_C(gD);          // (MMA, MMA_M, MMA_N)
  // fill zero for accumulator
  clear(tCrD);

  // from global memory to shared memory
  G2SCopyA g2s_tiled_copy_a;
  auto g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(idx);
  auto tAgA_copy = g2s_thr_copy_a.partition_S(gA); // (CPY, CPY_M, CPY_K, k)
  auto tAsA_copy =
      g2s_thr_copy_a.partition_D(sA); // (CPY, CPY_M, CPY_K, kStage)

  G2SCopyB g2s_tiled_copy_b;
  auto g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(idx);
  auto tBgB_copy = g2s_thr_copy_b.partition_S(gB); // (CPY, CPY_N, CPY_K, k)
  auto tBsB_copy =
      g2s_thr_copy_b.partition_D(sB); // (CPY, CPY_N, CPY_K, kStage)

  // from shared memory to register, use tiled_mma to generate tiled_copy
  auto s2r_tiled_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
  auto s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(idx);
  auto tAsA = s2r_thr_copy_a.partition_S(sA);     // (CPY, CPY_M, CPY_K, kStage)
  auto tCrA_view = s2r_thr_copy_a.retile_D(tCrA); // (CPY, CPY_M, CPY_K)

  auto s2r_tiled_copy_b = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
  auto s2r_thr_copy_b = s2r_tiled_copy_b.get_slice(idx);
  auto tBsB = s2r_thr_copy_b.partition_S(sB);     // (CPY, CPY_N, CPY_K, kStage)
  auto tCrB_view = s2r_thr_copy_b.retile_D(tCrB); // (CPY, CPY_N, CPY_K)

  // global -> shm, [0, k / kTileK]
  int itile_to_read = 0;
  // shm -> register, [0, kStage-1]
  int ismem_read = 0;
  // global -> shm, [0, kStage-1]
  int ismem_write = 0;

  // submit kStage - 1 tile
  // gmem -> shm
#pragma unroll
  for (int istage = 0; istage < kStage - 1; ++istage)
  {
    // copy  (CPY, CPY_M, CPY_K), asynchronous, thread-level
    copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, istage),
         tAsA_copy(_, _, _, istage));
    copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, istage),
         tBsB_copy(_, _, _, istage));
    cp_async_fence();

    ++itile_to_read;
    ++ismem_write;
  }

  // wait one submitted gmem->smem done
  cp_async_wait<kStage - 2>();
  // wait all threads in one warp complete
  __syncthreads();

  int ik = 0;
  // smem -> reg
  // copy  (CPY, CPY_M) ,sync
  copy(s2r_tiled_copy_a, tAsA(_, _, ik, ismem_read), tCrA_view(_, _, ik));
  // copy  (CPY, CPY_N) ,sync
  copy(s2r_tiled_copy_b, tBsB(_, _, ik, ismem_read), tCrB_view(_, _, ik));

  // loop over k: i. load tile, ii. mma
  int ntile = k / kTileK;

#ifdef PRINT_INFO
  /*
    一句话总结：
    MMA和CPY表示一个线程所有的基本数据的Shape，MMA表示寄存器的数据Shape，CPY表示拷贝的基本单元数据Shape
    _M,_N,_K表示重复次数，Warp的MMA需要在该方向循环多少次，线程的CPY需要在该方向循环多少次

    MMA就是表示一个线程持有数据的Shape，比如(_2,_2,_2)表示MMA中一个线程有8个数据，不用管Stride
    MMA_M表示Block 的 M 方向包含多少个 tiled-MMA tile。
    比如这里是4,表示128（kTileM) / 32(128个线程的一次MMA的M维度，在M方向有2个warp，也就是2x16=32) = 4
    MMA_K表示kTileK/ 16(一个SM80_16x8x16_F16F16F16F16_TN MMA的k维度 = 16) = 2
    MMA_N则表示kTileN / 8(base), 这里因为是在N上重复了一次，所以除以16,就等于128/ 16 = 8
    而tCrD的MMA是(_2,_2)，则表示16x8,每个线程有4个数据，Shape是(_2,_2)，Stride不管

    CPY就是要拷贝的基本shape,每个的shape都不同，比如(8, 1)就是k-major的8个数据，每个线程拷贝的数量
    CPY_M就是一个Atom要在kTileM上拷贝几次才拷贝完，也可以理解成一个线程要拷贝几次CPY才拷贝完这次需要的数据
    比如这里拷贝一次的32x32的shape,kTileM是128,也就是在kTileM拷贝4次
    CPY_K和CPY_N和上同理

    其中k是 K / kTileK，这里A和B的k不一样因为上面是作者的维度，也就是1024,我改写成4096了，也就是 1024 / 32 =32
    kStage就是预先规定好的，这里是3

    tCrD的(_2,_2)表示16x8的Atom Shape，一个线程保留的数据Shape就是(2, 2)

    这里是选gA的0的原因，因为值需要mma 计算一次的一组寄存器数量，而不是把所有的gA都加载到片上的寄存器数量
    也就是说，覆盖了M和K方向上循环的寄存器
    tCrA = thr_mma.partition_fragment_A(gA(_, _, 0))  (MMA, MMA_M, MMA_K) : ((_2,_2,_2),_4,_2)
    128 / 32 = 4, 32 / 32 = 1 ，k看下面
    tCrA = thr_mma.partition_fragment_A(gA(_, _, 0))  (MMA, MMA_M, MMA_K) : ((_2,_2,_2),_4,_2)
    tAgA_copy = g2s_thr_copy_a.partition_S(gA)  (CPY, CPY_M, CPY_K, k) : ((_8,_1),_4,_1,128)
    tAsA_copy = g2s_thr_copy_a.partition_D(sA)  (CPY, CPY_M, CPY_K, kStage) : ((_8,_1),_4,_1,_3)
    tAsA = s2r_thr_copy_a.partition_S(sA)  (CPY, CPY_M, CPY_K, kStage) : ((_8,_1),_4,_2,_3)
    tCrA_view = s2r_thr_copy_a.retile_D(tCrA) (CPY, CPY_M, CPY_K) : ((_8,_1),_4,_2)

    tCrB = thr_mma.partition_fragment_B(gB(_, _, 0))  (MMA, MMA_N, MMA_K) : ((_2,_2),_8,_2)
    上面32是因为shape不同,4096维度都是128,也就是4096/32, k表示的是想循环次数
    tBgB_copy = g2s_thr_copy_b.partition_S(gB)  (CPY, CPY_M, CPY_K, k) : ((_8,_1),_4,_1,128)
    tBsB_copy = g2s_thr_copy_b.partition_D(sB)  (CPY, CPY_M, CPY_K, kStage) : ((_8,_1),_4,_1,_3)
    tBsB = s2r_thr_copy_b.partition_S(sB)  (CPY, CPY_M, CPY_K, kStage) : ((_8,_1),_4,_2,_3)
    tCrB_view = s2r_thr_copy_b.retile_D(tCrB) (CPY, CPY_M, CPY_K) : ((_8,_1),_4,_2)

    tCrD = thr_mma.partition_fragment_C(gD); (MMA, MMA_M, MMA_N) : ((_2,_2),_4,_8)

    tCrA.stride = thr_mma.partition_fragment_A(gA(_, _, 0))  (MMA, MMA_M, MMA_K) : ((_1,_2,_4),_8,_32)
    tCrB.stride = thr_mma.partition_fragment_B(gB(_, _, 0))  (MMA, MMA_M, MMA_K) : ((_1,_2),_4,_32)
    tCrD.stride = thr_mma.partition_fragment_C(gD); (MMA, MMA_M, MMA_N) : ((_1,_2),_4,_16)
  */
  if (threadIdx.x == 0 && ix == 0 && iy == 0)
  {
    PRINT("tCrA = thr_mma.partition_fragment_A(gA(_, _, 0))  (MMA, MMA_M, MMA_K)", tCrA.shape());
    PRINT("tAgA_copy = g2s_thr_copy_a.partition_S(gA)  (CPY, CPY_M, CPY_K, k)", tAgA_copy.shape());
    PRINT("tAsA_copy = g2s_thr_copy_a.partition_D(sA)  (CPY, CPY_M, CPY_K, kStage)", tAsA_copy.shape());
    PRINT("tAsA = s2r_thr_copy_a.partition_S(sA)  (CPY, CPY_M, CPY_K, kStage)", tAsA.shape());
    PRINT("tCrA_view = s2r_thr_copy_a.retile_D(tCrA) (CPY, CPY_M, CPY_K)", tCrA_view.shape());

    PRINT("tCrB = thr_mma.partition_fragment_B(gB(_, _, 0))  (MMA, MMA_M, MMA_K)", tCrB.shape());
    PRINT("tBgB_copy = g2s_thr_copy_b.partition_S(gB)  (CPY, CPY_M, CPY_K, k)", tBgB_copy.shape());
    PRINT("tBsB_copy = g2s_thr_copy_b.partition_D(sB)  (CPY, CPY_M, CPY_K, kStage)", tBsB_copy.shape());
    PRINT("tBsB = s2r_thr_copy_b.partition_S(sB)  (CPY, CPY_M, CPY_K, kStage)", tBsB.shape());
    PRINT("tCrB_view = s2r_thr_copy_b.retile_D(tCrB) (CPY, CPY_M, CPY_K)", tCrB_view.shape());

    PRINT("tCrD = thr_mma.partition_fragment_C(gD); (MMA, MMA_M, MMA_N)", tCrD.shape());

    PRINT("tCrA.stride = thr_mma.partition_fragment_A(gA(_, _, 0))  (MMA, MMA_M, MMA_K)", tCrA.stride());
    PRINT("tCrB.stride = thr_mma.partition_fragment_B(gB(_, _, 0))  (MMA, MMA_M, MMA_K)", tCrB.stride());
    PRINT("tCrD.stride = thr_mma.partition_fragment_C(gD); (MMA, MMA_M, MMA_N)", tCrD.stride());
  }
#endif
#pragma unroll 1
  for (int itile = 0; itile < ntile; ++itile)
  {
    int nk = size<2>(tCrA);

#pragma unroll
    for (int ik = 0; ik < nk; ++ik)
    {
      int ik_next = (ik + 1) % nk;

      if (ik == nk - 1)
      {
        cp_async_wait<kStage - 2>();
        __syncthreads();

        ismem_read = (ismem_read + 1) % kStage;
      }

      // shm -> reg s[itile][ik + 1] -> r[ik + 1]
      // copy  (CPY, CPY_M), use in next iteration ,sync
      cute::copy(s2r_tiled_copy_a, tAsA(_, _, ik_next, ismem_read),
                 tCrA_view(_, _, ik_next));
      cute::copy(s2r_tiled_copy_b, tBsB(_, _, ik_next, ismem_read),
                 tCrB_view(_, _, ik_next));

      if (ik == 0)
      {
        if (itile_to_read < ntile)
        {
          // copy  (CPY, CPY_M, CPY_K)
          cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, itile_to_read),
                     tAsA_copy(_, _, _, ismem_write));
          cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, itile_to_read),
                     tBsB_copy(_, _, _, ismem_write));

          ++itile_to_read;
          ismem_write = (ismem_write + 1) % kStage;
        }

        cp_async_fence();
      }
      // instruction reordering
      // (MMA, MMA_M) x (MMA, MMA_N) => (MMA, MMA_M, MMA_N)
      cute::gemm(tiled_mma, tCrD, tCrA(_, _, ik), tCrB(_, _, ik), tCrD);
    } // for ik
  } // itile

  // use less shared memory as a scratchpad tile to use large wide instuction
  // Dreg -> shm -> global
  // (get<0>(MNK{}), get<1>(MNK{}), Int<kSmemLayoutCBatch>{})
  auto sC = make_tensor(sA(_, _, ismem_read).data(), SmemLayoutC{});

  auto r2s_tiled_copy_c = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma);
  auto r2s_thr_copy_c = r2s_tiled_copy_c.get_slice(idx);
  // retile相当于Pytorch的view,重新换个shape和stride,S和D类似，就是source和destination
  auto tCrC_r2s = r2s_thr_copy_c.retile_S(tCrD);  // (CPY, CPY_M, CPY_N)
  auto tCsC_r2s = r2s_thr_copy_c.partition_D(sC); // (CPY, _1, _1, pipe)

  S2GCopyC s2g_tiled_copy_c;
  auto s2g_thr_copy_c = s2g_tiled_copy_c.get_thread_slice(idx);
  auto tCsC_s2g = s2g_thr_copy_c.partition_S(sC); // (CPY, _1, _1, pipe)
  auto tCgC_s2g = s2g_thr_copy_c.partition_D(gD); // (CPY, CPY_M, CPY_N)

  auto tCgC_s2gx = group_modes<1, 3>(tCgC_s2g); // (CPY_, CPY_MN)
  auto tCrC_r2sx = group_modes<1, 3>(tCrC_r2s); // (CPY_, CPY_MN)

#ifdef PRINT_INFO
  /*
    (_2,(_2,_2)第一个2是2个half一个int,（2,2）表示线程的数据排布是2行2列，根据mmac的layout来的，也就是r2c,r的CPY，拷贝单元
    CPY_M表示最终的kTileM x kTileN的结果矩阵，在M维度一个线程需要拷贝4次，
    比如128(kTileM) / 32(MMA的基本矩阵，32x32的矩阵，一个线程是8个数据) = 4
    CPY_N同理
    sC : (_32,_32,_4)
    tCrC_r2s = r2s_thr_copy_c.retile_S(tCrD)  (CPY, CPY_M, CPY_N) : ((_2,(_2,_2)),_4,_4)
    tCsC_r2s = r2s_thr_copy_c.partition_D(sC)  (CPY, _1, _1, pipe) : ((_2,(_2,_2)),_1,_1,_4)
    tCsC_s2g = s2g_thr_copy_c.partition_S(sC)  (CPY, _1, _1, pipe) : ((_8,_1),_1,_1,_4)
    tCgC_s2g = s2g_thr_copy_c.partition_D(gD)  (CPY, CPY_M, CPY_N) : ((_8,_1),_4,_4)
    tCgC_s2gx = group_modes<1, 3>(tCgC_s2g) (CPY_, CPY_MN) : ((_8,_1),(_4,_4))
    tCrC_r2sx = group_modes<1, 3>(tCrC_r2s) (CPY_, CPY_MN) : ((_2,(_2,_2)),(_4,_4))
    tiled_mma::TiledShape_MNK  : (_32,_32,_16)
  */
  if (threadIdx.x == 0 && ix == 0 && iy == 0)
  {
    PRINT("sC", sC.shape());
    PRINT("tCrC_r2s = r2s_thr_copy_c.retile_S(tCrD)  (CPY, CPY_M, CPY_N)", tCrC_r2s.shape());
    PRINT("tCsC_r2s = r2s_thr_copy_c.partition_D(sC)  (CPY, _1, _1, pipe)", tCsC_r2s.shape());
    PRINT("tCsC_s2g = s2g_thr_copy_c.partition_S(sC)  (CPY, _1, _1, pipe)", tCsC_s2g.shape());
    PRINT("tCgC_s2g = s2g_thr_copy_c.partition_D(gD)  (CPY, CPY_M, CPY_N)", tCgC_s2g.shape());
    PRINT("tCgC_s2gx = group_modes<1, 3>(tCgC_s2g) (CPY_, CPY_MN)", tCgC_s2gx.shape());
    PRINT("tCrC_r2sx = group_modes<1, 3>(tCrC_r2s) (CPY_, CPY_MN)", tCrC_r2sx.shape());
    PRINT("tiled_mma::TiledShape_MNK (M, N, K)", typename TiledMMA::TiledShape_MNK{});
  }
#endif

  int step = size<3>(tCsC_r2s);
#pragma unroll
  for (int i = 0; i < size<1>(tCrC_r2sx); i += step)
  {
    // reg -> shm
#pragma unroll
    for (int j = 0; j < step; ++j)
    {
      // (_2,(_2,_2))
      cute::copy(r2s_tiled_copy_c, tCrC_r2sx(_, i + j), tCsC_r2s(_, 0, 0, j));
    }
    __syncthreads();

#pragma unroll
    // shm -> global
    for (int j = 0; j < step; ++j)
    {
      // (_8,_1)
      cute::copy(s2g_tiled_copy_c, tCsC_s2g(_, 0, 0, j), tCgC_s2gx(_, i + j));
    }

    __syncthreads();
  }

}

int main(int argc, char *argv[])
{
  std::string algo = "final";
  if (argc > 1)
  {
    algo = argv[1];
  }
  using T = cute::half_t;
  cudaEvent_t start, end;
  float elapsedTime;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  srand(1000);

  int M = 4096;
  int N = 4096;
  int K = 4096;

  int count = 100;
  int warmup = 5;
  int compare_start = 1024;
  int compare_end = 8192;
  int compare_step = 1024;

  if (algo == "compare4096" || algo == "compare")
  {
    if (argc > 2)
    {
      compare_start = atoi(argv[2]);
      compare_end = compare_start;
    }
    if (argc > 3)
    {
      compare_end = atoi(argv[3]);
    }
    if (argc > 4)
    {
      compare_step = atoi(argv[4]);
    }
    if (argc > 5)
    {
      count = atoi(argv[5]);
    }
    if (argc > 6)
    {
      warmup = atoi(argv[6]);
    }
    M = compare_end;
    N = compare_end;
    K = compare_end;
  }

  T *Aptr;
  T *Bptr;
  T *Dptr;
  cudaMalloc(&Aptr, sizeof(T) * M * K);
  cudaMalloc(&Bptr, sizeof(T) * N * K);
  cudaMalloc(&Dptr, sizeof(T) * M * N);

  T *Aptr_host;
  T *Bptr_host;
  Aptr_host = (T *)malloc(sizeof(T) * M * K);
  Bptr_host = (T *)malloc(sizeof(T) * N * K);
  auto tA = make_tensor(Aptr_host, make_shape(M, K), make_stride(K, 1));
  auto tB = make_tensor(Bptr_host, make_shape(N, K), make_stride(K, 1));
  cpu_rand_data(&tA);
  cpu_rand_data(&tB);
  cudaMemcpy(Aptr, Aptr_host, sizeof(T) * M * K, cudaMemcpyHostToDevice);
  cudaMemcpy(Bptr, Bptr_host, sizeof(T) * N * K, cudaMemcpyHostToDevice);

  // print(typename decltype(gemm_config)::MMA{});
  // print(typename decltype(gemm_config)::SmemLayoutA{});
  if (algo == "compare4096" || algo == "compare")
  {
    config::GemmConfig<T, 128, 128, 32, 3, 4, 4> gemm_config;
    dim3 block(size(decltype(gemm_config)::MMA{}));
    int shm_size = gemm_config.kShmSize;
    CUDA_CHECK(cudaFuncSetAttribute(gemm_opt_final<decltype(gemm_config)>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    shm_size));

    printf("GEMM compare sweep: count=%d, warmup=%d\n", count, warmup);
    printf("%8s %12s %12s %14s %14s %8s %12s\n", "SIZE", "final_ms",
           "cublas_ms", "final_TFLOPS", "cublas_TFLOPS", "algo",
           "achieve");

    for (int size = compare_start; size <= compare_end; size += compare_step)
    {
      M = size;
      N = size;
      K = size;
      dim3 grid = get_swizzled_grid<decltype(gemm_config)>(M, N);

      CUDA_CHECK(cudaMemset(Dptr, 0, sizeof(T) * M * N));
      for (int it = 0; it < warmup; ++it)
      {
        gemm_opt_final<decltype(gemm_config)>
            <<<grid, block, shm_size>>>(Dptr, Aptr, Bptr, M, N, K);
      }
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());

      CUDA_CHECK(cudaEventRecord(start));
      for (int it = 0; it < count; ++it)
      {
        gemm_opt_final<decltype(gemm_config)>
            <<<grid, block, shm_size>>>(Dptr, Aptr, Bptr, M, N, K);
      }
      CUDA_CHECK(cudaEventRecord(end));
      CUDA_CHECK(cudaEventSynchronize(end));
      CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start, end));
      double final_ms = elapsedTime / count;
      double final_tflops = gemm_tflops(M, N, K, final_ms);

      CUDA_CHECK(cudaMemset(Dptr, 0, sizeof(T) * M * N));
      CublasLtGemm<T> cublaslt_gemm;
      cublaslt_gemm.init(Dptr, Aptr, Bptr, M, N, K);

      int cublaslt_best_algo = -1;
      double cublaslt_ms = 1.0e30;
      int cublaslt_algo_count = cublaslt_gemm.algo_count();
      if (cublaslt_algo_count <= 0)
      {
        printf("cuBLASLt returned no algorithms for size %d\n", size);
        continue;
      }

      for (int algo_idx = 0; algo_idx < cublaslt_algo_count; ++algo_idx)
      {
        CUDA_CHECK(cudaMemset(Dptr, 0, sizeof(T) * M * N));
        for (int it = 0; it < warmup; ++it)
        {
          cublaslt_gemm.run_algo(algo_idx);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int it = 0; it < count; ++it)
        {
          cublaslt_gemm.run_algo(algo_idx);
        }
        CUDA_CHECK(cudaEventRecord(end));
        CUDA_CHECK(cudaEventSynchronize(end));
        CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start, end));

        double algo_ms = elapsedTime / count;
        if (algo_ms < cublaslt_ms)
        {
          cublaslt_ms = algo_ms;
          cublaslt_best_algo = algo_idx;
        }
      }

      double cublaslt_tflops = gemm_tflops(M, N, K, cublaslt_ms);
      printf("%8d %12.6f %12.6f %14.6f %14.6f %8d %11.2f%%\n", size,
             final_ms, cublaslt_ms, final_tflops, cublaslt_tflops,
             cublaslt_best_algo,
             final_tflops / cublaslt_tflops * 100);
    }
  }
  else if (algo == "shm")
  {
    config::GemmConfigV1<T, 128, 128, 32, 8> gemm_config;
    dim3 grid = get_swizzled_grid<decltype(gemm_config)>(M, N);
    dim3 block(size(decltype(gemm_config)::MMA{}));
    int shm_size = gemm_config.kShmSize;

    cudaMemset(Dptr, 0, sizeof(T) * M * N);
    cudaFuncSetAttribute(gemm_opt_shm<decltype(gemm_config)>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);

    cudaEventRecord(start);
    for (int it = 0; it < count; ++it)
    {
      gemm_opt_shm<decltype(gemm_config)>
          <<<grid, block, shm_size>>>(Dptr, Aptr, Bptr, M, N, K);
    }

    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsedTime, start, end);
    printf("gemm_opt_shm took %f ms.\n", elapsedTime / count);
  }
  else if (algo == "p1")
  {
    config::GemmConfig<T, 128, 128, 32, 3, 4, 8> gemm_config;
    dim3 grid = get_swizzled_grid<decltype(gemm_config)>(M, N);
    dim3 block(size(decltype(gemm_config)::MMA{}));
    int shm_size = gemm_config.kShmSize;

    cudaMemset(Dptr, 0, sizeof(T) * M * N);
    cudaFuncSetAttribute(gemm_opt_p1<decltype(gemm_config)>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);

    cudaEventRecord(start);
    for (int it = 0; it < count; ++it)
    {
      gemm_opt_p1<decltype(gemm_config)>
          <<<grid, block, shm_size>>>(Dptr, Aptr, Bptr, M, N, K);
    }

    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsedTime, start, end);
    printf("gemm_opt_p1 took %f ms.\n", elapsedTime / count);
  }
  else if (algo == "p2")
  {
    config::GemmConfig<T, 128, 128, 32, 3, 4, 8> gemm_config;
    dim3 grid = get_swizzled_grid<decltype(gemm_config)>(M, N);
    dim3 block(size(decltype(gemm_config)::MMA{}));
    int shm_size = gemm_config.kShmSize;

    cudaMemset(Dptr, 0, sizeof(T) * M * N);
    cudaFuncSetAttribute(gemm_opt_p2<decltype(gemm_config)>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);

    cudaEventRecord(start);
    for (int it = 0; it < count; ++it)
    {
      gemm_opt_p2<decltype(gemm_config)>
          <<<grid, block, shm_size>>>(Dptr, Aptr, Bptr, M, N, K);
    }

    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsedTime, start, end);
    printf("gemm_opt_p2 took %f ms.\n", elapsedTime / count);
  }
  else if (algo == "final")
  {
    config::GemmConfig<T, 128, 128, 32, 3, 4, 8> gemm_config;
    dim3 grid = get_swizzled_grid<decltype(gemm_config)>(M, N);
    dim3 block(size(decltype(gemm_config)::MMA{}));
    int shm_size = gemm_config.kShmSize;

    cudaMemset(Dptr, 0, sizeof(T) * M * N);
    cudaFuncSetAttribute(gemm_opt_final<decltype(gemm_config)>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);

    cudaEventRecord(start);
    for (int it = 0; it < 1; ++it)
    {
      gemm_opt_final<decltype(gemm_config)>
          <<<grid, block, shm_size>>>(Dptr, Aptr, Bptr, M, N, K);
    }

    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsedTime, start, end);
    printf("gemm_opt_final took %f ms.\n", elapsedTime / count);
  }
}
