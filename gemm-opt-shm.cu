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

int main(int argc, char *argv[])
{
  using T = cute::half_t;

  int M = 4096;
  int N = 4096;
  int K = 4096;
  int count = 100;
  if (argc > 1)
  {
    count = atoi(argv[1]);
  }

  srand(1000);

  T *Aptr;
  T *Bptr;
  T *Dptr;
  CUDA_CHECK(cudaMalloc(&Aptr, sizeof(T) * M * K));
  CUDA_CHECK(cudaMalloc(&Bptr, sizeof(T) * N * K));
  CUDA_CHECK(cudaMalloc(&Dptr, sizeof(T) * M * N));

  T *Aptr_host = (T *)malloc(sizeof(T) * M * K);
  T *Bptr_host = (T *)malloc(sizeof(T) * N * K);
  auto tA = make_tensor(Aptr_host, make_shape(M, K), make_stride(K, 1));
  auto tB = make_tensor(Bptr_host, make_shape(N, K), make_stride(K, 1));
  cpu_rand_data(&tA);
  cpu_rand_data(&tB);
  CUDA_CHECK(cudaMemcpy(Aptr, Aptr_host, sizeof(T) * M * K,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(Bptr, Bptr_host, sizeof(T) * N * K,
                        cudaMemcpyHostToDevice));

  config::GemmConfigV1<T, 128, 128, 32, 8> gemm_config;
  dim3 grid = get_swizzled_grid<decltype(gemm_config)>(M, N);
  dim3 block(size(decltype(gemm_config)::MMA{}));
  int shm_size = gemm_config.kShmSize;

  CUDA_CHECK(cudaMemset(Dptr, 0, sizeof(T) * M * N));
  CUDA_CHECK(cudaFuncSetAttribute(gemm_opt_shm<decltype(gemm_config)>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  shm_size));

  cudaEvent_t start, end;
  float elapsedTime;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&end));
  CUDA_CHECK(cudaEventRecord(start));
  for (int it = 0; it < count; ++it)
  {
    gemm_opt_shm<decltype(gemm_config)>
        <<<grid, block, shm_size>>>(Dptr, Aptr, Bptr, M, N, K);
  }
  CUDA_CHECK(cudaEventRecord(end));
  CUDA_CHECK(cudaEventSynchronize(end));
  CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start, end));
  printf("gemm_opt_shm took %f ms.\n", elapsedTime / count);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(end));
  CUDA_CHECK(cudaFree(Aptr));
  CUDA_CHECK(cudaFree(Bptr));
  CUDA_CHECK(cudaFree(Dptr));
  free(Aptr_host);
  free(Bptr_host);
  return 0;
}


