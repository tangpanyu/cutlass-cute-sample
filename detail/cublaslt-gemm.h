#include <cublasLt.h>
#include <cuda_fp16.h>
#include <stdio.h>

#define check(call)                                        \
  do {                                                     \
    auto err = call;                                       \
    if (err != CUBLAS_STATUS_SUCCESS) {                    \
      printf("err = %d, str = %s, line = %d, %s\n", err,   \
             cublasGetStatusString(err), __LINE__, #call); \
      exit(0);                                             \
    }                                                      \
  } while (0)

template <typename T>
struct ComputeTypeTraits {
  static constexpr cublasComputeType_t kComputeType = CUBLAS_COMPUTE_16F;
  static constexpr cudaDataType_t kScaleType = CUDA_R_16F;
};

template <>
struct ComputeTypeTraits<float> {
  static constexpr cublasComputeType_t kComputeType = CUBLAS_COMPUTE_32F;
  static constexpr cudaDataType_t kScaleType = CUDA_R_32F;
};

template <typename T, typename ComputeType = T>
struct CublasLtGemm {
  CublasLtGemm();
  ~CublasLtGemm();

  cublasLtHandle_t handle_;

  cublasLtMatrixLayout_t a_desc_;
  cublasLtMatrixLayout_t b_desc_;
  cublasLtMatrixLayout_t c_desc_;

  cublasLtMatmulDesc_t matmul_desc_;

  cublasLtMatmulPreference_t preference_;

  static constexpr int kAlgoMaxNum = 1024;
  cublasLtMatmulHeuristicResult_t algos_[kAlgoMaxNum];
  int ret_algo_num_;

  ComputeType alpha_;
  ComputeType beta_;
  static constexpr cublasComputeType_t kComputeType =
      ComputeTypeTraits<ComputeType>::kComputeType;
  static constexpr cudaDataType_t kScaleType =
      ComputeTypeTraits<ComputeType>::kScaleType;

  void *workspace_;
  size_t workspace_size_;

  const void *a_;
  const void *b_;
  void *c_;

  void init(T *c, const T *a, const T *b, int m, int n, int k);
  bool run();
  bool run_algo(int algo_idx = 0);
  int algo_count() const { return ret_algo_num_; }
};

template <typename T, typename ComputeType>
CublasLtGemm<T, ComputeType>::CublasLtGemm()
    : handle_(nullptr),
      a_desc_(nullptr),
      b_desc_(nullptr),
      c_desc_(nullptr),
      matmul_desc_(nullptr),
      preference_(nullptr),
      ret_algo_num_(0),
      workspace_(nullptr),
      workspace_size_(0),
      a_(nullptr),
      b_(nullptr),
      c_(nullptr) {}

template <typename T, typename ComputeType>
CublasLtGemm<T, ComputeType>::~CublasLtGemm() {
  if (workspace_ != nullptr) {
    cudaFree(workspace_);
  }
  if (preference_ != nullptr) {
    cublasLtMatmulPreferenceDestroy(preference_);
  }
  if (matmul_desc_ != nullptr) {
    cublasLtMatmulDescDestroy(matmul_desc_);
  }
  if (a_desc_ != nullptr) {
    cublasLtMatrixLayoutDestroy(a_desc_);
  }
  if (b_desc_ != nullptr) {
    cublasLtMatrixLayoutDestroy(b_desc_);
  }
  if (c_desc_ != nullptr) {
    cublasLtMatrixLayoutDestroy(c_desc_);
  }
  if (handle_ != nullptr) {
    cublasLtDestroy(handle_);
  }
}

template <typename T, typename ComputeType>
bool CublasLtGemm<T, ComputeType>::run_algo(int algo_idx) {
  if (algo_idx < 0 || algo_idx >= ret_algo_num_) {
    printf("invalid cublasLt algo_idx = %d, ret_algo_num = %d\n", algo_idx,
           ret_algo_num_);
    return false;
  }

  auto algo = algos_[algo_idx];
  check(cublasLtMatmul(handle_, matmul_desc_, &alpha_, a_, a_desc_, b_,
                       b_desc_, &beta_, c_, c_desc_, c_, c_desc_,
                       &(algo.algo), workspace_, workspace_size_, 0));
  return true;
}

template <typename T, typename ComputeType>
bool CublasLtGemm<T, ComputeType>::run() {
  for (int i = 0; i < ret_algo_num_; ++i) {
    /*
    printf("algo-id = %d, workspace_size = %zu, waves = %f\n", i,
    algo.workspaceSize, algo.wavesCount);
    */
    run_algo(i);
  }
  return true;
}

template <typename T, typename ComputeType>
void CublasLtGemm<T, ComputeType>::init(T *c, const T *a, const T *b, int m,
                                        int n, int k) {
  auto version = cublasLtGetVersion();
  static bool version_printed = false;
  if (!version_printed) {
    printf("cublasLt version: %zu\n", version);
    version_printed = true;
  }

  check(cublasLtCreate(&handle_));

  // cublasLtLoggerSetLevel(5);

  int batch = 1;
  int64_t a_stride = m * k;
  int64_t b_stride = n * k;
  int64_t c_stride = m * n;
  cublasOperation_t transa = CUBLAS_OP_T;
  cublasOperation_t transb = CUBLAS_OP_N;

  check(cublasLtMatrixLayoutCreate(&a_desc_, CUDA_R_16F, k, m, k));
  check(cublasLtMatrixLayoutCreate(&b_desc_, CUDA_R_16F, k, n, k));
  check(cublasLtMatrixLayoutCreate(&c_desc_, CUDA_R_16F, m, n, m));

  check(cublasLtMatrixLayoutSetAttribute(
      a_desc_, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)));
  check(cublasLtMatrixLayoutSetAttribute(
      b_desc_, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)));
  check(cublasLtMatrixLayoutSetAttribute(
      c_desc_, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)));

  check(cublasLtMatrixLayoutSetAttribute(
      a_desc_, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &a_stride,
      sizeof(a_stride)));
  check(cublasLtMatrixLayoutSetAttribute(
      b_desc_, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &b_stride,
      sizeof(b_stride)));
  check(cublasLtMatrixLayoutSetAttribute(
      c_desc_, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &c_stride,
      sizeof(c_stride)));

  check(cublasLtMatmulDescCreate(&matmul_desc_, kComputeType, kScaleType));
  check(cublasLtMatmulDescSetAttribute(
      matmul_desc_, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
  check(cublasLtMatmulDescSetAttribute(
      matmul_desc_, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

  alpha_ = 1.f;
  beta_ = 0.f;
  workspace_size_ = 64 * 1024 * 1024;
  cudaError_t cuda_err = cudaMalloc(&workspace_, workspace_size_);
  if (cuda_err != cudaSuccess) {
    printf("cudaMalloc workspace failed: %s\n", cudaGetErrorString(cuda_err));
    workspace_ = nullptr;
    workspace_size_ = 0;
  }

  cublasLtMatmulPreferenceCreate(&preference_);
  check(cublasLtMatmulPreferenceSetAttribute(
      preference_, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size_,
      sizeof(workspace_size_)));

  cublasLtMatmulAlgoGetHeuristic(handle_, matmul_desc_, a_desc_, b_desc_,
                                 c_desc_, c_desc_, preference_, kAlgoMaxNum,
                                 algos_, &ret_algo_num_);

  a_ = a;
  b_ = b;
  c_ = c;
}

#undef check
