#ifndef CAFFE_UTIL_MATH_FUNCTIONS_H_
#define CAFFE_UTIL_MATH_FUNCTIONS_H_

#include <cmath> // for std::fabs and std::signbit
#include <cstdint>

#include "glog/logging.h"

#include "caffe/common.hpp"
#include "caffe/util/device_alternate.hpp"
#include "caffe/util/mkl_alternate.hpp"

namespace caffe {

template <typename Dtype> Dtype epsilon() {
  return std::numeric_limits<Dtype>().epsilon();
}

//#define sigmoid(x)  logistic_activate(x)
//#define sigmoid_gradient(x)  logistic_gradient(x)

template <typename Dtype> static inline Dtype logistic_activate(Dtype x) {
  return 1. / (1. + exp(-x));
}
template <typename Dtype> static inline Dtype logistic_gradient(Dtype x) {
  return (1 - x) * x;
}
static inline float hard_sigmoid(float x) {
  return std::min(1., std::max(0., x * 0.2 + 0.5));
}

template <typename Dtype> void caffe_cpu_logistic_activate(Dtype *x, int n) {
  caffe_sigmoid(n, x, x);
}

template <typename Dtype>
void caffe_gpu_logistic_activate(const int N, const Dtype *a, Dtype *y);

template <typename Dtype>
void caffe_gpu_hard_sigmoid(const int N, const Dtype *a, Dtype *y);

template <typename Dtype> void caffe_cpu_hard_sigmoid(Dtype *x, const int n);

// Caffe gemm provides a simpler interface to the gemm functions, with the
// limitation that the data has to be contiguous in memory.
/**
 * @brief 接收三个矩阵A, B, C
 * C:= alpha*A*B + beta*C
 * @tparam Dtype
 * @param TransA    对于矩阵 A 是否做转置（CBlasTrans, CBlasNoTrans）
 * @param TransB    对于矩阵 B 是否做转置（CBlasTrans, CBlasNoTrans）
 * @param M         矩阵A, C的行数
 * @param N         矩阵B, C的列数
 * @param K         矩阵 A 的列数，矩阵 B 的行数
 * @param alpha     系数
 * @param A
 * @param B
 * @param beta      系数
 * @param C
 */
template <typename Dtype>
void caffe_cpu_gemm(CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB, int M,
                    int N, int K, Dtype alpha, const Dtype *A, const Dtype *B,
                    Dtype beta, Dtype *C);

/**
 * @brief 接收一个矩阵 A，以及两个向量x，y。
 * y:= alpha*A*x + beta*y
 * @tparam Dtype
 * @param TransA    对于矩阵 A 是否做转置（CBlasTrans, CBlasNoTrans）
 * @param M         矩阵A的行数
 * @param N         矩阵A的列数
 * @param alpha     系数
 * @param A         矩阵
 * @param x         向量
 * @param beta      系数
 * @param y         向量
 */
template <typename Dtype>
void caffe_cpu_gemv(CBLAS_TRANSPOSE TransA, int M, int N, Dtype alpha,
                    const Dtype *A, const Dtype *x, const Dtype beta, Dtype *y);

/**
 * @brief 接收两个向量 x，y
 * y:= alpha*x + y
 * @tparam Dtype
 * @param N         向量x和y的元素数量
 * @param alpha     系数
 * @param X
 * @param Y
 */
template <typename Dtype>
void caffe_axpy(int N, Dtype alpha, const Dtype *X, Dtype *Y);

template <typename Dtype>
void caffe_cpu_axpby(const int N, const Dtype alpha, const Dtype *X,
                     const Dtype beta, Dtype *Y);

template <typename Dtype>
void caffe_copy(const int N, const Dtype *X, Dtype *Y);

template <typename Dtype>
void caffe_set(const int N, const Dtype alpha, Dtype *X);

inline void caffe_memset(const size_t N, const int alpha, void *X) {
  memset(X, alpha, N); // NOLINT(caffe/alt_fn)
}

template <typename Dtype>
void caffe_add_scalar(const int N, const Dtype alpha, Dtype *X);

template <typename Dtype>
void caffe_scal(const int N, const Dtype alpha, Dtype *X);

template <typename Dtype> void caffe_sqr(const int N, const Dtype *a, Dtype *y);

template <typename Dtype>
void caffe_sqrt(const int N, const Dtype *a, Dtype *y);

template <typename Dtype>
void caffe_add(const int N, const Dtype *a, const Dtype *b, Dtype *y);

template <typename Dtype>
void caffe_sub(const int N, const Dtype *a, const Dtype *b, Dtype *y);

template <typename Dtype>
void caffe_sub(int N, const Dtype *a, Dtype b, Dtype *y);

template <typename Dtype> void caffe_softmax(int N, const Dtype *a, Dtype *y);

template <typename Dtype>
void caffe_softmax(int N, const Dtype *a, int stride, Dtype *y);

template <typename Dtype> void caffe_sigmoid(int N, const Dtype *a, Dtype *y);

template <typename Dtype>
void caffe_sigmoid(int N, const Dtype *a, int stride, Dtype *y);

template <typename Dtype>
void caffe_mul(const int N, const Dtype *a, const Dtype *b, Dtype *y);

template <typename Dtype>
void caffe_div(const int N, const Dtype *a, const Dtype *b, Dtype *y);

template <typename Dtype>
void caffe_powx(const int n, const Dtype *a, const Dtype b, Dtype *y);

unsigned int caffe_rng_rand();

template <typename Dtype> Dtype caffe_nextafter(const Dtype b);

template <typename Dtype>
void caffe_rng_uniform(const int n, const Dtype a, const Dtype b, Dtype *r);

template <typename Dtype>
void caffe_rng_gaussian(const int n, const Dtype mu, const Dtype sigma,
                        Dtype *r);

template <typename Dtype>
void caffe_rng_bernoulli(const int n, const Dtype p, int *r);

template <typename Dtype>
void caffe_rng_bernoulli(const int n, const Dtype p, unsigned int *r);

template <typename Dtype> void caffe_exp(const int n, const Dtype *a, Dtype *y);

template <typename Dtype> void caffe_log(const int n, const Dtype *a, Dtype *y);

template <typename Dtype> void caffe_abs(const int n, const Dtype *a, Dtype *y);

template <typename Dtype>
Dtype caffe_cpu_dot(const int n, const Dtype *x, const Dtype *y);

template <typename Dtype>
Dtype caffe_cpu_strided_dot(const int n, const Dtype *x, const int incx,
                            const Dtype *y, const int incy);

// Returns the sum of the absolute values of the elements of vector x
template <typename Dtype> Dtype caffe_cpu_asum(const int n, const Dtype *x);

// the branchless, type-safe version from
// http://stackoverflow.com/questions/1903954/is-there-a-standard-sign-function-signum-sgn-in-c-c
template <typename Dtype> inline int8_t caffe_sign(Dtype val) {
  return (Dtype(0) < val) - (val < Dtype(0));
}

template <typename Dtype>
inline Dtype caffe_cpu_clip(Dtype a, Dtype min, Dtype max) {
  return std::max(std::min(a, max), min);
}

template <typename Dtype>
inline void caffe_cpu_clip(int N, const Dtype *src, Dtype *dst, Dtype min,
                           Dtype max) {
  for (int i = 0; i < N; ++i) {
    dst[i] = caffe_cpu_clip(src[i], min, max);
  }
}

// The following two macros are modifications of DEFINE_VSL_UNARY_FUNC
//   in include/caffe/util/mkl_alternate.hpp authored by @Rowland Depp.
// Please refer to commit 7e8ef25c7 of the boost-eigen branch.
// Git cherry picking that commit caused a conflict hard to resolve and
//   copying that file in convenient for code reviewing.
// So they have to be pasted here temporarily.
#define DEFINE_CAFFE_CPU_UNARY_FUNC(name, operation)                           \
  template <typename Dtype>                                                    \
  void caffe_cpu_##name(const int n, const Dtype *x, Dtype *y) {               \
    CHECK_GT(n, 0);                                                            \
    CHECK(x);                                                                  \
    CHECK(y);                                                                  \
    for (int i = 0; i < n; ++i) {                                              \
      operation;                                                               \
    }                                                                          \
  }

// output is 1 for the positives, 0 for zero, and -1 for the negatives
DEFINE_CAFFE_CPU_UNARY_FUNC(sign, y[i] = caffe_sign<Dtype>(x[i]))

// This returns a nonzero value if the input has its sign bit set.
// The name sngbit is meant to avoid conflicts with std::signbit in the macro.
// The extra parens are needed because CUDA < 6.5 defines signbit as a macro,
// and we don't want that to expand here when CUDA headers are also included.
DEFINE_CAFFE_CPU_UNARY_FUNC(sgnbit,
                            y[i] = static_cast<bool>((std::signbit)(x[i])))

DEFINE_CAFFE_CPU_UNARY_FUNC(fabs, y[i] = std::fabs(x[i]))

template <typename Dtype>
void caffe_cpu_scale(const int n, const Dtype alpha, const Dtype *x, Dtype *y);

#ifndef CPU_ONLY // GPU

// Decaf gpu gemm provides an interface that is almost the same as the cpu
// gemm function - following the c convention and calling the fortran-order
// gpu code under the hood.
template <typename Dtype>
void caffe_gpu_gemm(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
                    const int M, const int N, const int K, const Dtype alpha,
                    const Dtype *A, const Dtype *B, const Dtype beta, Dtype *C);

template <typename Dtype>
void caffe_gpu_gemv(const CBLAS_TRANSPOSE TransA, const int M, const int N,
                    const Dtype alpha, const Dtype *A, const Dtype *x,
                    const Dtype beta, Dtype *y);

template <typename Dtype>
void caffe_gpu_axpy(const int N, const Dtype alpha, const Dtype *X, Dtype *Y);

template <typename Dtype>
void caffe_gpu_axpby(const int N, const Dtype alpha, const Dtype *X,
                     const Dtype beta, Dtype *Y);

void caffe_gpu_memcpy(const size_t N, const void *X, void *Y);

template <typename Dtype>
void caffe_gpu_set(const int N, const Dtype alpha, Dtype *X);

inline void caffe_gpu_memset(const size_t N, const int alpha, void *X) {
#ifndef CPU_ONLY
  CUDA_CHECK(cudaMemset(X, alpha, N)); // NOLINT(caffe/alt_fn)
#else
  NO_GPU;
#endif
}

template <typename Dtype>
void caffe_gpu_add_scalar(const int N, const Dtype alpha, Dtype *X);

template <typename Dtype>
void caffe_gpu_scal(const int N, const Dtype alpha, Dtype *X);

#ifndef CPU_ONLY
template <typename Dtype>
void caffe_gpu_scal(const int N, const Dtype alpha, Dtype *X, cudaStream_t str);
#endif

template <typename Dtype>
void caffe_gpu_add(const int N, const Dtype *a, const Dtype *b, Dtype *y);

template <typename Dtype>
void caffe_gpu_sub(const int N, const Dtype *a, const Dtype *b, Dtype *y);

template <typename Dtype>
void caffe_gpu_mul(const int N, const Dtype *a, const Dtype *b, Dtype *y);

template <typename Dtype>
void caffe_gpu_div(const int N, const Dtype *a, const Dtype *b, Dtype *y);

template <typename Dtype>
void caffe_gpu_abs(const int n, const Dtype *a, Dtype *y);

template <typename Dtype>
void caffe_gpu_exp(const int n, const Dtype *a, Dtype *y);

template <typename Dtype>
void caffe_gpu_log(const int n, const Dtype *a, Dtype *y);

template <typename Dtype>
void caffe_gpu_powx(const int n, const Dtype *a, const Dtype b, Dtype *y);

template <typename Dtype>
void caffe_gpu_sqrt(const int n, const Dtype *a, Dtype *y);

// caffe_gpu_rng_uniform with two arguments generates integers in the range
// [0, UINT_MAX].
void caffe_gpu_rng_uniform(const int n, unsigned int *r);

// caffe_gpu_rng_uniform with four arguments generates floats in the range
// (a, b] (strictly greater than a, less than or equal to b) due to the
// specification of curandGenerateUniform.  With a = 0, b = 1, just calls
// curandGenerateUniform; with other limits will shift and scale the outputs
// appropriately after calling curandGenerateUniform.
template <typename Dtype>
void caffe_gpu_rng_uniform(const int n, const Dtype a, const Dtype b, Dtype *r);

template <typename Dtype>
void caffe_gpu_rng_gaussian(const int n, const Dtype mu, const Dtype sigma,
                            Dtype *r);

template <typename Dtype>
void caffe_gpu_rng_bernoulli(const int n, const Dtype p, int *r);

template <typename Dtype>
void caffe_gpu_dot(const int n, const Dtype *x, const Dtype *y, Dtype *out);

template <typename Dtype>
void caffe_gpu_asum(const int n, const Dtype *x, Dtype *y);

template <typename Dtype>
void caffe_gpu_sign(const int n, const Dtype *x, Dtype *y);

template <typename Dtype>
void caffe_gpu_sgnbit(const int n, const Dtype *x, Dtype *y);

template <typename Dtype>
void caffe_gpu_fabs(const int n, const Dtype *x, Dtype *y);

template <typename Dtype>
void caffe_gpu_scale(const int n, const Dtype alpha, const Dtype *x, Dtype *y);

#define DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(name, operation)                 \
  template <typename Dtype>                                                    \
  __global__ void name##_kernel(const int n, const Dtype *x, Dtype *y) {       \
    CUDA_KERNEL_LOOP(index, n) { operation; }                                  \
  }                                                                            \
  template <>                                                                  \
  void caffe_gpu_##name<float>(const int n, const float *x, float *y) {        \
    /* NOLINT_NEXT_LINE(whitespace/operators) */                               \
    name##_kernel<float>                                                       \
        <<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(n, x, y);            \
  }                                                                            \
  template <>                                                                  \
  void caffe_gpu_##name<double>(const int n, const double *x, double *y) {     \
    /* NOLINT_NEXT_LINE(whitespace/operators) */                               \
    name##_kernel<double>                                                      \
        <<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(n, x, y);            \
  }

#endif // !CPU_ONLY

} // namespace caffe

#endif // CAFFE_UTIL_MATH_FUNCTIONS_H_
