#ifndef CAFFE_UTIL_MATH_FUNCTIONS_H_
#define CAFFE_UTIL_MATH_FUNCTIONS_H_

#include <cmath> // for std::fabs and std::signbit
#include <cstdint>

#include "glog/logging.h"

#include "caffe/common.hpp"
#include "caffe/util/device_alternate.hpp"
#include "caffe/util/mkl_alternate.hpp"

#ifdef USE_TBB
#include <tbb/parallel_for.h>
#include <tbb/parallel_for_each.h>
#include <tbb/parallel_reduce.h>
#endif

namespace caffe {

template <typename Dtype>
Dtype epsilon() {
  return std::numeric_limits<Dtype>().epsilon();
}

template <typename Dtype>
void caffe_gpu_logistic_activate(int N, const Dtype *a, Dtype *y);

template <typename Dtype>
void caffe_gpu_hard_sigmoid(int N, const Dtype *a, Dtype *y);

template <typename Dtype>
void caffe_cpu_hard_sigmoid(Dtype *x, int n);

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
void caffe_blas_gemm(CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB, int M,
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
void caffe_blas_gemv(CBLAS_TRANSPOSE TransA, int M, int N, Dtype alpha,
                     const Dtype *A, const Dtype *x, Dtype beta, Dtype *y);

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
void caffe_blas_axpy(int N, Dtype alpha, const Dtype *X, Dtype *Y);

template <typename Dtype>
void caffe_blas_axpby(int N, Dtype alpha, const Dtype *X, Dtype beta, Dtype *Y);

template <typename Dtype>
void caffe_copy(int N, const Dtype *X, Dtype *Y);

template <typename Dtype>
void caffe_set(int N, Dtype alpha, Dtype *X);

inline void caffe_memset(const size_t N, const int alpha, void *X) {
  memset(X, alpha, N); // NOLINT(caffe/alt_fn)
}

template <typename Dtype>
void caffe_blas_scal(int N, Dtype alpha, Dtype *X);

template <typename Dtype>
void caffe_add(int N, const Dtype *a, const Dtype *b, Dtype *y);

template <typename Dtype>
void caffe_sub(int N, const Dtype *a, const Dtype *b, Dtype *y);

template <typename Dtype>
void caffe_mul(int N, const Dtype *a, const Dtype *b, Dtype *y);

template <typename Dtype>
void caffe_div(int N, const Dtype *a, const Dtype *b, Dtype *y);

template <typename Dtype>
void caffe_powx(int n, const Dtype *a, Dtype b, Dtype *y);

template <typename Dtype>
void caffe_sqr(int N, const Dtype *a, Dtype *y);

template <typename Dtype>
void caffe_sqrt(int N, const Dtype *a, Dtype *y);

template <typename Dtype>
void caffe_exp(int n, const Dtype *a, Dtype *y);

template <typename Dtype>
void caffe_log(int n, const Dtype *a, Dtype *y);

template <typename Dtype>
void caffe_abs(int n, const Dtype *a, Dtype *y);

template <typename Dtype>
Dtype caffe_blas_strided_dot(int n, const Dtype *x, int incx, const Dtype *y,
                             int incy);

template <typename Dtype>
Dtype caffe_blas_dot(int n, const Dtype *x, const Dtype *y);

// Returns the sum of the absolute values of the elements of vector x
template <typename Dtype>
Dtype caffe_blas_asum(int n, const Dtype *x);

template <typename Dtype>
void caffe_blas_scale(int n, Dtype alpha, const Dtype *x, Dtype *y);

// the branchless, type-safe version from
// http://stackoverflow.com/questions/1903954/is-there-a-standard-sign-function-signum-sgn-in-c-c
template <typename Dtype>
inline int8_t caffe_sign(Dtype val) {
  return (Dtype(0) < val) - (val < Dtype(0));
}

template <typename Dtype>
inline Dtype caffe_clip(Dtype a, Dtype min, Dtype max) {
  return std::max(std::min(a, max), min);
}

template <typename Dtype>
inline void caffe_clip(int N, const Dtype *src, Dtype *dst, Dtype min,
                       Dtype max) {
  for (int i = 0; i < N; ++i) {
    dst[i] = caffe_clip(src[i], min, max);
  }
}

unsigned int caffe_rng_rand();

template <typename Dtype>
Dtype caffe_nextafter(Dtype b);

template <typename Dtype>
void caffe_rng_uniform(int n, Dtype a, Dtype b, Dtype *r);

template <typename Dtype>
void caffe_rng_gaussian(int n, Dtype mu, Dtype sigma, Dtype *r);

template <typename Dtype>
void caffe_rng_bernoulli(int n, Dtype p, int *r);

template <typename Dtype>
void caffe_rng_bernoulli(int n, Dtype p, unsigned int *r);

#ifndef CPU_ONLY // GPU

// Decaf gpu gemm provides an interface that is almost the same as the cpu
// gemm function - following the c convention and calling the fortran-order
// gpu code under the hood.
template <typename Dtype>
void caffe_gpu_gemm(CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB, int M,
                    int N, int K, Dtype alpha, const Dtype *A, const Dtype *B,
                    Dtype beta, Dtype *C);

template <typename Dtype>
void caffe_gpu_gemv(CBLAS_TRANSPOSE TransA, int M, int N, Dtype alpha,
                    const Dtype *A, const Dtype *x, Dtype beta, Dtype *y);

template <typename Dtype>
void caffe_gpu_axpy(int N, Dtype alpha, const Dtype *X, Dtype *Y);

template <typename Dtype>
void caffe_gpu_axpby(int N, Dtype alpha, const Dtype *X, Dtype beta, Dtype *Y);

void caffe_gpu_memcpy(size_t N, const void *X, void *Y);

template <typename Dtype>
void caffe_gpu_set(int N, Dtype alpha, Dtype *X);

inline void caffe_gpu_memset(const size_t N, const int alpha, void *X) {
#ifndef CPU_ONLY
  CUDA_CHECK(cudaMemset(X, alpha, N)); // NOLINT(caffe/alt_fn)
#else
  NO_GPU;
#endif
}

template <typename Dtype>
void caffe_gpu_add_scalar(int N, Dtype alpha, Dtype *X);

template <typename Dtype>
void caffe_gpu_scal(int N, Dtype alpha, Dtype *X);

#ifndef CPU_ONLY
template <typename Dtype>
void caffe_gpu_scal(int N, Dtype alpha, Dtype *X, cudaStream_t str);
#endif

template <typename Dtype>
void caffe_gpu_add(int N, const Dtype *a, const Dtype *b, Dtype *y);

template <typename Dtype>
void caffe_gpu_sub(int N, const Dtype *a, const Dtype *b, Dtype *y);

template <typename Dtype>
void caffe_gpu_mul(int N, const Dtype *a, const Dtype *b, Dtype *y);

template <typename Dtype>
void caffe_gpu_div(int N, const Dtype *a, const Dtype *b, Dtype *y);

template <typename Dtype>
void caffe_gpu_abs(int n, const Dtype *a, Dtype *y);

template <typename Dtype>
void caffe_gpu_exp(int n, const Dtype *a, Dtype *y);

template <typename Dtype>
void caffe_gpu_log(int n, const Dtype *a, Dtype *y);

template <typename Dtype>
void caffe_gpu_powx(int n, const Dtype *a, Dtype b, Dtype *y);

template <typename Dtype>
void caffe_gpu_sqrt(int n, const Dtype *a, Dtype *y);

// caffe_gpu_rng_uniform with two arguments generates integers in the range
// [0, UINT_MAX].
void caffe_gpu_rng_uniform(int n, unsigned int *r);

// caffe_gpu_rng_uniform with four arguments generates floats in the range
// (a, b] (strictly greater than a, less than or equal to b) due to the
// specification of curandGenerateUniform.  With a = 0, b = 1, just calls
// curandGenerateUniform; with other limits will shift and scale the outputs
// appropriately after calling curandGenerateUniform.
template <typename Dtype>
void caffe_gpu_rng_uniform(int n, Dtype a, Dtype b, Dtype *r);

template <typename Dtype>
void caffe_gpu_rng_gaussian(int n, Dtype mu, Dtype sigma, Dtype *r);

template <typename Dtype>
void caffe_gpu_rng_bernoulli(int n, Dtype p, int *r);

template <typename Dtype>
void caffe_gpu_dot(int n, const Dtype *x, const Dtype *y, Dtype *out);

template <typename Dtype>
void caffe_gpu_asum(int n, const Dtype *x, Dtype *y);

template <typename Dtype>
void caffe_gpu_sign(int n, const Dtype *x, Dtype *y);

template <typename Dtype>
void caffe_gpu_sgnbit(int n, const Dtype *x, Dtype *y);

template <typename Dtype>
void caffe_gpu_fabs(int n, const Dtype *x, Dtype *y);

template <typename Dtype>
void caffe_gpu_scale(int n, Dtype alpha, const Dtype *x, Dtype *y);

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

// The following two macros are modifications of DEFINE_VSL_UNARY_FUNC
//   in include/caffe/util/mkl_alternate.hpp authored by @Rowland Depp.
// Please refer to commit 7e8ef25c7 of the boost-eigen branch.
// Git cherry picking that commit caused a conflict hard to resolve and
//   copying that file in convenient for code reviewing.
// So they have to be pasted here temporarily.
#ifdef USE_TBB
#define DEFINE_CAFFE_CPU_UNARY_FUNC(name, function)                            \
  template <typename Dtype>                                                    \
  inline Dtype caffe_fn_##name(Dtype x) {                                      \
    return function;                                                           \
  }                                                                            \
  template <typename Dtype>                                                    \
  inline void caffe_##name(const int n, int INCX, const Dtype *x, int INCY,    \
                           Dtype *y) {                                         \
    CHECK_GT(n, 0);                                                            \
    CHECK_GT(INCX, 0);                                                         \
    CHECK_GT(INCY, 0);                                                         \
    CHECK(x);                                                                  \
    CHECK(y);                                                                  \
    tbb::parallel_for(                                                         \
        tbb::blocked_range<int>(0, n),                                         \
        [&](tbb::blocked_range<int> r) {                                       \
          int ix, iy;                                                          \
          for (int i = r.begin(); i < r.end(); i++) {                          \
            ix = i * INCX;                                                     \
            iy = i * INCY;                                                     \
            y[iy] = caffe_fn_##name(x[ix]);                                    \
          }                                                                    \
        },                                                                     \
        tbb::auto_partitioner());                                              \
  }                                                                            \
  template <typename Dtype>                                                    \
  inline void caffe_##name(const int n, const Dtype *x, Dtype *y) {            \
    caffe_##name(n, 1, x, 1, y);                                               \
  }
#elif USE_OMP
#else
#define DEFINE_CAFFE_CPU_UNARY_FUNC(name, function)                            \
  template <typename Dtype>                                                    \
  inline Dtype caffe_fn_##name(Dtype x) {                                      \
    return function;                                                           \
  }                                                                            \
  template <typename Dtype>                                                    \
  inline void caffe_##name(const int n, int INCX, const Dtype *x, int INCY,    \
                           Dtype *y) {                                         \
    CHECK_GT(n, 0);                                                            \
    CHECK_GT(INCX, 0);                                                         \
    CHECK_GT(INCY, 0);                                                         \
    CHECK(x);                                                                  \
    CHECK(y);                                                                  \
    int ix, iy;                                                                \
    for (int i = 0; i < n; i++) {                                              \
      ix = i * INCX;                                                           \
      iy = i * INCY;                                                           \
      function;                                                                \
    }                                                                          \
  }                                                                            \
  template <typename Dtype>                                                    \
  inline void caffe_##name(const int n, const Dtype *x, Dtype *y) {            \
    caffe_##name(n, 1, x, 1, y);                                               \
  }
#endif

// output is 1 for the positives, 0 for zero, and -1 for the negatives
DEFINE_CAFFE_CPU_UNARY_FUNC(sign, caffe_sign<Dtype>(x))

// This returns a nonzero value if the input has its sign bit set.
// The name sngbit is meant to avoid conflicts with std::signbit in the macro.
// The extra parens are needed because CUDA < 6.5 defines signbit as a macro,
// and we don't want that to expand here when CUDA headers are also included.
DEFINE_CAFFE_CPU_UNARY_FUNC(sgnbit, static_cast<bool>((std::signbit)(x)))

DEFINE_CAFFE_CPU_UNARY_FUNC(fabs, std::fabs(x))

////////////////////////////////////////////////////////////////////// new added

/**
 * @brief This macro create a func that manipulate tensor with a scalar
 */
#ifdef USE_TBB
#define DEFINE_CAFFE_CPU_BINARY_SCALAR_FUNC(name, operation)                   \
  template <typename Dtype>                                                    \
  inline void caffe_##name##_scalar(const int n, Dtype b, int INCX,            \
                                    const Dtype *x, int INCY, Dtype *y) {      \
    CHECK_GT(n, 0);                                                            \
    CHECK_GT(INCX, 0);                                                         \
    CHECK_GT(INCY, 0);                                                         \
    CHECK(x);                                                                  \
    CHECK(y);                                                                  \
    tbb::parallel_for(                                                         \
        tbb::blocked_range<int>(0, n),                                         \
        [&](tbb::blocked_range<int> r) {                                       \
          int ix, iy;                                                          \
          for (int i = r.begin(); i < r.end(); i++) {                          \
            ix = i * INCX;                                                     \
            iy = i * INCY;                                                     \
            operation;                                                         \
          }                                                                    \
        },                                                                     \
        tbb::auto_partitioner());                                              \
  }                                                                            \
  template <typename Dtype>                                                    \
  inline void caffe_##name##_scalar(const int n, Dtype b, Dtype *x) {          \
    caffe_##name##_scalar(n, b, 1, x, 1, x);                                   \
  }                                                                            \
  template <typename Dtype>                                                    \
  inline void caffe_##name##_scalar(const int n, Dtype b, const Dtype *x,      \
                                    Dtype *y) {                                \
    caffe_##name##_scalar(n, b, 1, x, 1, y);                                   \
  }
#elif USE_OMP
#else
#define DEFINE_CAFFE_CPU_BINARY_SCALAR_FUNC(name, operation)                   \
  template <typename Dtype>                                                    \
  inline void caffe_##name##_scalar(const int n, Dtype b, int INCX,            \
                                    const Dtype *x, int INCY, Dtype *y) {      \
    CHECK_GT(n, 0);                                                            \
    CHECK_GT(INCX, 0);                                                         \
    CHECK_GT(INCY, 0);                                                         \
    CHECK(x);                                                                  \
    CHECK(y);                                                                  \
    int ix, iy;                                                                \
    for (int i = 0; i < n; i++) {                                              \
      ix = i * INCX;                                                           \
      iy = i * INCY;                                                           \
      operation;                                                               \
    }                                                                          \
  }                                                                            \
  template <typename Dtype>                                                    \
  inline void caffe_##name##_scalar(const int n, Dtype b, Dtype *x) {          \
    caffe_##name##_scalar(n, b, 1, x, 1, x);                                   \
  }                                                                            \
  template <typename Dtype>                                                    \
  inline void caffe_##name##_scalar(const int n, Dtype b, const Dtype *x,      \
                                    Dtype *y) {                                \
    caffe_##name##_scalar(n, b, 1, x, 1, y);                                   \
  }
#endif

/**
 * @brief This macro define a single math function like `y = f(x)`
 */

template <typename Dtype>
void caffe_softmax(int N, const Dtype *a, Dtype *y);

template <typename Dtype>
void caffe_softmax(int N, const Dtype *a, int stride, Dtype *y);

DEFINE_CAFFE_CPU_BINARY_SCALAR_FUNC(add, y[iy] = x[ix] + b)
DEFINE_CAFFE_CPU_BINARY_SCALAR_FUNC(sub, y[iy] = x[ix] - b)
DEFINE_CAFFE_CPU_BINARY_SCALAR_FUNC(mul, y[iy] = x[ix] * b)
DEFINE_CAFFE_CPU_BINARY_SCALAR_FUNC(div, y[iy] = x[ix] / b)

DEFINE_CAFFE_CPU_UNARY_FUNC(tanh, tanh(x))
DEFINE_CAFFE_CPU_UNARY_FUNC(tanh_grad, 1 - pow(tanh(x), 2))
DEFINE_CAFFE_CPU_UNARY_FUNC(sigmoid, 1. / (1. + exp(-x)))
DEFINE_CAFFE_CPU_UNARY_FUNC(sigmoid_grad_fast, (1 - x) * x)
DEFINE_CAFFE_CPU_UNARY_FUNC(sigmoid_grad,
                            (1 - caffe_fn_sigmoid(x)) * caffe_fn_sigmoid(x))
DEFINE_CAFFE_CPU_UNARY_FUNC(hard_sigmoid,
                            std::min(1., std::max(0., x * 0.2 + 0.5)))
DEFINE_CAFFE_CPU_UNARY_FUNC(softplus, std::log(1 + std::exp(x)))
DEFINE_CAFFE_CPU_UNARY_FUNC(softplus_grad, caffe_fn_sigmoid(x))
DEFINE_CAFFE_CPU_UNARY_FUNC(mish, x *caffe_fn_tanh(caffe_fn_softplus(x)))
DEFINE_CAFFE_CPU_UNARY_FUNC(mish_grad, (exp(x) *
                                        (4 * (x + 1) + 4 * exp(2 * x) +
                                         exp(3 * x) + exp(x) * (4 * x + 6)) /
                                        pow(2 * exp(x) + exp(2 * x) + 2, 2)))

////////////////////////////////////////////////////////////////// end new added

} // namespace caffe

#endif // CAFFE_UTIL_MATH_FUNCTIONS_H_
