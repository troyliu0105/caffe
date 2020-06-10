#ifndef CAFFE_UTIL_MKL_ALTERNATE_H_
#define CAFFE_UTIL_MKL_ALTERNATE_H_

#ifdef USE_TBB
#include <tbb/mutex.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_sort.h>
#elif defined(USE_OMP)
#ifdef _MSC_VER
#define OMP_PRAGMA_FOR __pragma(omp parallel for)
#define OMP_PRAGMA __pragma(omp)
#define OMP_ATOMIC __pragma(omp critical(dataupdate))
#else
#define OMP_PRAGMA_FOR _Pragma("omp parallel for")
#define OMP_PRAGMA _Pragma(omp)
#define OMP_ATOMIC _Pragma("omp critical(dataupdate)")
#endif
#endif

template <typename Index, typename IteratorFunction>
void parallel_for(Index b_, Index e_, Index stride_,
                  const IteratorFunction &func_) {
#ifdef USE_TBB
  tbb::parallel_for(b_, e_, stride_, func_);
#else
#ifdef USE_OMP
  OMP_PRAGMA_FOR
#endif
  for (Index i = b_; i < e_; i += stride_) {
    func_(i);
  }
#endif
}

template <typename Index, typename IteratorFunction>
void parallel_for(Index e_, const IteratorFunction &func_) {
  parallel_for(static_cast<Index>(0), e_, static_cast<Index>(1), func_);
}

template <typename It, typename Compare>
void parallel_sort(It first, It last, Compare comp) {
#ifdef USE_TBB
  tbb::parallel_sort(first, last, comp);
#else
  std::sort(first, last, comp);
#endif
}

template <typename Mutex, typename Function>
void atomic_update(Mutex &mutex, const Function &func) {
#ifdef USE_TBB
  tbb::mutex::scoped_lock lock(mutex);
#elif defined(USE_OMP)
  OMP_ATOMIC
#endif
  func();
}

#ifdef USE_MKL

#include <mkl.h>

#else // If use MKL, simply include the MKL header

#ifdef USE_ACCELERATE
#include <Accelerate/Accelerate.h>
#else
extern "C" {
#include <cblas.h>
}
#endif // USE_ACCELERATE

#include <cmath>

// Functions that caffe uses but are not present if MKL is not linked.

// A simple way to define the vsl unary functions. The operation should
// be in the form e.g. y[i] = sqrt(a[i])
#define DEFINE_VSL_UNARY_FUNC(name, operation)                                 \
  template <typename Dtype>                                                    \
  void v##name(const int n, const Dtype *a, Dtype *y) {                        \
    CHECK_GT(n, 0);                                                            \
    CHECK(a);                                                                  \
    CHECK(y);                                                                  \
    parallel_for(0, n, 1, [&](int i) { operation; });                          \
  }                                                                            \
  inline void vs##name(const int n, const float *a, float *y) {                \
    v##name<float>(n, a, y);                                                   \
  }                                                                            \
  inline void vd##name(const int n, const double *a, double *y) {              \
    v##name<double>(n, a, y);                                                  \
  }

DEFINE_VSL_UNARY_FUNC(Sqr, y[i] = a[i] * a[i])
DEFINE_VSL_UNARY_FUNC(Sqrt, y[i] = sqrt(a[i]))
DEFINE_VSL_UNARY_FUNC(Exp, y[i] = exp(a[i]))
DEFINE_VSL_UNARY_FUNC(Ln, y[i] = log(a[i]))
DEFINE_VSL_UNARY_FUNC(Abs, y[i] = fabs(a[i]))

// A simple way to define the vsl unary functions with singular parameter b.
// The operation should be in the form e.g. y[i] = pow(a[i], b)
#define DEFINE_VSL_UNARY_FUNC_WITH_PARAM(name, operation)                      \
  template <typename Dtype>                                                    \
  void v##name(const int n, const Dtype *a, const Dtype b, Dtype *y) {         \
    CHECK_GT(n, 0);                                                            \
    CHECK(a);                                                                  \
    CHECK(y);                                                                  \
    parallel_for(0, n, 1, [&](int i) { operation; });                          \
  }                                                                            \
  inline void vs##name(const int n, const float *a, const float b, float *y) { \
    v##name<float>(n, a, b, y);                                                \
  }                                                                            \
  inline void vd##name(const int n, const double *a, const float b,            \
                       double *y) {                                            \
    v##name<double>(n, a, b, y);                                               \
  }

DEFINE_VSL_UNARY_FUNC_WITH_PARAM(Powx, y[i] = pow(a[i], b))

// A simple way to define the vsl binary functions. The operation should
// be in the form e.g. y[i] = a[i] + b[i]
#define DEFINE_VSL_BINARY_FUNC(name, operation)                                \
  template <typename Dtype>                                                    \
  void v##name(const int n, const Dtype *a, const Dtype *b, Dtype *y) {        \
    CHECK_GT(n, 0);                                                            \
    CHECK(a);                                                                  \
    CHECK(b);                                                                  \
    CHECK(y);                                                                  \
    parallel_for(0, n, 1, [&](int i) { operation; });                          \
  }                                                                            \
  inline void vs##name(const int n, const float *a, const float *b,            \
                       float *y) {                                             \
    v##name<float>(n, a, b, y);                                                \
  }                                                                            \
  inline void vd##name(const int n, const double *a, const double *b,          \
                       double *y) {                                            \
    v##name<double>(n, a, b, y);                                               \
  }

DEFINE_VSL_BINARY_FUNC(Add, y[i] = a[i] + b[i])
DEFINE_VSL_BINARY_FUNC(Sub, y[i] = a[i] - b[i])
DEFINE_VSL_BINARY_FUNC(Mul, y[i] = a[i] * b[i])
DEFINE_VSL_BINARY_FUNC(Div, y[i] = a[i] / b[i])

// In addition, MKL comes with an additional function axpby that is not present
// in standard blas. We will simply use a two-step (inefficient, of course) way
// to mimic that.
inline void cblas_saxpby(const int N, const float alpha, const float *X,
                         const int incX, const float beta, float *Y,
                         const int incY) {
  cblas_sscal(N, beta, Y, incY);
  cblas_saxpy(N, alpha, X, incX, Y, incY);
}
inline void cblas_daxpby(const int N, const double alpha, const double *X,
                         const int incX, const double beta, double *Y,
                         const int incY) {
  cblas_dscal(N, beta, Y, incY);
  cblas_daxpy(N, alpha, X, incX, Y, incY);
}

#endif // USE_MKL
#endif // CAFFE_UTIL_MKL_ALTERNATE_H_