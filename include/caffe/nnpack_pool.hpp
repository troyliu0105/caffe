//
// Created by Troy Liu on 2019/11/19.
//

#ifndef CAFFE_INCLUDE_CAFFE_NNPACK_POOL_HPP
#define CAFFE_INCLUDE_CAFFE_NNPACK_POOL_HPP

#pragma once

#include <boost/noncopyable.hpp>
#include <thread>

#include "nnpack.h"
#ifdef USE_MKL
#include <mkl.h>
#endif

namespace caffe {
class NNPACKPool : public boost::noncopyable {
public:
  NNPACKPool() {
#ifdef USE_MKL
    const size_t num_mkl_threads = mkl_get_max_threads();
#else
    // Can we do better here?
    const size_t num_mkl_threads = std::thread::hardware_concurrency();
#endif
    if (num_mkl_threads > 1) {
      pool_ = pthreadpool_create(num_mkl_threads);
    } else {
      pool_ = nullptr;
    }
  }
  ~NNPACKPool() {
    if (pool_) {
      pthreadpool_destroy(pool_);
    }
    pool_ = NULL;
  }

  pthreadpool_t pool() { return pool_; };

private:
  pthreadpool_t pool_;
};

} // namespace caffe

#endif // CAFFE_INCLUDE_CAFFE_NNPACK_POOL_HPP
