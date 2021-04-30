#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/matmul_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MatMulLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                                     const vector<Blob<Dtype> *> &top) {
  const Dtype *a = bottom[0]->gpu_data();
  const Dtype *b = bottom[1]->gpu_data();
  Dtype *y = top[0]->mutable_gpu_data();
  if (has_batch_) {
    int batch_size = top[0]->shape(0);
    int y_inner = top[0]->count(1);
    int a_inner = bottom[0]->count(1);
    int b_inner = bottom[1]->count(1);
    for (int i = 0; i < batch_size; ++i) {
      caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, M_, N_, K_, Dtype(1.0),
                     a + a_inner * i, b + b_inner * i, Dtype(0.0),
                     y + y_inner * i);
    }
  } else {
    caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, M_, N_, K_, Dtype(1.0), a, b,
                   Dtype(0.0), y);
  }
}

template <typename Dtype>
void MatMulLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype> *> &top,
                                      const vector<bool> &propagate_down,
                                      const vector<Blob<Dtype> *> &bottom) {
  const Dtype *top_diff = top[0]->gpu_diff();
  const Dtype *a_data = bottom[0]->gpu_data();
  const Dtype *b_data = bottom[1]->gpu_data();
  Dtype *a_diff = bottom[0]->mutable_gpu_diff();
  Dtype *b_diff = bottom[1]->mutable_gpu_diff();
  if (has_batch_) {
    int batch_size = top[0]->shape(0);
    int y_inner = top[0]->count(1);
    int a_inner = bottom[0]->count(1);
    int b_inner = bottom[1]->count(1);
    for (int i = 0; i < batch_size; ++i) {
      caffe_gpu_gemm(CblasNoTrans, CblasTrans, M_, K_, N_, Dtype(1.0),
                     top_diff + y_inner * i, b_data + b_inner * i, Dtype(0.0),
                     a_diff + a_inner * i);
      caffe_gpu_gemm(CblasTrans, CblasNoTrans, K_, N_, M_, Dtype(1.0),
                     a_data + a_inner * i, top_diff + y_inner * i, Dtype(0.0),
                     b_diff + b_inner * i);
    }
  } else {
    caffe_gpu_gemm(CblasNoTrans, CblasTrans, M_, K_, N_, Dtype(1.0), top_diff,
                   b_data, Dtype(0.0), a_diff);
    caffe_gpu_gemm(CblasTrans, CblasNoTrans, K_, N_, M_, Dtype(1.0), a_data,
                   top_diff, Dtype(0.0), b_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(MatMulLayer);

} // namespace caffe