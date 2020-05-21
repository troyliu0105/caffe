//
// Created by Troy Liu on 2020/5/20.
//

#include "caffe/layers/matmul_layer.hpp"

namespace caffe {
template <typename Dtype>
void MatMulLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                 const vector<Blob<Dtype> *> &top) {
  CHECK(bottom.size() == 2) << "Must provide 2 bottoms";
  vector<int> a_shape = bottom[0]->shape();
  vector<int> b_shape = bottom[1]->shape();
  CHECK(a_shape.size() == 3 || a_shape.size() == 2);
  CHECK(b_shape.size() == 3 || a_shape.size() == 2);
  CHECK(a_shape.size() == b_shape.size());
  int batch;
  if (a_shape.size() == 3) {
    batch = a_shape[0];
    M_ = a_shape[1];
    N_ = b_shape[2];
    K_ = a_shape[2];
    Blob<Dtype> *top_blob = top[0];
    top_blob->Reshape({batch, M_, N_});
    has_batch_ = true;
  } else {
    M_ = a_shape[0];
    N_ = b_shape[1];
    K_ = a_shape[1];
    Blob<Dtype> *top_blob = top[0];
    top_blob->Reshape({M_, N_});
  }
}
template <typename Dtype>
void MatMulLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                     const vector<Blob<Dtype> *> &top) {
  const Dtype *a = bottom[0]->cpu_data();
  const Dtype *b = bottom[1]->cpu_data();
  Dtype *y = top[0]->mutable_cpu_data();
  if (has_batch_) {
    int batch_size = top[0]->shape(0);
    int y_inner = top[0]->count(1);
    int a_inner = bottom[0]->count(1);
    int b_inner = bottom[1]->count(1);
    for (int i = 0; i < batch_size; ++i) {
      caffe_blas_gemm(CblasNoTrans, CblasNoTrans, M_, N_, K_, Dtype(1.0),
                      a + a_inner * i, b + b_inner * i, Dtype(0.0),
                      y + y_inner * i);
    }
  } else {
    caffe_blas_gemm(CblasNoTrans, CblasNoTrans, M_, N_, K_, Dtype(1.0), a, b,
                    Dtype(0.0), y);
  }
}
template <typename Dtype>
void MatMulLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *> &top,
                                      const vector<bool> &propagate_down,
                                      const vector<Blob<Dtype> *> &bottom) {
  const Dtype *top_diff = top[0]->cpu_diff();
  const Dtype *a_data = bottom[0]->cpu_data();
  const Dtype *b_data = bottom[1]->cpu_data();
  Dtype *a_diff = bottom[0]->mutable_cpu_diff();
  Dtype *b_diff = bottom[1]->mutable_cpu_diff();
  if (has_batch_) {
    int batch_size = top[0]->shape(0);
    int y_inner = top[0]->count(1);
    int a_inner = bottom[0]->count(1);
    int b_inner = bottom[1]->count(1);
    for (int i = 0; i < batch_size; ++i) {
      caffe_blas_gemm(CblasNoTrans, CblasTrans, M_, K_, N_, Dtype(1.0),
                      top_diff + y_inner * i, b_data + b_inner * i, Dtype(0.0),
                      a_diff + a_inner * i);
      caffe_blas_gemm(CblasTrans, CblasNoTrans, K_, N_, M_, Dtype(1.0),
                      a_data + a_inner * i, top_diff + y_inner * i, Dtype(0.0),
                      b_diff + b_inner * i);
    }
  } else {
    caffe_blas_gemm(CblasNoTrans, CblasTrans, M_, K_, N_, Dtype(1.0), top_diff,
                    b_data, Dtype(0.0), a_diff);
    caffe_blas_gemm(CblasTrans, CblasNoTrans, K_, N_, M_, Dtype(1.0), a_data,
                    top_diff, Dtype(0.0), b_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(MatMulLayer);
#endif

INSTANTIATE_CLASS(MatMulLayer);
REGISTER_LAYER_CLASS(MatMul);

} // namespace caffe