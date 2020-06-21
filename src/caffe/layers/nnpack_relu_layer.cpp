//
// Created by Troy Liu on 2019/11/19.
//

#ifdef USE_NNPACK
#include <algorithm>
#include <vector>

#include "caffe/layers/nnpack_relu_layer.hpp"

namespace caffe {

template <typename Dtype>
void NNPackReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                         const vector<Blob<Dtype> *> &top) {
  return ReLULayer<Dtype>::Forward_cpu(bottom, top);
}

template <>
void NNPackReLULayer<float>::Forward_cpu(const vector<Blob<float> *> &bottom,
                                         const vector<Blob<float> *> &top) {
  const auto batch_size = static_cast<size_t>(bottom[0]->num());
  const auto count = static_cast<size_t>(bottom[0]->count());
  float negative_slope = this->layer_param_.relu_param().negative_slope();
  const nnp_status status = nnp_relu_output(
      batch_size, count / batch_size, bottom[0]->cpu_data(),
      top[0]->mutable_cpu_data(), negative_slope, Caffe::nnpack_threadpool());
  CHECK_EQ(nnp_status_success, status);
}

template <typename Dtype>
void NNPackReLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *> &top,
                                          const vector<bool> &propagate_down,
                                          const vector<Blob<Dtype> *> &bottom) {
  return ReLULayer<Dtype>::Backward_cpu(top, propagate_down, bottom);
}

template <>
void NNPackReLULayer<float>::Backward_cpu(const vector<Blob<float> *> &top,
                                          const vector<bool> &propagate_down,
                                          const vector<Blob<float> *> &bottom) {
  const auto batch_size = static_cast<size_t>(bottom[0]->num());
  const auto count = static_cast<size_t>(bottom[0]->count());
  float negative_slope = this->layer_param_.relu_param().negative_slope();
  if (propagate_down[0]) {
    const nnp_status status = nnp_relu_input_gradient(
        batch_size, count / batch_size, top[0]->cpu_diff(),
        bottom[0]->cpu_data(), bottom[0]->mutable_cpu_diff(), negative_slope,
        Caffe::nnpack_threadpool());
    CHECK_EQ(nnp_status_success, status);
  }
}

INSTANTIATE_CLASS(NNPackReLULayer);

} // namespace caffe
#endif