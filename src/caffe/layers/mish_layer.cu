//
// Created by troyl on 5/11/2020.
//
#include "caffe/layers/mish_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void MishForward(const int n, const Dtype *in, Dtype *out) {
  CUDA_KERNEL_LOOP(index, n) {
    Dtype x = in[index];
    out[index] = x * tanh(log(1 + exp(x)));
  }
}

template <typename Dtype>
void MishLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                                   const vector<Blob<Dtype> *> &top) {
  const Dtype *bottom_data = bottom[0]->gpu_data();
  Dtype *top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  MishForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void MishBackward(const int n, const Dtype *in_diff,
                             const Dtype *in_data, Dtype *out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    Dtype x = in_data[index];
    Dtype w = 4 * (x + 1) + (4 * std::exp(2 * x)) + std::exp(3 * x) +
              std::exp(x) * (4 * x + 6);
    Dtype sigma = 2 * std::exp(x) + std::exp(2 * x) + 2;
    out_diff[index] = (std::exp(x) * w / std::pow(sigma, 2)) * in_diff[index];
  }
}

template <typename Dtype>
void MishLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype> *> &top,
                                    const vector<bool> &propagate_down,
                                    const vector<Blob<Dtype> *> &bottom) {
  if (propagate_down[0]) {
    const Dtype *bottom_data = bottom[0]->gpu_data();
    const Dtype *top_diff = top[0]->gpu_diff();
    Dtype *bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    MishBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, bottom_data, bottom_diff);
    CUDA_POST_KERNEL_CHECK;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(MishLayer);

} // namespace caffe