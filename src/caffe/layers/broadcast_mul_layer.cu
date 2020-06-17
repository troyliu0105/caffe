/**
 * @brief From jolibrain/caffe
 * Source:
 * https://github.com/jolibrain/caffe/commit/c7eb0529c0d9c48d4b0aa5e927ee44f148a93d98
 */
#include <algorithm>
#include <vector>

#include "caffe/layers/broadcast_mul_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void BroadcastMulForward(const int n, const Dtype *in,
                                    const Dtype *scale, const int scale_dim,
                                    const int inner_dim, Dtype *out) {
  CUDA_KERNEL_LOOP(index, n) {
    const int scale_index = (index / inner_dim) % scale_dim;
    out[index] = in[index] * scale[scale_index];
  }
}

template <typename Dtype>
void BroadcastMulLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                                           const vector<Blob<Dtype> *> &top) {
  int inner_dim, scale_dim;
  const Dtype *bottom_data = NULL, *scale_data = NULL;
  if (bottom[0]->count() == top[0]->count()) {
    bottom_data = bottom[0]->gpu_data();
    scale_data = bottom[1]->gpu_data();
    inner_dim = bottom[0]->width() * bottom[0]->height();
    scale_dim = bottom[1]->count();
  } else {
    bottom_data = bottom[1]->gpu_data();
    scale_data = bottom[0]->gpu_data();
    inner_dim = bottom[1]->width() * bottom[1]->height();
    scale_dim = bottom[0]->count();
  }

  Dtype *top_data = top[0]->mutable_gpu_data();
  int count = top[0]->count();
  BroadcastMulForward<Dtype> // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          top[0]->count(), bottom_data, scale_data, scale_dim, inner_dim,
          top_data);
}

template <typename Dtype>
void BroadcastMulLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down,
    const vector<Blob<Dtype> *> &bottom) {
  int inner_dim, scale_dim;
  const Dtype *bottom_data = NULL, *scale_data = NULL;
  Dtype *bottom_data_diff = NULL, *scale_data_diff = NULL;
  if (bottom[0]->count() == top[0]->count()) {
    bottom_data = bottom[0]->gpu_data();
    scale_data = bottom[1]->gpu_data();
    bottom_data_diff = bottom[0]->mutable_gpu_diff();
    scale_data_diff = bottom[1]->mutable_gpu_diff();
    inner_dim = bottom[0]->width() * bottom[0]->height();
    scale_dim = bottom[1]->count();
  } else {
    bottom_data = bottom[1]->gpu_data();
    scale_data = bottom[0]->gpu_data();
    bottom_data_diff = bottom[1]->mutable_gpu_diff();
    scale_data_diff = bottom[0]->mutable_gpu_diff();
    inner_dim = bottom[1]->width() * bottom[1]->height();
    scale_dim = bottom[0]->count();
  }

  const Dtype *top_data_diff = top[0]->gpu_diff();
  // diff to scale data
  for (int i = 0; i < scale_dim; ++i) {
    caffe_gpu_dot<Dtype>(inner_dim, top_data_diff, bottom_data,
                         scale_data_diff);
    top_data_diff += inner_dim;
    bottom_data += inner_dim;
    ++scale_data_diff;
  }
  top_data_diff = top[0]->gpu_diff();
  int count = top[0]->count();
  // diff to bottom_data
  BroadcastMulForward<Dtype> // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          top[0]->count(), top_data_diff, scale_data, scale_dim, inner_dim,
          bottom_data_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(BroadcastMulLayer);

} // namespace caffe