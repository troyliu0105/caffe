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
void BroadcastMulLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                          const vector<Blob<Dtype> *> &top) {}

template <typename Dtype>
void BroadcastMulLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                       const vector<Blob<Dtype> *> &top) {
  if (bottom[0]->count() > bottom[1]->count())
    top[0]->ReshapeLike(*bottom[0]);
  else
    top[0]->ReshapeLike(*bottom[1]);
}

template <typename Dtype>
void BroadcastMulLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                           const vector<Blob<Dtype> *> &top) {
  const Dtype *bottom_data, *scale_data = nullptr;
  if (bottom[0]->count() == top[0]->count()) {
    bottom_data = bottom[0]->cpu_data();
    scale_data = bottom[1]->cpu_data();
  } else {
    bottom_data = bottom[1]->cpu_data();
    scale_data = bottom[0]->cpu_data();
  }

  Dtype *top_data = top[0]->mutable_cpu_data();
  for (int n = 0; n < top[0]->num(); ++n) {
    for (int c = 0; c < top[0]->channels(); ++c) {
      for (int i = 0; i < top[0]->height() * top[0]->width(); ++i)
        *(top_data++) = *(bottom_data++) * (*scale_data);
      scale_data++;
    }
  }
}

template <typename Dtype>
void BroadcastMulLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down,
    const vector<Blob<Dtype> *> &bottom) {
  const Dtype *bottom_data = NULL, *scale_data = NULL;
  Dtype *bottom_data_diff = NULL, *scale_data_diff = NULL;
  if (bottom[0]->count() == top[0]->count()) {
    bottom_data = bottom[0]->cpu_data();
    scale_data = bottom[1]->cpu_data();

    bottom_data_diff = bottom[0]->mutable_cpu_data();
    scale_data_diff = bottom[1]->mutable_cpu_data();
  } else {
    bottom_data = bottom[1]->cpu_data();
    scale_data = bottom[0]->cpu_data();

    bottom_data_diff = bottom[1]->mutable_cpu_data();
    scale_data_diff = bottom[0]->mutable_cpu_data();
  }

  const Dtype *top_data_diff = top[0]->mutable_cpu_diff();
  for (int n = 0; n < top[0]->num(); ++n) {
    for (int c = 0; c < top[0]->channels(); ++c) {
      Dtype scale_diff = 0;
      for (int i = 0; i < top[0]->height() * top[0]->width(); ++i) {
        scale_diff += *top_data_diff * (*bottom_data);
        *bottom_data_diff = *top_data_diff * (*scale_data);
        bottom_data++;
        top_data_diff++;
        bottom_data_diff++;
      }
      scale_data++;
      *(scale_data_diff++) = scale_diff;
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(BroadcastMulLayer);
#endif

INSTANTIATE_CLASS(BroadcastMulLayer);
REGISTER_LAYER_CLASS(BroadcastMul);
} // namespace caffe