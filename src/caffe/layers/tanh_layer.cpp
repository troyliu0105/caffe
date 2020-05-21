// TanH neuron activation function layer.
// Adapted from ReLU layer code written by Yangqing Jia

#include <vector>

#include "caffe/layers/tanh_layer.hpp"

namespace caffe {

template <typename Dtype>
void TanHLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                   const vector<Blob<Dtype> *> &top) {
  const Dtype *bottom_data = bottom[0]->cpu_data();
  Dtype *top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  caffe_tanh(count, bottom_data, top_data);
}

template <typename Dtype>
void TanHLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *> &top,
                                    const vector<bool> &propagate_down,
                                    const vector<Blob<Dtype> *> &bottom) {
  if (propagate_down[0]) {
    const Dtype *top_data = top[0]->cpu_data();
    const Dtype *top_diff = top[0]->cpu_diff();
    Dtype *bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    FOR_LOOP_WITH_PREPARE(
        count, i,
        {
          tanhx = top_data[i];
          bottom_diff[i] = top_diff[i] * (1 - tanhx * tanhx);
        },
        Dtype tanhx)
  }
}

#ifdef CPU_ONLY
STUB_GPU(TanHLayer);
#endif

INSTANTIATE_CLASS(TanHLayer);

} // namespace caffe
