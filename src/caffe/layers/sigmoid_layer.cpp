#include <cmath>
#include <vector>

#include "caffe/layers/sigmoid_layer.hpp"

namespace caffe {

template <typename Dtype>
inline Dtype sigmoid(Dtype x) {
  return 0.5 * tanh(0.5 * x) + 0.5;
}

template <typename Dtype>
void SigmoidLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                      const vector<Blob<Dtype> *> &top) {
  const Dtype *bottom_data = bottom[0]->cpu_data();
  Dtype *top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  caffe_sigmoid(count, bottom_data, top_data);
}

template <typename Dtype>
void SigmoidLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *> &top,
                                       const vector<bool> &propagate_down,
                                       const vector<Blob<Dtype> *> &bottom) {
  if (propagate_down[0]) {
    const Dtype *top_data = top[0]->cpu_data();
    const Dtype *top_diff = top[0]->cpu_diff();
    Dtype *bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    FOR_LOOP(
        count, i,
        {
          sigmoid_x = top_data[i];
          bottom_diff[i] = top_diff[i] * sigmoid_x * (1. - sigmoid_x);
        },
        Dtype sigmoid_x)
  }
}

#ifdef CPU_ONLY
STUB_GPU(SigmoidLayer);
#endif

INSTANTIATE_CLASS(SigmoidLayer);

} // namespace caffe
