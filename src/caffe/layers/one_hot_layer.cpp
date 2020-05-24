//
// Created by troyl on 5/24/2020.
//

#include "caffe/layers/one_hot_layer.hpp"

namespace caffe {
template <typename Dtype>
void OneHotLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                    const vector<Blob<Dtype> *> &top) {
  Layer::LayerSetUp(bottom, top);
  OneHotParameter param = this->layer_param().one_hot_param();
  num_class_ = param.num_class();
}
template <typename Dtype>
void OneHotLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                 const vector<Blob<Dtype> *> &top) {
  axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.one_hot_param().axis());
  vector<int> shape;
  std::copy(bottom[0]->shape().cbegin(), bottom[0]->shape().cend(),
            std::back_inserter(shape));
  CHECK_LT(axis_, shape.size()) << "axis should be smaller than dimension";
  CHECK_EQ(1, shape[axis_]) << "can only expand 1-ele dim to onehot";
  outer_num_ = bottom[0]->count(0, axis_);
  inner_num_ = bottom[0]->count(axis_);
  shape[axis_] = num_class_;
  top[0]->Reshape(shape);
}
template <typename Dtype>
void OneHotLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                     const vector<Blob<Dtype> *> &top) {
  const int count = top[0]->count();
  const int dim = count / outer_num_;
  const Dtype *bottom_data = bottom[0]->cpu_data();
  Dtype *top_data = top[0]->mutable_cpu_data();
  Blob<Dtype> *t = top[0];
  int label;
  caffe_set(count, Dtype(0.), top_data);
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; ++j) {
      label = static_cast<int>(bottom_data[i * inner_num_ + j]);
      top_data[i * dim + label * inner_num_ + j] = 1;
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(OneHotLayer);
#endif

INSTANTIATE_CLASS(OneHotLayer);
REGISTER_LAYER_CLASS(OneHot);
} // namespace caffe