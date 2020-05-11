//
// Created by troyl on 5/11/2020.
//

#ifndef CAFFE_INCLUDE_CAFFE_LAYERS_MISH_LAYER_HPP
#define CAFFE_INCLUDE_CAFFE_LAYERS_MISH_LAYER_HPP

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {
template <typename Dtype>
class MishLayer : public NeuronLayer<Dtype> {
public:
  explicit MishLayer(const LayerParameter &param) : NeuronLayer<Dtype>(param) {}
  virtual inline const char *type() const { return "Mish"; }

protected:
  void Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                   const vector<Blob<Dtype> *> &top) override;
  void Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                   const vector<Blob<Dtype> *> &top) override;
  void Backward_cpu(const vector<Blob<Dtype> *> &top,
                    const vector<bool> &propagate_down,
                    const vector<Blob<Dtype> *> &bottom) override;
  void Backward_gpu(const vector<Blob<Dtype> *> &top,
                    const vector<bool> &propagate_down,
                    const vector<Blob<Dtype> *> &bottom) override;
};
} // namespace caffe
#endif // CAFFE_INCLUDE_CAFFE_LAYERS_MISH_LAYER_HPP
