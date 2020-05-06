//
// Created by Troy Liu on 2019/11/19.
//

#ifndef CAFFE_INCLUDE_CAFFE_LAYERS_NNPACK_INNER_PRODUCT_LAYER_HPP
#define CAFFE_INCLUDE_CAFFE_LAYERS_NNPACK_INNER_PRODUCT_LAYER_HPP

#pragma once

#include "caffe/layers/inner_product_layer.hpp"

namespace caffe {
#ifdef USE_NNPACK
template <typename Dtype>
class NNPackInnerProductLayer : public InnerProductLayer<Dtype> {
public:
  explicit NNPackInnerProductLayer(const LayerParameter &param)
      : InnerProductLayer<Dtype>(param) {}
  virtual inline const char *type() const { return "NNPackInnerProduct"; }
  virtual void Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                           const vector<Blob<Dtype> *> &top);
};
#endif
} // namespace caffe

#endif // CAFFE_INCLUDE_CAFFE_LAYERS_NNPACK_INNER_PRODUCT_LAYER_HPP
