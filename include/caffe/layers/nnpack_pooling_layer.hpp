//
// Created by Troy Liu on 2019/11/19.
//

#ifndef CAFFE_INCLUDE_CAFFE_LAYERS_NNPACK_POOLING_LAYER_HPP
#define CAFFE_INCLUDE_CAFFE_LAYERS_NNPACK_POOLING_LAYER_HPP

#pragma once

#include "caffe/layers/pooling_layer.hpp"

namespace caffe {
#ifdef USE_NNPACK
template<typename Dtype>
class NNPackPoolingLayer : public PoolingLayer<Dtype> {
public:
  explicit NNPackPoolingLayer(const LayerParameter &param)
      : PoolingLayer<Dtype>(param) {}
  virtual inline const char *type() const {
    return "NNPackPooling";
  }
  virtual void Forward_cpu(
      const vector<Blob < Dtype> *
  >& bottom,
  const vector<Blob < Dtype>*>& top);
};
#endif
}

#endif //CAFFE_INCLUDE_CAFFE_LAYERS_NNPACK_POOLING_LAYER_HPP
