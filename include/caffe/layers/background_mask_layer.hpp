//
// Created by Troy Liu on 2020/6/14.
//

#ifndef CAFFE_INCLUDE_CAFFE_LAYERS_BACKGROUND_MASK_LAYER_HPP
#define CAFFE_INCLUDE_CAFFE_LAYERS_BACKGROUND_MASK_LAYER_HPP

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
template <typename Dtype>
class BackgroundMaskLayer : public Layer<Dtype> {
public:
  explicit BackgroundMaskLayer(const LayerParameter &param)
      : Layer<Dtype>(param) {}
  void LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                  const vector<Blob<Dtype> *> &top) override;
  void Reshape(const vector<Blob<Dtype> *> &bottom,
               const vector<Blob<Dtype> *> &top) override;
  const char *type() const override { return "BackgroundMask"; };
  int ExactNumBottomBlobs() const override { return 2; };
  int ExactNumTopBlobs() const override { return 1; };

protected:
  void Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                   const vector<Blob<Dtype> *> &top) override;
  void Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                   const vector<Blob<Dtype> *> &top) override{};
  void Backward_cpu(const vector<Blob<Dtype> *> &top,
                    const vector<bool> &propagate_down,
                    const vector<Blob<Dtype> *> &bottom) override {
    for (const auto &i : propagate_down) {
      if (i) {
        NOT_IMPLEMENTED;
      }
    }
  }
  void Backward_gpu(const vector<Blob<Dtype> *> &top,
                    const vector<bool> &propagate_down,
                    const vector<Blob<Dtype> *> &bottom) override {
    for (const auto &i : propagate_down) {
      if (i) {
        NOT_IMPLEMENTED;
      }
    }
  }

private:
  int temp_scale_;
};
} // namespace caffe

#endif // CAFFE_INCLUDE_CAFFE_LAYERS_BACKGROUND_MASK_LAYER_HPP
