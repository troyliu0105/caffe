//
// Created by troyl on 5/24/2020.
//

#ifndef CAFFE_INCLUDE_CAFFE_LAYERS_ONE_HOT_LAYER_HPP
#define CAFFE_INCLUDE_CAFFE_LAYERS_ONE_HOT_LAYER_HPP

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
template <typename Dtype>
class OneHotLayer : public Layer<Dtype> {
public:
  explicit OneHotLayer(const LayerParameter &param) : Layer<Dtype>(param) {}
  void LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                  const vector<Blob<Dtype> *> &top) override;
  void Reshape(const vector<Blob<Dtype> *> &bottom,
               const vector<Blob<Dtype> *> &top) override;
  const char *type() const override { return "OneHot"; };
  int ExactNumBottomBlobs() const override { return 1; };

protected:
  void Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                   const vector<Blob<Dtype> *> &top) override;
  void Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                   const vector<Blob<Dtype> *> &top) override;
  void Backward_cpu(const vector<Blob<Dtype> *> &top,
                    const vector<bool> &propagate_down,
                    const vector<Blob<Dtype> *> &bottom) override {
    if (propagate_down[0]) {
      NOT_IMPLEMENTED;
    }
  };
  void Backward_gpu(const vector<Blob<Dtype> *> &top,
                    const vector<bool> &propagate_down,
                    const vector<Blob<Dtype> *> &bottom) override;

private:
  int axis_;
  int num_class_;
  int outer_num_, inner_num_;
};
} // namespace caffe

#endif // CAFFE_INCLUDE_CAFFE_LAYERS_ONE_HOT_LAYER_HPP
