//
// Created by Troy Liu on 2020/5/20.
//

#ifndef CAFFE_INCLUDE_CAFFE_LAYERS_MATMUL_LAYER_HPP
#define CAFFE_INCLUDE_CAFFE_LAYERS_MATMUL_LAYER_HPP

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
template <typename Dtype>
class MatMulLayer : public Layer<Dtype> {
public:
  explicit MatMulLayer(const LayerParameter &param) : Layer<Dtype>(param) {}

  const char *type() const override { return "MatMul"; }
  int ExactNumBottomBlobs() const override { return 2; }
  int ExactNumTopBlobs() const override { return 1; }
  void Reshape(const vector<Blob<Dtype> *> &bottom,
  const vector<Blob<Dtype> *> &top);

protected:
  void Forward_cpu(const vector<Blob<Dtype> *> &bottom,
  const vector<Blob<Dtype> *> &top);
  void Forward_gpu(const vector<Blob<Dtype> *> &bottom,
  const vector<Blob<Dtype> *> &top);
  void Backward_gpu(const vector<Blob<Dtype> *> &top,
  const vector<bool> &propagate_down,
  const vector<Blob<Dtype> *> &bottom);
  void Backward_cpu(const vector<Blob<Dtype> *> &top,
  const vector<bool> &propagate_down,
  const vector<Blob<Dtype> *> &bottom);

private:
  int N_;
  int K_;
  int M_;
  bool has_batch_;
};
} // namespace caffe

#endif // CAFFE_INCLUDE_CAFFE_LAYERS_MATMUL_LAYER_HPP