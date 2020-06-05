//
// Created by Troy Liu on 6/5/2020.
//

#include "caffe/util/yolo_utils.hpp"

namespace caffe {
template <class Dtype>
void activate_yolo_gpu(int stride, int index, int num_classes,
                       const Dtype *input_data, Dtype *output_data, ARC arc,
                       bool gaussian_box, bool is_train, float xy_scale) {
  if (gaussian_box) {
    caffe_gpu_logistic_activate(4 * stride, input_data + index,
                                output_data + index);
    index += 4 * stride;
    caffe_copy(stride, input_data + index, output_data + index);
    index += 1 * stride;
    caffe_gpu_logistic_activate(stride, input_data + index,
                                output_data + index);
    index += 1 * stride;
    caffe_copy(stride, input_data + index, output_data + index);
    index += 1 * stride;
    caffe_gpu_logistic_activate(stride, input_data + index,
                                output_data + index);
    index += 1 * stride;
    caffe_gpu_logistic_activate((num_classes + 1) * stride, input_data + index,
                                output_data + index);
  } else {
    caffe_gpu_logistic_activate(2 * stride, input_data + index,
                                output_data + index);
    index += 2 * stride; // w, h
    // caffe_exp(2 * stride, input_data + index, output_data + index);
    caffe_copy(2 * stride, input_data + index, output_data + index);
    index += 2 * stride; // conf, clz
    if (arc == DEFAULT) {
      caffe_gpu_logistic_activate((num_classes + 1) * stride,
                                  input_data + index, output_data + index);
    } else if (arc == CE) {
      // TODO softmax activate
      // CE arc 使用 softmax 激活
      // io[..., 4:] = F.softmax(io[..., 4:], dim=4)
      // io[..., 4] = 1
      // caffe_cpu_channel_softmax(num_classes + 1, stride, input_data + index,
      //                          output_data + index);
      // if (!is_train) {
      //  caffe_set<Dtype>(stride, 1.0, output_data + index);
      //}
    }
  }
}
template void activate_yolo_gpu(int stride, int index, int num_classes,
                                const float *input_data, float *output_data,
                                ARC arc, bool gaussian_box, bool is_train,
                                float xy_scale);
template void activate_yolo_gpu(int stride, int index, int num_classes,
                                const double *input_data, double *output_data,
                                ARC arc, bool gaussian_box, bool is_train,
                                float xy_scale);

} // namespace caffe