//
// Created by Troy Liu on 2020/5/9.
//

#include "caffe/util/yolo_utils.hpp"
#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void caffe_cpu_channel_softmax(const int len, const int grid_size,
                               const Dtype *input_data, Dtype *output_data) {
  FOR_LOOP(grid_size, s, {
    auto *pixel = input_data + s;
    auto *out_pixel = output_data + s;
    caffe_softmax(len, pixel, grid_size, out_pixel);
  })
}

template void caffe_cpu_channel_softmax(const int len, const int grid_size,
                                        const float *input_data,
                                        float *output_data);
template void caffe_cpu_channel_softmax(const int len, const int grid_size,
                                        const double *input_data,
                                        double *output_data);

template <class Dtype>
void activate_yolo_cpu(int stride, int index, int num_classes,
                       const Dtype *input_data, Dtype *output_data, ARC arc,
                       bool gaussian_box, bool is_train, float xy_scale) {
  if (gaussian_box) {
    caffe_sigmoid(4 * stride, input_data + index, output_data + index);
    index += 4 * stride;
    caffe_copy(stride, input_data + index, output_data + index);
    index += 1 * stride;
    caffe_sigmoid(stride, input_data + index, output_data + index);
    index += 1 * stride;
    caffe_copy(stride, input_data + index, output_data + index);
    index += 1 * stride;
    caffe_sigmoid(stride, input_data + index, output_data + index);
    index += 1 * stride;
    caffe_sigmoid((+1) * stride, input_data + index, output_data + index);
  } else {
    caffe_sigmoid(2 * stride, input_data + index, output_data + index);
    index += 2 * stride; // w, h
    // caffe_exp(2 * stride, input_data + index, output_data + index);
    caffe_copy(2 * stride, input_data + index, output_data + index);
    index += 2 * stride; // conf, clz
    if (arc == DEFAULT) {
      caffe_sigmoid((num_classes + 1) * stride, input_data + index,
                    output_data + index);
    } else if (arc == CE) {
      // CE arc 使用 softmax 激活
      // io[..., 4:] = F.softmax(io[..., 4:], dim=4)
      // io[..., 4] = 1
      caffe_cpu_channel_softmax(num_classes + 1, stride, input_data + index,
                                output_data + index);
      if (!is_train) {
        caffe_set<Dtype>(stride, 1.0, output_data + index);
      }
    }
  }
}

template <typename Dtype>
void build_target_idx(Blob<Dtype> *input, Blob<Dtype> *label) {}
template void activate_yolo_cpu<float>(int, int, int, const float *, float *,
                                       ARC, bool, bool, float);
template void activate_yolo_cpu<double>(int, int, int, const double *, double *,
                                        ARC, bool, bool, float);
} // namespace caffe