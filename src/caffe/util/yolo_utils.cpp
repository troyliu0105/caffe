//
// Created by Troy Liu on 2020/5/9.
//

#include "caffe/util/yolo_utils.hpp"

namespace caffe {

template <typename Dtype>
void caffe_cpu_channel_softmax(const int len, const int grid_size,
                               const Dtype *input_data, Dtype *output_data) {
  parallel_for(grid_size, [&](int s) {
    auto *pixel = input_data + s;
    auto *out_pixel = output_data + s;
    caffe_softmax(len, pixel, grid_size, out_pixel);
  });
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
    caffe_sigmoid((num_classes + 1) * stride, input_data + index,
                  output_data + index);
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
template void activate_yolo_cpu<float>(int, int, int, const float *, float *,
                                       ARC, bool, bool, float);
template void activate_yolo_cpu<double>(int, int, int, const double *, double *,
                                        ARC, bool, bool, float);

template <typename Dtype>
void get_region_box(box *b, const Dtype *x, vector<Dtype> biases, int n,
                    int index, int i, int j, int lw, int lh, int w, int h,
                    int stride) {
  b->x = ((i + (x[index + 0 * stride])) / lw);
  b->y = ((j + (x[index + 1 * stride])) / lh);
  b->w = (exp(x[index + 2 * stride]) * biases[2 * n] / (w));
  b->h = (exp(x[index + 3 * stride]) * biases[2 * n + 1] / (h));
}
template void get_region_box(box *b, const float *x, vector<float> biases,
                             int n, int index, int i, int j, int lw, int lh,
                             int w, int h, int stride);
template void get_region_box(box *b, const double *x, vector<double> biases,
                             int n, int index, int i, int j, int lw, int lh,
                             int w, int h, int stride);

template <typename Dtype>
void get_region_box(vector<Dtype> &b, const Dtype *x, vector<Dtype> biases,
                    int n, int index, int i, int j, int lw, int lh, int w,
                    int h, int stride) {

  b.clear();
  b.push_back((i + (x[index + 0 * stride])) / lw);
  b.push_back((j + (x[index + 1 * stride])) / lh);
  b.push_back(exp(x[index + 2 * stride]) * biases[2 * n] / (w));
  b.push_back(exp(x[index + 3 * stride]) * biases[2 * n + 1] / (h));
}
template void get_region_box(vector<float> &b, const float *x,
                             vector<float> biases, int n, int index, int i,
                             int j, int lw, int lh, int w, int h, int stride);
template void get_region_box(vector<double> &b, const double *x,
                             vector<double> biases, int n, int index, int i,
                             int j, int lw, int lh, int w, int h, int stride);

template <typename Dtype>
void get_gaussian_yolo_box(box *b, const Dtype *x, vector<Dtype> biases, int n,
                           int index, int i, int j, int lw, int lh, int w,
                           int h, int stride) {

  b->x = (i + (x[index + 0 * stride])) / lw;
  b->y = (j + (x[index + 2 * stride])) / lh;
  b->w = exp(x[index + 4 * stride]) * biases[2 * n] / (w);
  b->h = exp(x[index + 6 * stride]) * biases[2 * n + 1] / (h);
}
template void get_gaussian_yolo_box(box *b, const float *x,
                                    vector<float> biases, int n, int index,
                                    int i, int j, int lw, int lh, int w, int h,
                                    int stride);
template void get_gaussian_yolo_box(box *b, const double *x,
                                    vector<double> biases, int n, int index,
                                    int i, int j, int lw, int lh, int w, int h,
                                    int stride);

template <typename Dtype>
void get_gaussian_yolo_box(vector<Dtype> &b, const Dtype *x,
                           vector<Dtype> biases, int n, int index, int i, int j,
                           int lw, int lh, int w, int h, int stride) {

  b.clear();
  b.push_back((i + (x[index + 0 * stride])) / lw);
  b.push_back((j + (x[index + 2 * stride])) / lh);
  b.push_back(exp(x[index + 4 * stride]) * biases[2 * n] / (w));
  b.push_back(exp(x[index + 6 * stride]) * biases[2 * n + 1] / (h));
}
template void get_gaussian_yolo_box(vector<float> &b, const float *x,
                                    vector<float> biases, int n, int index,
                                    int i, int j, int lw, int lh, int w, int h,
                                    int stride);
template void get_gaussian_yolo_box(vector<double> &b, const double *x,
                                    vector<double> biases, int n, int index,
                                    int i, int j, int lw, int lh, int w, int h,
                                    int stride);

template <typename Dtype>
void build_target_idx(Blob<Dtype> *input, Blob<Dtype> *label) {}
} // namespace caffe