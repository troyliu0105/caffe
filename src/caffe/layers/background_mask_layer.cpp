//
// Created by Troy Liu on 2020/6/14.
//

#include "caffe/layers/background_mask_layer.hpp"
#include <opencv2/opencv.hpp>

namespace caffe {
template <typename Dtype>
void BackgroundMaskLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                            const vector<Blob<Dtype> *> &top) {
  Layer<Dtype>::LayerSetUp(bottom, top);
  temp_scale_ = 2;
}
template <typename Dtype>
void BackgroundMaskLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                         const vector<Blob<Dtype> *> &top) {
  Blob<Dtype> *input_blob = bottom[0];
  Blob<Dtype> *mask_blob = top[0];

  // auto shape = input_blob->shape();
  // shape[1] = 1;
  mask_blob->ReshapeLike(*input_blob);
  // mask_blob->Reshape(shape);
}
template <typename Dtype>
void BackgroundMaskLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  Blob<Dtype> *input_blob = bottom[0];
  Blob<Dtype> *label_blob = bottom[1];
  Blob<Dtype> *mask_blob = top[0];
  const int channel = input_blob->channels();
  const int height = input_blob->height();
  const int width = input_blob->width();

  const int scaled_height = height / temp_scale_;
  const int scaled_width = width / temp_scale_;

  double min, max;
  const Dtype *label_data = label_blob->cpu_data();
  Dtype *mask_data = mask_blob->mutable_cpu_data();
  cv::Mat bbox_mask = cv::Mat::zeros(scaled_height, scaled_width, CV_32FC1);
  auto scaled_size = cv::Size(scaled_width, scaled_height);
  auto size = cv::Size(width, height);
  for (int b = 0; b < input_blob->num(); ++b) {
    Dtype *batch_data = mask_data + mask_blob->offset(b);
    cv::Mat batch_mask(height, width, CV_32FC1, batch_data);
    for (int t = 0; t < 300; ++t) {
      int clz = label_data[b * 300 * 5 + t * 5 + 0];
      Dtype x = label_data[b * 300 * 5 + t * 5 + 1];
      Dtype y = label_data[b * 300 * 5 + t * 5 + 2];
      Dtype w = label_data[b * 300 * 5 + t * 5 + 3];
      Dtype h = label_data[b * 300 * 5 + t * 5 + 4];
      if (!x)
        break;
      cv::resize(bbox_mask, bbox_mask, scaled_size);
      bbox_mask.setTo(cv::Scalar(0.));
      *(reinterpret_cast<float *>(
          bbox_mask.ptr(static_cast<int>(y * scaled_height),
                        static_cast<int>(x * scaled_width)))) = 1.0F;
      float kw = w * scaled_width / 4;
      float kh = h * scaled_height / 4;
      cv::GaussianBlur(bbox_mask, bbox_mask, {0, 0}, kw, kh);
      cv::resize(bbox_mask, bbox_mask, size);
      cv::minMaxLoc(bbox_mask, &min, &max);
      bbox_mask -= min;
      auto range = max - min;
      if (range <= 0)
        range = 1;
      bbox_mask *= 1 / range;
      batch_mask += bbox_mask;
    }
    cv::minMaxLoc(batch_mask, &min, &max);
    batch_mask -= min;
    auto range = max - min;
    if (range <= 0)
      range = 1;
    batch_mask *= 1 / range;
    for (int i = 1; i < channel; ++i) {
      caffe_copy(height * width, batch_data, batch_data + i * height * width);
    }
  }
}

INSTANTIATE_CLASS(BackgroundMaskLayer);
REGISTER_LAYER_CLASS(BackgroundMask);

} // namespace caffe