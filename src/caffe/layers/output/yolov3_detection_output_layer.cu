#include <cmath>
#include <vector>

#include "caffe/layers/yolov3_detection_output_layer.hpp"
#include "caffe/util/bbox_util.hpp"
#include "caffe/util/yolo_utils.hpp"

namespace caffe {
template <typename Dtype>
void Yolov3DetectionOutputLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  int len = 4 + num_class_ + 1;
  if (gaussian_box_)
    len = 8 + num_class_ + 1;
  int mask_offset = 0;

  predicts_.clear();
  auto *class_score = new Dtype[num_class_];

  for (int t = 0; t < bottom.size(); t++) {
    side_w_ = bottom[t]->width();
    side_h_ = bottom[t]->height();
    int stride = side_w_ * side_h_;
    swap_.ReshapeLike(*bottom[t]);
    Dtype *swap_data = swap_.mutable_gpu_data();
    const Dtype *input_data = bottom[t]->gpu_data();
    int nw = side_w_ * anchors_scale_[t];
    int nh = side_h_ * anchors_scale_[t];
    for (int b = 0; b < bottom[t]->num(); b++) {
      for (int n = 0; n < num_; ++n) {
        int index = n * len * stride + b * bottom[0]->count(1);
        activate_yolo_gpu(stride, index, num_class_, input_data, swap_data,
                          DEFAULT, gaussian_box_, false);
      }
      swap_data = swap_.mutable_cpu_data();
      for (int s = 0; s < side_w_ * side_h_; s++) {
        for (int n = 0; n < num_; n++) {
          // LOG(INFO) << bottom[t]->count(1);
          int index = n * len * stride + s + b * bottom[t]->count(1);
          box pred;
          if (gaussian_box_) {
            for (int c = 9; c < len; ++c) {
              int index2 = c * stride + index;
              class_score[c - 9] = (swap_data[index2 + 0]);
            }
          } else {
            for (int c = 5; c < len; ++c) {
              int index2 = c * stride + index;
              class_score[c - 5] = (swap_data[index2 + 0]);
            }
          }
          int y2 = s / side_w_;
          int x2 = s % side_w_;
          Dtype obj_score;
          if (gaussian_box_) {
            Dtype uc_ver =
                4.0 - swap_data[index + 1 * stride] -
                swap_data[index + 1 * stride] - swap_data[index + 3 * stride] -
                swap_data[index + 5 * stride] - swap_data[index + 7 * stride];
            obj_score = swap_data[index + 8 * stride] * uc_ver / 4.0;
          } else {
            obj_score = swap_data[index + 4 * stride];
          }
          PredictionResult<Dtype> predict;
          for (int c = 0; c < num_class_; ++c) {
            class_score[c] *= obj_score;
            if (class_score[c] > confidence_threshold_) {
              if (gaussian_box_) {
                get_gaussian_yolo_box(&pred, swap_data, biases_,
                                      mask_[n + mask_offset], index, x2, y2,
                                      side_w_, side_h_, nw, nh, stride);
              } else {
                get_region_box(&pred, swap_data, biases_,
                               mask_[n + mask_offset], index, x2, y2, side_w_,
                               side_h_, nw, nh, stride);
              }
              predict.x = pred.x;
              predict.y = pred.y;
              predict.w = pred.w;
              predict.h = pred.h;
              predict.classType = c;
              predict.confidence = class_score[c];
              correct_yolo_boxes(predict, side_w_, side_h_, nw, nh, true);
              if (is_predict_valid(predict))
                predicts_.push_back(predict);

              // LOG(INFO) << predict.x << "," << predict.y << "," << predict.w
              // << "," << predict.h; LOG(INFO) << predict.confidence;
            }
          }
        }
      }
    }
    mask_offset += groups_num_;
  }

  delete[] class_score;

  Forward_cpu(bottom, top);
}

template <typename Dtype>
void Yolov3DetectionOutputLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down,
    const vector<Blob<Dtype> *> &bottom) {}

INSTANTIATE_LAYER_GPU_FUNCS(Yolov3DetectionOutputLayer);

} // namespace caffe
