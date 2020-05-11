//
// Created by Troy Liu on 2020/5/9.
//

#include "caffe/util/yolo_utils.hpp"
#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

// template<class Dtype>
// void activate_yolo_cpu(int len, int stride, int index, int num_classes,
//                       const Dtype *input_data, Dtype *output_data, Dtype
//                       *scores, ARC arc, bool gaussian_box) {
//  Dtype *tmp_scores = nullptr;
//  if (scores == nullptr && arc != DEFAULT) {
//    tmp_scores = new Dtype[num_classes + 1];
//  }
//  for (int c = 0; c < len; ++c) {                     // bbox with scores
//    int index2 = c * stride + index;
//    //LOG(INFO)<<index2;
//    if (gaussian_box) {
//      if (c == 4 || c == 6) {
//        output_data[index2 + 0] = (input_data[index2 + 0]);
//      } else {
//        if (c > 7) {
//          //LOG(INFO) << c - 5;
//          scores[c - 8] = sigmoid(input_data[index2 + 0]);
//        } else {
//          output_data[index2 + 0] = sigmoid(input_data[index2 + 0]);
//        }
//      }
//    } else {
//      if (c < 4) {                                                  // 坐标
//        if (c < 2) {
//          // cx, cy
//          output_data[index2] = sigmoid(input_data[index2]);
//        } else {
//          // width and height
//          output_data[index2] = exp(caffe_cpu_clip<Dtype>(input_data[index2],
//          -6, 6));
//        }
//      } else {                                                      //
//      类别分数（置信度）
//        if (arc == DEFAULT) {
//          // 默认的 arc
////          if (scores != nullptr) output_data[index2] = scores[c - 4]  =
/// sigmoid(input_data[index2]);        // classes /          else
/// output_data[index2] = tmp_scores[c - 4] = sigmoid(input_data[index2]);
//          output_data[index2] = sigmoid(input_data[index2]);
//        } else if (arc == CE) {
//          // CE arc 使用 softmax 激活
//          // io[..., 4:] = F.softmax(io[..., 4:], dim=4)
//          // io[..., 4] = 1
//          if (scores != nullptr) {
//            caffe_softmax(1 + num_classes, &input_data[index2], stride,
//            scores); output_data[index2] = scores[0] = 1.0;
//          } else {
//            caffe_softmax(1 + num_classes, &input_data[index2], stride,
//            tmp_scores); output_data[index2] = tmp_scores[0] = 1.0;
//          }
//          break;
//        } else {
//          // BCE 激活，（背景也是一类）
//          // torch.sigmoid_(io[..., 5:])
//          // io[..., 4] = 1
//          if (c == 4) {
//            if (scores != nullptr) scores[c - 4] = output_data[index2] =
//            Dtype(1); else tmp_scores[c - 4] = output_data[index2] = Dtype(1);
//          } else {
//            if (scores != nullptr) scores[c - 4] = output_data[index2] =
//            sigmoid(input_data[index2]); else tmp_scores[c - 4] =
//            output_data[index2] = sigmoid(input_data[index2]);
//          }
//        }
//      }
//    }
//  }  // 完成激活 （除了 w, h）
//  if (scores == nullptr && arc != DEFAULT) {
//    for (int c = 4; c < len; ++c) {
//      int index2 = c * stride + index;
//      output_data[index2] = tmp_scores[c - 4];
//    }
//    delete[] tmp_scores;
//  }
//}

template <typename Dtype>
void caffe_cpu_channel_softmax(const int len, const int grid_size,
                               const Dtype *input_data, Dtype *output_data) {
  Dtype *act = new Dtype[len];
  Dtype tmp = 0;
#pragma omp parallel for default(none) private(act, tmp)
  for (int s = 0; s < grid_size; ++s) {
    auto *pixel = input_data + s;
    auto *out_pixel = output_data + s;
    tmp = pixel[0];
    for (int c = 0; c < len; ++c) {
      act[c] = pixel[c * grid_size];
      if (act[c] > tmp)
        tmp = act[c];
    }
    caffe_sub(len, act, tmp, act);
    caffe_exp(len, act, act);
    tmp = caffe_cpu_asum(len, act);
    for (int c = 0; c < len; ++c) {
      act[c] /= tmp;
      out_pixel[c * grid_size] = act[c];
    }
    caffe_set(len, Dtype(0.0), act);
  }
  delete[] act;
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
    caffe_exp(2 * stride, input_data + index, output_data + index);
    index += 2 * stride; // conf, clz
    if (arc == DEFAULT) {
      caffe_sigmoid((num_classes + 1) * stride, input_data + index,
                    output_data + index);
    } else if (arc == CE) {
      caffe_cpu_channel_softmax(num_classes + 1, stride, input_data + index,
                                output_data + index);
      if (!is_train) {
        caffe_set<Dtype>(stride, 1.0, output_data + index);
      }
    } else if (arc == BCE) {
      if (!is_train) {
        caffe_set<Dtype>(stride, 1.0, output_data + index);
        index += stride;
      }
      caffe_sigmoid(num_classes * stride, input_data + index,
                    output_data + index);
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