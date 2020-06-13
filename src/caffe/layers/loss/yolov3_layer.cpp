/*
 * @Author: Eric612
 * @Date:   2018-08-20
 * @https://github.com/eric612/Caffe-YOLOv2-Windows
 * @https://github.com/eric612/MobileNet-YOLO
 * Avisonic
 */
#include "caffe/layers/loss/yolov3_layer.hpp"
#include "caffe/util/yolo_utils.hpp"
#include <cfloat>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif // USE_OPENCV

#if defined(USE_TBB) || defined(USE_OMP)
#include <boost/atomic.hpp>
using Statistic = struct {
  boost::atomic_float_t tot_iou = 0.F;
  boost::atomic_float_t tot_giou = 0.F;
  boost::atomic_float_t tot_diou = 0.F;
  boost::atomic_float_t tot_ciou = 0.F;
  boost::atomic_float_t tot_iou_loss = 0.F;
  boost::atomic_float_t tot_giou_loss = 0.F;
  boost::atomic_float_t tot_diou_loss = 0.F;
  boost::atomic_float_t tot_ciou_loss = 0.F;

  boost::atomic_float_t recall = 0.F;
  boost::atomic_float_t recall75 = 0.F;
  boost::atomic_float_t avg_cat = 0.F;
  boost::atomic_float_t avg_obj = 0.F;
  boost::atomic_float_t avg_anyobj = 0.F;
  boost::atomic_int32_t count = 0;
  boost::atomic_int32_t class_count = 0;
};
#else
using Statistic = struct {
  float tot_iou = 0.F;
  float tot_giou = 0.F;
  float tot_diou = 0.F;
  float tot_ciou = 0.F;
  float tot_iou_loss = 0.F;
  float tot_giou_loss = 0.F;
  float tot_diou_loss = 0.F;
  float tot_ciou_loss = 0.F;

  float recall = 0.F;
  float recall75 = 0.F;
  float avg_cat = 0.F;
  float avg_obj = 0.F;
  float avg_anyobj = 0.F;
  int count = 0;
  int class_count = 0;
};
#endif

#if defined(DEBUG) && defined(DRAW)
static char *CLASSES[21] = {"__background__",
                            "aeroplane",
                            "bicycle",
                            "bird",
                            "boat",
                            "bottle",
                            "bus",
                            "car",
                            "cat",
                            "chair",
                            "cow",
                            "diningtable",
                            "dog",
                            "horse",
                            "motorbike",
                            "person",
                            "pottedplant",
                            "sheep",
                            "sofa",
                            "train",
                            "tvmonitor"};
#endif

namespace caffe {

static inline float fix_nan_inf(float val) {
  if (isnan(val) || isinf(val))
    val = 0;
  return val;
}

template <typename Dtype>
static int int_index(const vector<Dtype> &a, int val, int n) {
  int i;
  for (i = 0; i < n; ++i) {
    if (a[i] == val)
      return i;
  }
  return -1;
}

template <typename Dtype>
static bool compare_yolo_class(const Dtype *swap_data, int class_num,
                               int class_index, int stride, float objectness,
                               int class_id, float conf_thresh) {
  float prob;
  for (int j = 0; j < class_num; ++j) {
    // float prob = objectness * output[class_index + stride*j];
    prob = swap_data[class_index + j * stride];
    if (prob > conf_thresh) {
      return true;
    }
  }
  return false;
}

template <typename Dtype>
static void averages_yolo_deltas(int class_index, int box_index, int stride,
                                 int classes, Dtype *delta) {

  int classes_in_one_box = 0;
  int c;
  for (c = 0; c < classes; ++c) {
    if (delta[class_index + c * stride] > 0)
      classes_in_one_box++;
  }

  if (classes_in_one_box > 0) {
    delta[box_index + 0 * stride] /= classes_in_one_box;
    delta[box_index + 1 * stride] /= classes_in_one_box;
    delta[box_index + 2 * stride] /= classes_in_one_box;
    delta[box_index + 3 * stride] /= classes_in_one_box;
  }
}

template <typename Dtype>
static void delta_region_class_v3(Dtype *input_data, Dtype *&diff,
                                  int class_index, int class_label, int classes,
                                  float scale, Statistic *statistic, int stride,
                                  bool use_focal_loss, float label_smooth_eps) {
  if (diff[class_index + stride * class_label]) {

    float y_true = 1;
    y_true = y_true * (1.F - label_smooth_eps) + 0.5F * label_smooth_eps;
    float result_delta =
        y_true - input_data[class_index + stride * class_label];
    if (!isnan(result_delta) && !isinf(result_delta))
      diff[class_index + stride * class_label] = (-1.0) * scale * result_delta;
    // delta[class_index + stride*class_id] = 1 - output[class_index +
    // stride*class_id];

    if (statistic)
      statistic->avg_cat += input_data[class_index + stride * class_label];

    // diff[class_index + stride*class_label] = (-1.0) * (1 -
    // input_data[class_index + stride*class_label]); *avg_cat +=
    // input_data[class_index + stride*class_label]*scale; LOG(INFO) << "test";
    return;
  }
  if (use_focal_loss) {
    // Reference :
    // https://github.com/AlexeyAB/darknet/blob/master/src/yolo_layer.c
    float alpha = 0.5; // 0.25 or 0.5
    // float gamma = 2;    // hardcoded in many places of the grad-formula

    int ti = class_index + stride * class_label;
    float pt = input_data[ti] + 0.000000000000001F;
    // http://fooplot.com/#W3sidHlwZSI6MCwiZXEiOiItKDEteCkqKDIqeCpsb2coeCkreC0xKSIsImNvbG9yIjoiIzAwMDAwMCJ9LHsidHlwZSI6MTAwMH1d
    float grad = -(1 - pt) *
                 (2 * pt * logf(pt) + pt -
                  1); // http://blog.csdn.net/linmingan/article/details/77885832
    // float grad = (1 - pt) * (2 * pt*logf(pt) + pt - 1);    //
    // https://github.com/unsky/focal-loss

    for (int n = 0; n < classes; ++n) {
      diff[class_index + stride * n] =
          (-1.0) * scale *
          (((n == class_label) ? 1 : 0) - input_data[class_index + n * stride]);

      diff[class_index + stride * n] *= alpha * grad;

      if (n == class_label) {
        if (statistic)
          statistic->avg_cat += input_data[class_index + stride * n];
      }
    }

  } else {
    for (int n = 0; n < classes; ++n) {
      float y_true = ((n == class_label) ? 1.F : 0.F);
      y_true = y_true * (1.F - label_smooth_eps) + 0.5F * label_smooth_eps;
      float result_delta = y_true - input_data[class_index + stride * n];
      if (!isnan(result_delta) && !isinf(result_delta))
        diff[class_index + stride * n] = (-1.0) * scale * result_delta;
      // delta[class_index + stride*class_id] = 1 - output[class_index +
      // stride*class_id];

      if (n == class_label && statistic)
        statistic->avg_cat += input_data[class_index + stride * class_label];
      // diff[class_index + n*stride] = (-1.0) * scale * (((n == class_label) ?
      // 1 : 0)
      // - input_data[class_index + n*stride]);
      //      diff[class_index + n * stride] =
      //          (-1.0) * scale *
      //          (((n == class_label) ? 1 : 0) - input_data[class_index + n *
      //          stride]);
      // std::cout<<diff[class_index+n]<<",";
      //      if (n == class_label) {
      //        *avg_cat += input_data[class_index + n * stride];
      // std::cout<<"avg_cat:"<<input_data[class_index+n]<<std::endl;
      //      }
    }
  }
}

template <typename Dtype>
static ious
delta_region_box(const box &truth, Dtype *x, vector<Dtype> biases, int n,
                 int index, int i, int j, int lw, int lh, int w, int h,
                 Dtype *delta, float scale, int stride, IOU_LOSS iou_loss,
                 float iou_normalizer, float max_delta, bool accumulate) {
  box pred;
  get_region_box(&pred, x, biases, n, index, i, j, lw, lh, w, h, stride);

  ious all_ious = {0};
  all_ious.iou = box_iou(pred, truth);
  all_ious.giou = box_giou(pred, truth);
  all_ious.diou = box_diou(pred, truth);
  all_ious.ciou = box_ciou(pred, truth);

  if (pred.w == 0) {
    pred.w = 1.0;
  }
  if (pred.h == 0) {
    pred.h = 1.0;
  }

  if (!accumulate) {
    delta[index + 0 * stride] = 0;
    delta[index + 1 * stride] = 0;
    delta[index + 2 * stride] = 0;
    delta[index + 3 * stride] = 0;
  }

  if (iou_loss == MSE) // old loss
  {
    float iou = box_iou(pred, truth);
    // LOG(INFO) << pred[0] << "," << pred[1] << "," << pred[2] << "," <<
    // pred[3] << ";"<< truth[0] << "," << truth[1] << "," << truth[2] << "," <<
    // truth[3];
    float tx = truth.x * lw - i;                     // 0.5
    float ty = truth.y * lh - j;                     // 0.5
    float tw = log(truth.w * w / biases[2 * n]);     // truth[2]=biases/w tw = 0
    float th = log(truth.h * h / biases[2 * n + 1]); // th = 0

    // delta[index + 0] = (-1) * scale * (tx - sigmoid(x[index + 0 * stride])) *
    // sigmoid(x[index + 0 * stride]) * (1 - sigmoid(x[index + 0 * stride]));
    // delta[index + 1 * stride] = (-1) * scale * (ty - sigmoid(x[index + 1 *
    // stride])) * sigmoid(x[index + 1 * stride]) * (1 - sigmoid(x[index + 1 *
    // stride]));
    delta[index + 0 * stride] += (-1) * scale * (tx - x[index + 0 * stride]);
    delta[index + 1 * stride] += (-1) * scale * (ty - x[index + 1 * stride]);
    delta[index + 2 * stride] += (-1) * scale * (tw - x[index + 2 * stride]);
    delta[index + 3 * stride] += (-1) * scale * (th - x[index + 3 * stride]);
  } else {
    // Reference code :
    // https://github.com/AlexeyAB/darknet/blob/master/src/yolo_layer.c

    // https://github.com/generalized-iou/g-darknet
    // https://arxiv.org/abs/1902.09630v2
    // https://giou.stanford.edu/

    // i - step in layer width
    // j - step in layer height
    //  Returns a box in absolute coordinates
    // box pred = get_yolo_box(x, biases, n, index, i, j, lw, lh, w, h, stride);

    all_ious.dx_iou = dx_box_iou(pred, truth, iou_loss);

    // jacobian^t (transpose)
    float dx = all_ious.dx_iou.dt;
    float dy = all_ious.dx_iou.db;
    float dw = all_ious.dx_iou.dl;
    float dh = all_ious.dx_iou.dr;
    // predict exponential, apply gradient of e^delta_t ONLY for w,h
    dw *= exp(x[index + 2 * stride]);
    dh *= exp(x[index + 3 * stride]);

    // normalize iou weight
    dx *= iou_normalizer;
    dy *= iou_normalizer;
    dw *= iou_normalizer;
    dh *= iou_normalizer;

    dx = fix_nan_inf(dx);
    dy = fix_nan_inf(dy);
    dw = fix_nan_inf(dw);
    dh = fix_nan_inf(dh);

    if (max_delta != FLT_MAX) {
      dx = caffe_clip(dx, -max_delta, max_delta);
      dy = caffe_clip(dy, -max_delta, max_delta);
      dw = caffe_clip(dw, -max_delta, max_delta);
      dh = caffe_clip(dh, -max_delta, max_delta);
    }

    // accumulate delta
    delta[index + 0 * stride] += -dx;
    delta[index + 1 * stride] += -dy;
    delta[index + 2 * stride] += -dw;
    delta[index + 3 * stride] += -dh;
  }
  return all_ious;
}
template <typename Dtype>
void Yolov3Layer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                    const vector<Blob<Dtype> *> &top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  Yolov3Parameter param = this->layer_param_.yolov3_param();
  iter_ = 0;
  time_count_ = 0;
  class_count_ = 0;
  display_ = param.display();
  num_class_ = param.num_class(); // 20
  num_ = param.num();             // 5
  side_w_ = bottom[0]->width();
  side_h_ = bottom[0]->height();
  anchors_scale_ = param.anchors_scale();
  object_scale_ = param.object_scale();     // 5.0
  noobject_scale_ = param.noobject_scale(); // 1.0
  class_scale_ = param.class_scale();       // 1.0
  coord_scale_ = param.coord_scale();       // 1.0
  thresh_ = param.thresh();                 // 0.6

  use_extra_matched_anchor_ = param.use_extra_matched_anchor();
  objectness_smooth_ = param.objectness_smooth();
  use_logic_gradient_ = param.use_logic_gradient();
  use_focal_loss_ = param.use_focal_loss();
  iou_loss_ = (IOU_LOSS)param.iou_loss();

  iou_normalizer_ = param.iou_normalizer();
  iou_thresh_ = param.iou_thresh();
  max_delta_ = param.max_delta();
  accumulate_ = param.accumulate();
  label_smooth_eps_ = param.label_smooth_eps();
  std::move(param.biases().begin(), param.biases().end(),
            std::back_inserter(biases_));
  std::move(param.mask().begin(), param.mask().end(),
            std::back_inserter(mask_));
  biases_size_ = param.biases_size() / 2;
  int input_count =
      bottom[0]->count(1); // h*w*n*(classes+coords+1) = 13*13*5*(20+4+1)
  int label_count = bottom[1]->count(1); // 30*5-
  // outputs: classes, iou, coordinates
  int tmp_input_count =
      side_w_ * side_h_ * num_ *
      (4 + num_class_ +
       1); // 13*13*5*(20+4+1) label: isobj, class_label, coordinates
  int tmp_label_count = 300 * num_;
  CHECK_EQ(input_count, tmp_input_count);
  // CHECK_EQ(label_count, tmp_label_count);
}

template <typename Dtype>
void Yolov3Layer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                 const vector<Blob<Dtype> *> &top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  diff_.ReshapeLike(*bottom[0]);
  //  real_diff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void Yolov3Layer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                     const vector<Blob<Dtype> *> &top) {
  side_w_ = bottom[0]->width();
  side_h_ = bottom[0]->height();
  // LOG(INFO)<<"iou loss" << iou_loss_<<","<<GIOU;
  // LOG(INFO) << side_*anchors_scale_;
  if (diff_.width() != side_w_ || diff_.height() != side_h_) {
    diff_.ReshapeLike(*bottom[0]);
  }
  Dtype *diff = diff_.mutable_cpu_data();
  caffe_set<Dtype>(diff_.count(), 0.0, diff);

  //  Dtype avg_anyobj(0.0), avg_obj(0.0), avg_iou(0.0), avg_cat(0.0),
  //      avg_iou_loss(0.0), recall(0.0), recall75(0.0), loss(0.0);
  Statistic statistic;

  const Dtype *input_data = bottom[0]->cpu_data();
  const Dtype *label_data = bottom[1]->cpu_data(); //[label,x,y,w,h]
  if (swap_.width() != bottom[0]->width()) {
    swap_.ReshapeLike(*bottom[0]);
  }
  Dtype *swap_data = swap_.mutable_cpu_data();
  int len = 4 + num_class_ + 1;
  int stride = side_w_ * side_h_;
  // skip activate when this function invoked from ForwardGPU
  if (Caffe::mode() == Caffe::CPU) {
    for (int b = 0; b < bottom[0]->num(); ++b) {
      for (int n = 0; n < num_; ++n) {
        int index = bottom[0]->count(1) * b + n * len * stride;
        activate_yolo_cpu(stride, index, num_class_, input_data, swap_data,
                          DEFAULT, false, true);
      }
    }
  }
#if defined(DEBUG) && defined(DRAW)
  for (int b = 0; b < bottom[0]->num(); ++b) {
    char buf[100];
    int idx = iter_ * bottom[0]->num() + b;
    sprintf(buf, "input/input_%05d.jpg", idx);
    // int idx = (iter*swap.num() % 200) + b;
    cv::Mat cv_img = cv::imread(buf);
    for (int t = 0; t < 300; ++t) {
      vector<Dtype> truth;
      Dtype c = label_data[b * 300 * 5 + t * 5 + 0];
      Dtype x = label_data[b * 300 * 5 + t * 5 + 1];
      Dtype y = label_data[b * 300 * 5 + t * 5 + 2];
      Dtype w = label_data[b * 300 * 5 + t * 5 + 3];
      Dtype h = label_data[b * 300 * 5 + t * 5 + 4];
      if (!x)
        break;

      cv::Point pt1;
      cv::Point pt2;
      pt1.x = (x - w / 2.) * cv_img.cols;
      pt1.y = (y - h / 2.) * cv_img.rows;
      pt2.x = (x + w / 2.) * cv_img.cols;
      pt2.y = (y + h / 2.) * cv_img.rows;

      cv::rectangle(cv_img, pt1, pt2, cv::Scalar(0, 255, 0), 1, 8, 0);
      char label[100];
      sprintf(label, "%s", CLASSES[static_cast<int>(c + 1)]);
      int baseline;
      cv::Size size =
          cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 0, &baseline);
      cv::Point pt3;
      pt3.x = pt1.x + size.width;
      pt3.y = pt1.y - size.height;
      cv::rectangle(cv_img, pt1, pt3, cv::Scalar(0, 255, 0), -1);

      cv::putText(cv_img, label, pt1, cv::FONT_HERSHEY_SIMPLEX, 0.5,
                  cv::Scalar(0, 0, 0));
      LOG(INFO) << "Truth box"
                << ", " << c << std::fixed << std::setprecision(2) << ", " << x
                << ", " << y << ", " << w << ", " << h;
    }
    sprintf(buf, "out/out_%05d.jpg", idx);
    cv::imwrite(buf, cv_img);
  }
#endif
  /*for (int i = 0; i < 81; i++) {
    char label[100];
    sprintf(label, "%d,%s\n",i, CLASSES[static_cast<int>(i )]);
    printf(label);
  }*/

  parallel_for(bottom[0]->num(), [&](int b) {
    // Assume that all detections are negative samples
    box pred;
    box truth;
    for (int s = 0; s < stride; ++s) {
      for (int n = 0; n < num_; n++) {
        int index = b * bottom[0]->count(1) + n * len * stride + s;
        int box_index = index;
        int obj_index = index + 4 * stride;
        int class_index = index + 5 * stride;
        // LOG(INFO)<<index;
        float best_iou = 0;
        int best_class = -1;
        int best_match_class = -1;
        box best_truth;

        float best_match_iou = 0;
        box best_match_truth;
        int x2 = s % side_w_;
        int y2 = s / side_w_;
        get_region_box(&pred, swap_data, biases_, mask_[n], box_index, x2, y2,
                       side_w_, side_h_, side_w_ * anchors_scale_,
                       side_h_ * anchors_scale_, stride);
        // find best respond gt box in image
        for (int t = 0; t < 300; ++t) {
          int class_id = label_data[b * 300 * 5 + t * 0];
          Dtype x = label_data[b * 300 * 5 + t * 5 + 1];
          Dtype y = label_data[b * 300 * 5 + t * 5 + 2];
          Dtype w = label_data[b * 300 * 5 + t * 5 + 3];
          Dtype h = label_data[b * 300 * 5 + t * 5 + 4];

          if (!x)
            break;
          Dtype objectness = swap_data[obj_index];
          if (isnan(objectness) || isinf(objectness))
            swap_data[obj_index] = 0;

          truth.x = x;
          truth.y = y;
          truth.w = w;
          truth.h = h;
          float iou = box_iou(pred, truth, iou_loss_);
          if (iou > best_match_iou &&
              compare_yolo_class(swap_data, num_class_, class_index, stride,
                                 objectness, class_id, 0.25)) {
            best_match_iou = iou;
            best_match_truth = truth;
            best_match_class = class_id;
          }
          if (iou > best_iou) {
            best_class = class_id;
            best_iou = iou;
            best_truth = truth;
          }
        }
        statistic.avg_anyobj += swap_data[obj_index];
        diff[obj_index] = (-1) * (0 - swap_data[obj_index]) * noobject_scale_;
        if (best_match_iou > thresh_) {
          // ================================================== alexeyAB version
          if (objectness_smooth_) {
            const float iou_multiplier = best_match_iou * best_match_iou;
            diff[obj_index] = (-1) * (iou_multiplier - swap_data[obj_index]) *
                              noobject_scale_;
            diff[class_index + best_match_class * stride] =
                (-1) * (iou_multiplier -
                        swap_data[class_index + best_match_class * stride]);
          } else {
            diff[obj_index] = 0;
          }
        }
        if (best_iou > 1) {
          LOG(INFO) << "best_iou > 1"; // plz tell me ..
          const float iou_multiplier = best_iou * best_iou;
          if (objectness_smooth_) {
            diff[obj_index] =
                (-1) * object_scale_ * (iou_multiplier - swap_data[obj_index]);
          } else {
            diff[obj_index] = (-1) * object_scale_ * (1 - swap_data[obj_index]);
          }

          delta_region_class_v3(swap_data, diff, class_index, best_class,
                                num_class_, class_scale_, nullptr, stride,
                                use_focal_loss_, label_smooth_eps_);
          if (objectness_smooth_) {
            diff[class_index + best_class * stride] =
                (-1) *
                (iou_multiplier - swap_data[class_index + best_class * stride]);
          }
          delta_region_box(best_truth, swap_data, biases_, mask_[n], box_index,
                           x2, y2, side_w_, side_h_, side_w_ * anchors_scale_,
                           side_h_ * anchors_scale_, diff,
                           coord_scale_ * (2 - best_truth.w * best_truth.h),
                           stride, iou_loss_, iou_normalizer_, max_delta_,
                           accumulate_);
        }
      }
    }
    // vector<Dtype> used;
    // used.clear();
    box truth_shift;
    float iou;
    int mask_n;
    for (int t = 0; t < 300; ++t) {
      int class_label = label_data[b * 300 * 5 + t * 5 + 0];
      truth.x = label_data[b * 300 * 5 + t * 5 + 1];
      truth.y = label_data[b * 300 * 5 + t * 5 + 2];
      truth.w = label_data[b * 300 * 5 + t * 5 + 3];
      truth.h = label_data[b * 300 * 5 + t * 5 + 4];

      if (!truth.w)
        break;

      float best_iou = 0;
      int best_index = 0;
      int best_n = -1;
      int i = truth.x * side_w_;
      int j = truth.y * side_h_;
      int pos = j * side_w_ + i;
      truth_shift.x = 0;
      truth_shift.y = 0;
      truth_shift.w = truth.w;
      truth_shift.h = truth.h;

      // LOG(INFO) << j << "," << i << "," << anchors_scale_;
      // find the best anchor matches label,
      // no matter the best iou is low
      for (int n = 0; n < biases_size_; ++n) {
        pred.x = 0;
        pred.y = 0;
        pred.w = biases_[2 * n] / (float)(side_w_ * anchors_scale_);
        pred.h = biases_[2 * n + 1] / (float)(side_h_ * anchors_scale_);
        iou = box_iou(pred, truth_shift, iou_loss_);
        if (iou > best_iou) {
          best_n = n;
          best_iou = iou;
        }
      }
      mask_n = int_index(mask_, best_n, num_);
      if (mask_n >= 0) {
        bool overlap = false;
        best_n = mask_n;
        // LOG(INFO) << best_n;
        best_index = b * bottom[0]->count(1) + best_n * len * stride + pos;

        ious all_ious = delta_region_box(
            truth, swap_data, biases_, mask_[best_n], best_index, i, j, side_w_,
            side_h_, side_w_ * anchors_scale_, side_h_ * anchors_scale_, diff,
            coord_scale_ * (2 - truth.w * truth.h), stride, iou_loss_,
            iou_normalizer_, max_delta_, accumulate_);

        diff[best_index + 4 * stride] =
            (-1.0) * (1 - swap_data[best_index + 4 * stride]) * object_scale_;

        delta_region_class_v3(swap_data, diff, best_index + 5 * stride,
                              class_label, num_class_, class_scale_, &statistic,
                              stride, use_focal_loss_,
                              label_smooth_eps_); // softmax_tree_

        {
          ++statistic.count;
          ++statistic.class_count;
          if (all_ious.iou > 0.5)
            statistic.recall += 1;
          if (all_ious.iou > 0.75)
            statistic.recall75 += 1;
          statistic.tot_iou += all_ious.iou;
          statistic.tot_iou_loss += 1 - all_ious.iou;
          statistic.tot_giou += all_ious.giou;
          statistic.tot_giou_loss += 1 - all_ious.giou;
          statistic.tot_ciou += all_ious.ciou;
          statistic.tot_ciou_loss += 1 - all_ious.ciou;
          statistic.tot_diou += all_ious.diou;
          statistic.tot_diou_loss += 1 - all_ious.diou;

          statistic.avg_obj += swap_data[best_index + 4 * stride];
        }
      }

      // if the rest anchor, which iou is larger than threshold,
      // it is also the positive sample
      // ====================================================== alexeyAB version
      if (use_extra_matched_anchor_) {
        for (int n = 0; n < biases_size_; ++n) {
          mask_n = int_index(mask_, n, num_);
          if (mask_n >= 0 && n != best_n && iou_thresh_ < 1.0f) {
            pred.x = 0;
            pred.y = 0;
            pred.w = biases_[2 * n] / (float)(side_w_ * anchors_scale_);
            pred.h = biases_[2 * n + 1] / (float)(side_h_ * anchors_scale_);
            iou = box_iou(pred, truth_shift, iou_loss_);

            if (iou > iou_thresh_) {
              bool overlap = false;
              // LOG(INFO) << best_n;
              best_index =
                  b * bottom[0]->count(1) + mask_n * len * stride + pos;

              ious all_ious = delta_region_box(
                  truth, swap_data, biases_, mask_[mask_n], best_index, i, j,
                  side_w_, side_h_, side_w_ * anchors_scale_,
                  side_h_ * anchors_scale_, diff,
                  coord_scale_ * (2 - truth.w * truth.h), stride, iou_loss_,
                  iou_normalizer_, max_delta_, accumulate_);
              diff[best_index + 4 * stride] =
                  (-1.0) * (1 - swap_data[best_index + 4 * stride]) *
                  object_scale_;

              // diff[best_index + 4 * stride] = (-1.0) * (1 -
              // swap_data[best_index + 4 * stride]) ;

              delta_region_class_v3(swap_data, diff, best_index + 5 * stride,
                                    class_label, num_class_, class_scale_,
                                    &statistic, stride, use_focal_loss_,
                                    label_smooth_eps_); // softmax_tree_

              {
                ++statistic.count;
                ++statistic.class_count;
                if (all_ious.iou > 0.5)
                  statistic.recall += 1;
                if (all_ious.iou > 0.75)
                  statistic.recall75 += 1;
                statistic.tot_iou += all_ious.iou;
                statistic.tot_iou_loss += 1 - all_ious.iou;
                statistic.tot_giou += all_ious.giou;
                statistic.tot_giou_loss += 1 - all_ious.giou;
                statistic.tot_ciou += all_ious.ciou;
                statistic.tot_ciou_loss += 1 - all_ious.ciou;
                statistic.tot_diou += all_ious.diou;
                statistic.tot_diou_loss += 1 - all_ious.diou;

                statistic.avg_obj += swap_data[best_index + 4 * stride];
              }
            }
          }
        }
      }
    }

    // averages the deltas obtained by the function: delta_yolo_box()_accumulate
    for (int s = 0; s < stride; ++s) {
      for (int n = 0; n < num_; n++) {
        int index = b * bottom[0]->count(1) + n * len * stride + s;
        averages_yolo_deltas(index + 5 * stride, index + 0 * stride, stride,
                             num_class_, diff);
      }
    }
  });
  if (statistic.count == 0)
    statistic.count = 1;
  if (statistic.class_count == 0)
    statistic.class_count = 1;
  // LOG(INFO) << " ===================================================== " ;
  Dtype *no_iou_loss_delta = swap_.mutable_cpu_diff();
  caffe_copy(swap_.count(), diff_.cpu_data(), no_iou_loss_delta);
  for (int b = 0; b < swap_.num(); ++b) {
    for (int s = 0; s < stride; ++s) {
      for (int n = 0; n < num_; ++n) {
        int index = b * swap_.count(1) + n * len * stride + s;
        no_iou_loss_delta[index + 0 * stride] = 0;
        no_iou_loss_delta[index + 1 * stride] = 0;
        no_iou_loss_delta[index + 2 * stride] = 0;
        no_iou_loss_delta[index + 3 * stride] = 0;
      }
    }
  }
  caffe_powx<Dtype>(swap_.count(), no_iou_loss_delta, 2, no_iou_loss_delta);
  Dtype clz_loss = caffe_blas_asum(swap_.count(), no_iou_loss_delta);
  // use this data again to prevent alloc mem
  caffe_powx<Dtype>(swap_.count(), diff, 2, no_iou_loss_delta);
  Dtype loss = caffe_blas_asum(swap_.count(), no_iou_loss_delta);
  Dtype iou_loss = loss - clz_loss;
  if (iou_loss_ == MSE) {
    top[0]->mutable_cpu_data()[0] = loss / bottom[0]->num();
  } else {
    Dtype avg_iou_loss = 0;
    if (iou_loss_ == GIOU) {
      avg_iou_loss =
          statistic.count > 0
              ? iou_normalizer_ * (statistic.tot_giou_loss / statistic.count)
              : 0;
    } else {
      avg_iou_loss =
          statistic.count > 0
              ? iou_normalizer_ * (statistic.tot_iou_loss / statistic.count)
              : 0;
    }
    // LOG(INFO) << avg_iou_loss;
    top[0]->mutable_cpu_data()[0] =
        (avg_iou_loss + clz_loss) / bottom[0]->num();
  }
  loss /= bottom[0]->num();
  clz_loss /= bottom[0]->num();
  iou_loss /= bottom[0]->num();
  // LOG(INFO) << "avg_noobj: " << avg_anyobj / (side_ * side_ * num_ *
  // bottom[0]->num());
  iter_++;
  // LOG(INFO) << "iter: " << iter <<" loss: " << loss;
  if (!(iter_ % display_) && time_count_ > 0) {
    LOG(INFO) << std::fixed << std::setprecision(4)
              << "[scale:" << anchors_scale_ << "]:"
              << " anyobj: " << score_.avg_anyobj / time_count_
              << " obj: " << score_.avg_obj / time_count_
              << " iou: " << score_.avg_iou / time_count_
              << " cat: " << score_.avg_cat / time_count_
              << " count: " << class_count_ / time_count_;
    LOG(INFO) << std::fixed << std::setprecision(4)
              << "recall: " << score_.recall / time_count_
              << " recall75: " << score_.recall75 / time_count_
              << " iou_loss: " << score_.iou_loss / time_count_
              << " clz_loss: " << score_.clz_loss / time_count_
              << " loss: " << score_.loss / time_count_;
    // LOG(INFO) << "avg_noobj: "<<
    // avg_anyobj/(side_*side_*num_*bottom[0]->num()) << " avg_obj: " <<
    // avg_obj/count <<" avg_iou: " << avg_iou/count << " avg_cat: " <<
    // avg_cat/class_count << " recall: " << recall/count << " recall75: " <<
    // recall75 / count;
    score_ = {0};
    class_count_ = 0;
    time_count_ = 0;
  } else {
    score_.avg_anyobj +=
        statistic.avg_anyobj / (side_w_ * side_h_ * num_ * bottom[0]->num());
    if (statistic.count > 0) {
      score_.avg_obj += statistic.avg_obj / statistic.count;
      score_.avg_iou += statistic.tot_iou / statistic.count;
      score_.avg_cat += statistic.avg_cat / statistic.count;
      score_.recall += statistic.recall / statistic.count;
      score_.recall75 += statistic.recall75 / statistic.count;
      score_.loss += loss;
      score_.clz_loss += clz_loss;
      score_.iou_loss += iou_loss;
      class_count_ += statistic.class_count;
      time_count_++;
    }
  }
}

template <typename Dtype>
void Yolov3Layer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *> &top,
                                      const vector<bool> &propagate_down,
                                      const vector<Blob<Dtype> *> &bottom) {
  // LOG(INFO) <<" propagate_down: "<< propagate_down[1] << " " <<
  // propagate_down[0];
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    if (use_logic_gradient_) {
      const Dtype *top_data = swap_.cpu_data();
      Dtype *diff = diff_.mutable_cpu_data();
      side_w_ = bottom[0]->width();
      side_h_ = bottom[0]->height();
      int len = 4 + num_class_ + 1;
      int stride = side_w_ * side_h_;
      // LOG(INFO)<<swap.count(1);
      parallel_for(bottom[0]->num(), [&](int b) {
        for (int s = 0; s < stride; s++) {
          for (int n = 0; n < num_; n++) {
            int index = n * len * stride + s + b * bottom[0]->count(1);
            // LOG(INFO)<<index;
            //            vector<Dtype> pred;
            //            float best_iou = 0;
            //            int best_class = -1;
            //            vector<Dtype> best_truth;
            for (int c = 0; c < len; ++c) {
              int index2 = c * stride + index;
              // LOG(INFO)<<index2;
              if (c == 2 || c == 3) {
                diff[index2] = diff[index2];
              } else {
                diff[index2] =
                    diff[index2] * caffe_fn_sigmoid_grad_fast(top_data[index2]);
              }
            }
          }
        }
      });
    } else {
      // non-logic_gradient formula
      // https://blog.csdn.net/yanzi6969/article/details/80505421
      // https://xmfbit.github.io/2018/03/21/cs229-supervised-learning/
      // https://zlatankr.github.io/posts/2017/03/06/mle-gradient-descent
    }
    const Dtype sign(1.);
    const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[0]->num();
    // const Dtype alpha(1.0);
    // LOG(INFO) << "alpha:" << alpha;

    caffe_blas_axpby(bottom[0]->count(), alpha, diff_.cpu_data(), Dtype(0),
                     bottom[0]->mutable_cpu_diff());
  }
}

#ifdef CPU_ONLY
STUB_GPU(Yolov3Layer);
#endif

INSTANTIATE_CLASS(Yolov3Layer);
REGISTER_LAYER_CLASS(Yolov3);

} // namespace caffe
