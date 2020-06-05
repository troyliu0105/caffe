//
// Created by Troy Liu on 2020/5/9.
//

#ifndef CAFFE_INCLUDE_CAFFE_UTIL_YOLO_UTILS_HPP
#define CAFFE_INCLUDE_CAFFE_UTIL_YOLO_UTILS_HPP

#include "bbox_util.hpp"

namespace caffe {

/**
 * @brief GPU 的激活函数
 * @tparam Dtype
 * @param b             当前的 batch index
 * @param n             当前的 anchor index
 * @param len           每个 anchor 对应的深度
 * @param stride        lh * lw
 * @param index         初始状态下的 index，对应第一个 batch 的第一个 pixel
 * 的第一个 channel
 * @param num_classes   有多少个 class，不包含 conf_conf
 * @param input_data    输入 data，未使用化偏移
 * @param output_data   输出 data，未使用化偏移
 * @param scores
 * @param arc
 * @param gaussian_box
 */
template <class Dtype>
void activate_yolo_cpu(int stride, int index, int num_classes,
                       const Dtype *input_data, Dtype *output_data, ARC arc,
                       bool gaussian_box, bool is_train = true,
                       float xy_scale = 1.0);

/**
 * @brief get bbox from feature maps
 * @tparam Dtype    feature map data type
 * @param b         'box' instance
 * @param x         feature map pointer
 * @param biases    all anchors in vector, N_anchor x 2
 * @param n         anchor index
 * @param index     (x, y) dot index in feat map
 * @param i         cx in grid
 * @param j         cy in grid
 * @param lw        feat map width
 * @param lh        feat map height
 * @param w         original input image width
 * @param h         original input image height
 * @param stride    stride to jump, = lw * lh
 */
template <typename Dtype>
void get_region_box(box *b, const Dtype *x, vector<Dtype> biases, int n,
                    int index, int i, int j, int lw, int lh, int w, int h,
                    int stride);
template <typename Dtype>
void get_region_box(vector<Dtype> &b, const Dtype *x, vector<Dtype> biases,
                    int n, int index, int i, int j, int lw, int lh, int w,
                    int h, int stride);
template <typename Dtype>
void get_gaussian_yolo_box(box *b, const Dtype *x, vector<Dtype> biases, int n,
                           int index, int i, int j, int lw, int lh, int w,
                           int h, int stride);
template <typename Dtype>
void get_gaussian_yolo_box(vector<Dtype> &b, const Dtype *x,
                           vector<Dtype> biases, int n, int index, int i, int j,
                           int lw, int lh, int w, int h, int stride);

#ifndef CPU_ONLY
template <class Dtype>
void activate_yolo_gpu(int stride, int index, int num_classes,
                       const Dtype *input_data, Dtype *output_data, ARC arc,
                       bool gaussian_box, bool is_train = true,
                       float xy_scale = 1.0);
#endif
} // namespace caffe

#endif // CAFFE_INCLUDE_CAFFE_UTIL_YOLO_UTILS_HPP
