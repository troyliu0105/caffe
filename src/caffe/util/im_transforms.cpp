#ifdef USE_OPENCV
#include <opencv2/highgui/highgui.hpp>

#if CV_VERSION_MAJOR >= 3
#include <opencv2/imgcodecs/imgcodecs.hpp>
#define CV_GRAY2BGR cv::COLOR_GRAY2BGR
#define CV_BGR2GRAY cv::COLOR_BGR2GRAY
#define CV_BGR2YCrCb cv::COLOR_BGR2YCrCb
#define CV_YCrCb2BGR cv::COLOR_YCrCb2BGR
#define CV_IMWRITE_JPEG_QUALITY cv::IMWRITE_JPEG_QUALITY
#define CV_LOAD_IMAGE_COLOR cv::IMREAD_COLOR
#define CV_THRESH_BINARY_INV cv::THRESH_BINARY_INV
#define CV_THRESH_OTSU cv::THRESH_OTSU
#define CV_BGR2HSV cv::COLOR_BGR2HSV
#define CV_HSV2BGR cv::COLOR_HSV2BGR
#define CV_BGR2Lab cv::COLOR_BGR2Lab
#endif
#endif // USE_OPENCV

#include <algorithm>
#include <numeric>
#include <vector>

#include "caffe/util/im_transforms.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

namespace internal {
template <typename T>
bool is_border(const cv::Mat &edge, T color) {
  cv::Mat im = edge.clone().reshape(0, 1);
  bool res = true;
  for (int i = 0; i < im.cols; ++i) {
    res &= (color == im.at<T>(0, i));
  }
  return res;
}

template bool is_border(const cv::Mat &edge, uchar color);

template <typename T>
cv::Rect CropMask(const cv::Mat &src, T point, int padding) {
  cv::Rect win(0, 0, src.cols, src.rows);

  vector<cv::Rect> edges;
  edges.emplace_back(0, 0, src.cols, 1);
  edges.emplace_back(src.cols - 2, 0, 1, src.rows);
  edges.emplace_back(0, src.rows - 2, src.cols, 1);
  edges.emplace_back(0, 0, 1, src.rows);

  cv::Mat edge;
  int nborder = 0;
  T color = src.at<T>(0, 0);
  for (auto &i : edges) {
    edge = src(i);
    nborder += is_border(edge, color);
  }

  if (nborder < 4) {
    return win;
  }

  bool next;
  do {
    edge = src(cv::Rect(win.x, win.height - 2, win.width, 1));
    next = is_border(edge, color);
    if (next) {
      win.height--;
    }
  } while (next && (win.height > 0));

  do {
    edge = src(cv::Rect(win.width - 2, win.y, 1, win.height));
    next = is_border(edge, color);
    if (next) {
      win.width--;
    }
  } while (next && (win.width > 0));

  do {
    edge = src(cv::Rect(win.x, win.y, win.width, 1));
    next = is_border(edge, color);
    if (next) {
      win.y++;
      win.height--;
    }
  } while (next && (win.y <= src.rows));

  do {
    edge = src(cv::Rect(win.x, win.y, 1, win.height));
    next = is_border(edge, color);
    if (next) {
      win.x++;
      win.width--;
    }
  } while (next && (win.x <= src.cols));

  // add padding
  if (win.x > padding) {
    win.x -= padding;
  }
  if (win.y > padding) {
    win.y -= padding;
  }
  if ((win.width + win.x + padding) < src.cols) {
    win.width += padding;
  }
  if ((win.height + win.y + padding) < src.rows) {
    win.height += padding;
  }

  return win;
}

template cv::Rect CropMask(const cv::Mat &src, uchar point, int padding);

cv::Mat colorReduce(const cv::Mat &image, int div) {
  cv::Mat out_img;
  cv::Mat lookUpTable(1, 256, CV_8U);
  uchar *p = lookUpTable.data;
  const int div_2 = div / 2;
  for (int i = 0; i < 256; ++i) {
    p[i] = i / div * div + div_2;
  }
  cv::LUT(image, lookUpTable, out_img);
  return out_img;
}

void fillEdgeImage(const cv::Mat &edgesIn, cv::Mat *filledEdgesOut) {
  cv::Mat edgesNeg = edgesIn.clone();
  cv::Scalar val(255, 255, 255);
  cv::floodFill(edgesNeg, cv::Point(0, 0), val);
  cv::floodFill(edgesNeg, cv::Point(edgesIn.cols - 1, edgesIn.rows - 1), val);
  cv::floodFill(edgesNeg, cv::Point(0, edgesIn.rows - 1), val);
  cv::floodFill(edgesNeg, cv::Point(edgesIn.cols - 1, 0), val);
  cv::bitwise_not(edgesNeg, edgesNeg);
  *filledEdgesOut = (edgesNeg | edgesIn);
}

void CenterObjectAndFillBg(const cv::Mat &in_img, const bool fill_bg,
                           cv::Mat *out_img) {
  cv::Mat mask, crop_mask;
  if (in_img.channels() > 1) {
    cv::Mat in_img_gray;
    cv::cvtColor(in_img, in_img_gray, CV_BGR2GRAY);
    cv::threshold(in_img_gray, mask, 0, 255,
                  CV_THRESH_BINARY_INV | CV_THRESH_OTSU);
  } else {
    cv::threshold(in_img, mask, 0, 255, CV_THRESH_BINARY_INV | CV_THRESH_OTSU);
  }
  cv::Rect crop_rect = CropMask(mask, mask.at<uchar>(0, 0), 2);

  if (fill_bg) {
    cv::Mat temp_img = in_img(crop_rect);
    fillEdgeImage(mask, &mask);
    crop_mask = mask(crop_rect).clone();
    *out_img = cv::Mat::zeros(crop_rect.size(), in_img.type());
    temp_img.copyTo(*out_img, crop_mask);
  } else {
    *out_img = in_img(crop_rect).clone();
  }
}

cv::Mat AspectKeepingResizeAndPad(const cv::Mat &in_img, const int new_width,
                                  const int new_height, const int pad_type,
                                  const cv::Scalar &pad_val,
                                  const int interp_mode) {
  cv::Mat img_resized;
  float orig_aspect = static_cast<float>(in_img.cols) / in_img.rows;
  float new_aspect = static_cast<float>(new_width) / new_height;

  if (orig_aspect > new_aspect) {
    int height = floor(static_cast<float>(new_width) / orig_aspect);
    cv::resize(in_img, img_resized, cv::Size(new_width, height), 0, 0,
               interp_mode);
    cv::Size resSize = img_resized.size();
    int padding = floor((new_height - resSize.height) / 2.0);
    cv::copyMakeBorder(img_resized, img_resized, padding,
                       new_height - resSize.height - padding, 0, 0, pad_type,
                       pad_val);
  } else {
    int width = floor(orig_aspect * new_height);
    cv::resize(in_img, img_resized, cv::Size(width, new_height), 0, 0,
               interp_mode);
    cv::Size resSize = img_resized.size();
    int padding = floor((new_width - resSize.width) / 2.0);
    cv::copyMakeBorder(img_resized, img_resized, 0, 0, padding,
                       new_width - resSize.width - padding, pad_type, pad_val);
  }
  return img_resized;
}

cv::Mat AspectKeepingResizeBySmall(const cv::Mat &in_img, const int new_width,
                                   const int new_height,
                                   const int interp_mode) {
  cv::Mat img_resized;
  float orig_aspect = static_cast<float>(in_img.cols) / in_img.rows;
  float new_aspect = static_cast<float>(new_width) / new_height;

  if (orig_aspect < new_aspect) {
    int height = floor(static_cast<float>(new_width) / orig_aspect);
    cv::resize(in_img, img_resized, cv::Size(new_width, height), 0, 0,
               interp_mode);
  } else {
    int width = floor(orig_aspect * new_height);
    cv::resize(in_img, img_resized, cv::Size(width, new_height), 0, 0,
               interp_mode);
  }
  return img_resized;
}

void constantNoise(const int n, const vector<uchar> &val, cv::Mat *image) {
  const int cols = image->cols;
  const int rows = image->rows;

  if (image->channels() == 1) {
    for (int k = 0; k < n; ++k) {
      const int i = caffe_rng_rand() % cols;
      const int j = caffe_rng_rand() % rows;
      auto *ptr = image->ptr<uchar>(j);
      ptr[i] = val[0];
    }
  } else if (image->channels() == 3) { // color image
    for (int k = 0; k < n; ++k) {
      const int i = caffe_rng_rand() % cols;
      const int j = caffe_rng_rand() % rows;
      auto *ptr = image->ptr<cv::Vec3b>(j);
      (ptr[i])[0] = val[0];
      (ptr[i])[1] = val[1];
      (ptr[i])[2] = val[2];
    }
  }
}

void RandomBrightness(const cv::Mat &in_img, cv::Mat *out_img,
                      const float brightness_prob,
                      const float brightness_delta) {
  float prob;
  caffe_rng_uniform(1, 0.f, 1.f, &prob);
  if (prob < brightness_prob) {
    CHECK_GE(brightness_delta, 0) << "brightness_delta must be non-negative.";
    float delta;
    caffe_rng_uniform(1, -brightness_delta, brightness_delta, &delta);
    AdjustBrightness(in_img, delta, out_img);
  } else {
    *out_img = in_img;
  }
}

void AdjustBrightness(const cv::Mat &in_img, const float delta,
                      cv::Mat *out_img) {
  if (fabs(delta) > 0) {
    in_img.convertTo(*out_img, -1, 1, delta);
  } else {
    *out_img = in_img;
  }
}

void RandomContrast(const cv::Mat &in_img, cv::Mat *out_img,
                    const float contrast_prob, const float lower,
                    const float upper) {
  float prob;
  caffe_rng_uniform(1, 0.f, 1.f, &prob);
  if (prob < contrast_prob) {
    CHECK_GE(upper, lower) << "contrast upper must be >= lower.";
    CHECK_GE(lower, 0) << "contrast lower must be non-negative.";
    float delta;
    caffe_rng_uniform(1, lower, upper, &delta);
    AdjustContrast(in_img, delta, out_img);
  } else {
    *out_img = in_img;
  }
}

void AdjustContrast(const cv::Mat &in_img, const float delta,
                    cv::Mat *out_img) {
  if (fabs(delta - 1.f) > 1e-3) {
    in_img.convertTo(*out_img, -1, delta, 0);
  } else {
    *out_img = in_img;
  }
}

void RandomSaturation(const cv::Mat &in_img, cv::Mat *out_img,
                      const float saturation_prob, const float lower,
                      const float upper) {
  float prob;
  caffe_rng_uniform(1, 0.f, 1.f, &prob);
  if (prob < saturation_prob) {
    CHECK_GE(upper, lower) << "saturation upper must be >= lower.";
    CHECK_GE(lower, 0) << "saturation lower must be non-negative.";
    float delta;
    caffe_rng_uniform(1, lower, upper, &delta);
    AdjustSaturation(in_img, delta, out_img);
  } else {
    *out_img = in_img;
  }
}

void AdjustSaturation(const cv::Mat &in_img, const float delta,
                      cv::Mat *out_img) {
  if (fabs(delta - 1.f) != 1e-3) {
    // Convert to HSV colorspae.
    cv::cvtColor(in_img, *out_img, CV_BGR2HSV);

    // Split the image to 3 channels.
    vector<cv::Mat> channels;
    cv::split(*out_img, channels);

    // Adjust the saturation.
    channels[1].convertTo(channels[1], -1, delta, 0);
    cv::merge(channels, *out_img);

    // Back to BGR colorspace.
    cvtColor(*out_img, *out_img, CV_HSV2BGR);
  } else {
    *out_img = in_img;
  }
}

void RandomHue(const cv::Mat &in_img, cv::Mat *out_img, const float hue_prob,
               const float hue_delta) {
  float prob;
  caffe_rng_uniform(1, 0.f, 1.f, &prob);
  if (prob < hue_prob) {
    CHECK_GE(hue_delta, 0) << "hue_delta must be non-negative.";
    float delta;
    caffe_rng_uniform(1, -hue_delta, hue_delta, &delta);
    AdjustHue(in_img, delta, out_img);
  } else {
    *out_img = in_img;
  }
}

void AdjustHue(const cv::Mat &in_img, const float delta, cv::Mat *out_img) {
  if (fabs(delta) > 0) {
    // Convert to HSV colorspae.
    cv::cvtColor(in_img, *out_img, CV_BGR2HSV);

    // Split the image to 3 channels.
    vector<cv::Mat> channels;
    cv::split(*out_img, channels);

    // Adjust the hue.
    channels[0].convertTo(channels[0], -1, 1, delta);
    cv::merge(channels, *out_img);

    // Back to BGR colorspace.
    cvtColor(*out_img, *out_img, CV_HSV2BGR);
  } else {
    *out_img = in_img;
  }
}

void RandomOrderChannels(const cv::Mat &in_img, cv::Mat *out_img,
                         const float random_order_prob) {
  float prob;
  caffe_rng_uniform(1, 0.f, 1.f, &prob);
  if (prob < random_order_prob) {
    // Split the image to 3 channels.
    vector<cv::Mat> channels;
    cv::split(*out_img, channels);
    CHECK_EQ(channels.size(), 3);

    // Shuffle the channels.
    caffe::shuffle(channels.begin(), channels.end());
    cv::merge(channels, *out_img);
  } else {
    *out_img = in_img;
  }
}
void filterBBoxes(std::vector<NormalizedBBox> &bboxes,
                  std::vector<int> &instances) {
  std::vector<NormalizedBBox> filteredBBoxes;
  std::vector<int> filteredInstances;
  for (int i = 0; i < bboxes.size(); ++i) {
    NormalizedBBox bb = bboxes[i];
    if (bb.xmax() >= 0.0 && bb.xmin() <= 1.0 && bb.ymax() >= 0.0 &&
        bb.ymin() <= 1.0) {
      NormalizedBBox nbb;
      nbb.set_label(bb.label());
      nbb.set_difficult(bb.difficult());
      nbb.set_score(bb.score());
      nbb.set_size(bb.size());
      nbb.set_xmin(std::max(0.f, bb.xmin()));
      nbb.set_xmax(std::min(1.f, bb.xmax()));
      nbb.set_ymin(std::max(0.f, bb.ymin()));
      nbb.set_ymax(std::min(1.f, bb.ymax()));
      float surfbb = (bb.xmax() - bb.xmin()) * (bb.ymax() - bb.ymin());
      float surfnbb = (nbb.xmax() - nbb.xmin()) * (nbb.ymax() - nbb.ymin());
      if (surfnbb > 0.75 * surfbb) {
        filteredBBoxes.push_back(nbb);
        filteredInstances.push_back(instances[i]);
      }
    }
  }
  bboxes.clear();
  bboxes.insert(bboxes.end(), filteredBBoxes.begin(), filteredBBoxes.end());
  instances.clear();
  instances.insert(instances.end(), filteredInstances.begin(),
                   filteredInstances.end());
}

void warpBBoxes(std::vector<NormalizedBBox> &bboxes, cv::Mat warpMat, int rows,
                int cols) {
  std::vector<NormalizedBBox> newbboxes;
  for (NormalizedBBox &bb : bboxes) {
    std::vector<cv::Point2f> origBBox;
    std::vector<cv::Point2f> warpedBBox;
    cv::Point2f p1;
    p1.x = bb.xmin() * cols;
    p1.y = bb.ymin() * rows;
    origBBox.push_back(p1);
    cv::Point2f p2;
    p2.x = bb.xmax() * cols;
    p2.y = bb.ymax() * rows;
    origBBox.push_back(p2);
    cv::Point2f p3;
    p3.x = bb.xmin() * cols;
    p3.y = bb.ymax() * rows;
    origBBox.push_back(p3);
    cv::Point2f p4;
    p4.x = bb.xmax() * cols;
    p4.y = bb.ymin() * rows;
    origBBox.push_back(p4);

    perspectiveTransform(origBBox, warpedBBox, warpMat);

    float xmin = warpedBBox[0].x;
    float ymin = warpedBBox[0].y;
    float xmax = warpedBBox[0].x;
    float ymax = warpedBBox[0].y;
    for (int i = 1; i < 4; ++i) {
      if (warpedBBox[i].x < xmin)
        xmin = warpedBBox[i].x;
      if (warpedBBox[i].x > xmax)
        xmax = warpedBBox[i].x;
      if (warpedBBox[i].y < ymin)
        ymin = warpedBBox[i].y;
      if (warpedBBox[i].y > ymax)
        ymax = warpedBBox[i].y;
    }
    bb.set_xmin(xmin / cols);
    bb.set_ymin(ymin / rows);
    bb.set_xmax(xmax / cols);
    bb.set_ymax(ymax / rows);
  }
}

void mirror_x(NormalizedBBox &bb) {
  float xmin = 1.0 - bb.xmax();
  bb.set_xmax(1.0 - bb.xmin());
  bb.set_xmin(xmin);
}

void mirror_y(NormalizedBBox &bb) {
  float ymin = 1.0 - bb.ymax();
  bb.set_ymax(1.0 - bb.ymin());
  bb.set_ymin(ymin);
}

void shift_x(NormalizedBBox &bb, int n) {
  if (n == 0)
    return;
  bb.set_xmin(((float)n) + bb.xmin());
  bb.set_xmax(((float)n) + bb.xmax());
}

void shift_y(NormalizedBBox &bb, int n) {
  if (n == 0)
    return;
  bb.set_ymin(((float)n) + bb.ymin());
  bb.set_ymax(((float)n) + bb.ymax());
}

void getEnlargedBBoxes(int rows, int cols, const GeometryParameter &param,
                       std::vector<NormalizedBBox> &bboxes,
                       std::vector<int> &instances) {
  int imin, imax, jmin, jmax;
  switch (param.pad_mode()) {
  case GeometryParameter_Pad_mode_CONSTANT:
  case GeometryParameter_Pad_mode_REPEAT_NEAREST:
    imin = jmin = 1;
    imax = jmax = 2;
    break;
  case GeometryParameter_Pad_mode_MIRRORED:
    imin = jmin = 0;
    imax = jmax = 3;
    break;
  default:
    LOG(ERROR) << "Unknown pad mode.";
    LOG(FATAL) << "fatal error";
    return;
  }
  std::vector<int> newInstances;
  std::vector<NormalizedBBox> newBBoxes;
  int ii = 0;
  for (NormalizedBBox b : bboxes) {
    for (int i = imin; i < imax; ++i)   // iterate over x
      for (int j = jmin; j < jmax; ++j) // iterate over y
      {
        NormalizedBBox bb;
        bb.set_label(b.label());
        bb.set_difficult(b.difficult());
        bb.set_score(b.score());
        bb.set_size(b.size());
        bb.set_xmin(b.xmin());
        bb.set_xmax(b.xmax());
        bb.set_ymin(b.ymin());
        bb.set_ymax(b.ymax());
        if (i == 0 || i == 2)
          mirror_x(bb);
        if (j == 0 || j == 2)
          mirror_y(bb);
        shift_x(bb, i);
        shift_y(bb, j);
        newBBoxes.push_back(bb);
        newInstances.push_back(ii);
      }
    ii++;
  }
  bboxes.clear();
  bboxes.insert(bboxes.end(), newBBoxes.begin(), newBBoxes.end());
  instances.clear();
  instances.insert(instances.end(), newInstances.begin(), newInstances.end());
}

void getEnlargedImage(const cv::Mat &in_img, const GeometryParameter &param,
                      cv::Mat &in_img_enlarged) {
  int pad_mode = cv::BORDER_REFLECT101;
  switch (param.pad_mode()) {
  case GeometryParameter_Pad_mode_CONSTANT:
    pad_mode = cv::BORDER_CONSTANT;
    break;
  case GeometryParameter_Pad_mode_MIRRORED:
    pad_mode = cv::BORDER_REFLECT101;
    break;
  case GeometryParameter_Pad_mode_REPEAT_NEAREST:
    pad_mode = cv::BORDER_REPLICATE;
    break;
  default:
    LOG(ERROR) << "Unknown pad mode.";
    LOG(FATAL) << "fatal error";
  }

  copyMakeBorder(in_img, in_img_enlarged, in_img.rows, in_img.rows, in_img.cols,
                 in_img.cols, pad_mode);
}

void getQuads(int rows, int cols, const GeometryParameter &param,
              cv::Point2f (&inputQuad)[4], cv::Point2f (&outputQuad)[4]) {
  // The 4 points that select quadilateral on the input , from top-left in
  // clockwise order These four pts are the sides of the rect box used as input
  float x0, x1, y0, y1;
  x0 = cols;
  x1 = 2 * cols - 1;
  y0 = rows;
  y1 = 2 * rows - 1;
  if (param.zoom_out() || param.zoom_in() || param.all_effects()) {
    bool zoom_in = param.zoom_in() || param.all_effects();
    bool zoom_out = param.zoom_out() || param.all_effects();
    if (zoom_out && zoom_in) {
      vector<float> binary_probs = {0.5, 0.5};
      if (roll_weighted_die(binary_probs) == 0)
        zoom_in = false;
      else
        zoom_out = false;
    }

    float x0min, x0max, y0min, y0max;
    if (zoom_in) {
      x0max = cols + cols * param.zoom_factor();
      y0max = rows + rows * param.zoom_factor();
    } else {
      x0max = x0;
      y0max = y0;
    }
    if (zoom_out) {
      x0min = cols - cols * param.zoom_factor();
      y0min = rows - rows * param.zoom_factor();
    } else {
      x0min = x0;
      y0min = y0;
    }
    caffe_rng_uniform(1, x0min, x0max, &x0);
    x1 = 3 * cols - x0;
    caffe_rng_uniform(1, y0min, y0max, &y0);
    y1 = 3 * rows - y0;
  }

  inputQuad[0] = cv::Point2f(x0, y0);
  inputQuad[1] = cv::Point2f(x1, y0);
  inputQuad[2] = cv::Point2f(x1, y1);
  inputQuad[3] = cv::Point2f(x0, y1);

  // The 4 points where the mapping is to be done , from top-left in clockwise
  // order
  outputQuad[0] = cv::Point2f(0, 0);
  outputQuad[1] = cv::Point2f(cols - 1, 0);
  outputQuad[2] = cv::Point2f(cols - 1, rows - 1);
  outputQuad[3] = cv::Point2f(0, rows - 1);
  if (param.persp_horizontal() || param.all_effects()) {
    vector<float> binary_probs = {0.5, 0.5};
    if (roll_weighted_die(binary_probs) == 1) {
      // seen from right
      caffe_rng_uniform(1, (float)0.0, (float)rows * param.persp_factor(),
                        &outputQuad[0].y);
      outputQuad[3].y = rows - outputQuad[0].y;
    } else {
      // seen from left
      caffe_rng_uniform(1, (float)0.0, (float)rows * param.persp_factor(),
                        &outputQuad[1].y);
      outputQuad[2].y = rows - outputQuad[1].y;
    }
  }
  if (param.persp_vertical() || param.all_effects()) {
    vector<float> binary_probs = {0.5, 0.5};
    if (roll_weighted_die(binary_probs) == 1) {
      // seen from above
      caffe_rng_uniform(1, (float)0.0, (float)cols * param.persp_factor(),
                        &outputQuad[3].x);
      outputQuad[2].x = cols - outputQuad[3].x;
    } else {
      // seen from below
      caffe_rng_uniform(1, (float)0.0, (float)cols * param.persp_factor(),
                        &outputQuad[0].x);
      outputQuad[1].x = cols - outputQuad[0].x;
    }
  }
}
} // namespace internal

const float prob_eps = 0.01;

int roll_weighted_die(const vector<float> &probabilities) {
  vector<float> cumulative;
  std::partial_sum(probabilities.begin(), probabilities.end(),
                   std::back_inserter(cumulative));
  float val;
  caffe_rng_uniform(1, 0.F, cumulative.back(), &val);

  // Find the position within the sequence and add 1
  return (std::lower_bound(cumulative.begin(), cumulative.end(), val) -
          cumulative.begin());
}

void UpdateBBoxByResizePolicy(const ResizeParameter &param, const int old_width,
                              const int old_height, NormalizedBBox *bbox) {
  float new_height = param.height();
  float new_width = param.width();
  float orig_aspect = static_cast<float>(old_width) / old_height;
  float new_aspect = new_width / new_height;

  float x_min = bbox->xmin() * old_width;
  float y_min = bbox->ymin() * old_height;
  float x_max = bbox->xmax() * old_width;
  float y_max = bbox->ymax() * old_height;
  float padding;
  switch (param.resize_mode()) {
  case ResizeParameter_Resize_mode_WARP:
    x_min = std::max(0.f, x_min * new_width / old_width);
    x_max = std::min(new_width, x_max * new_width / old_width);
    y_min = std::max(0.f, y_min * new_height / old_height);
    y_max = std::min(new_height, y_max * new_height / old_height);
    break;
  case ResizeParameter_Resize_mode_FIT_LARGE_SIZE_AND_PAD:
    if (orig_aspect > new_aspect) {
      padding = (new_height - new_width / orig_aspect) / 2;
      x_min = std::max(0.f, x_min * new_width / old_width);
      x_max = std::min(new_width, x_max * new_width / old_width);
      y_min = y_min * (new_height - 2 * padding) / old_height;
      y_min = padding + std::max(0.f, y_min);
      y_max = y_max * (new_height - 2 * padding) / old_height;
      y_max = padding + std::min(new_height, y_max);
    } else {
      padding = (new_width - orig_aspect * new_height) / 2;
      x_min = x_min * (new_width - 2 * padding) / old_width;
      x_min = padding + std::max(0.f, x_min);
      x_max = x_max * (new_width - 2 * padding) / old_width;
      x_max = padding + std::min(new_width, x_max);
      y_min = std::max(0.f, y_min * new_height / old_height);
      y_max = std::min(new_height, y_max * new_height / old_height);
    }
    break;
  case ResizeParameter_Resize_mode_FIT_SMALL_SIZE:
    if (orig_aspect < new_aspect) {
      new_height = new_width / orig_aspect;
    } else {
      new_width = orig_aspect * new_height;
    }
    x_min = std::max(0.f, x_min * new_width / old_width);
    x_max = std::min(new_width, x_max * new_width / old_width);
    y_min = std::max(0.f, y_min * new_height / old_height);
    y_max = std::min(new_height, y_max * new_height / old_height);
    break;
  default:
    LOG(FATAL) << "Unknown resize mode.";
  }
  bbox->set_xmin(x_min / new_width);
  bbox->set_ymin(y_min / new_height);
  bbox->set_xmax(x_max / new_width);
  bbox->set_ymax(y_max / new_height);
}

void InferNewSize(const ResizeParameter &resize_param, const int old_width,
                  const int old_height, int *new_width, int *new_height) {
  int height = resize_param.height();
  int width = resize_param.width();
  float orig_aspect =
      static_cast<float>(old_width) / static_cast<float>(old_height);
  float aspect = static_cast<float>(width) / static_cast<float>(height);

  switch (resize_param.resize_mode()) {
  case ResizeParameter_Resize_mode_WARP:
  case ResizeParameter_Resize_mode_FIT_LARGE_SIZE_AND_PAD:
    break;
  case ResizeParameter_Resize_mode_FIT_SMALL_SIZE:
    if (orig_aspect < aspect) {
      height = static_cast<int>(width / orig_aspect);
    } else {
      width = static_cast<int>(orig_aspect * height);
    }
    break;
  default:
    LOG(FATAL) << "Unknown resize mode.";
  }
  *new_height = height;
  *new_width = width;
}

#ifdef USE_OPENCV

cv::Mat ApplyResize(const cv::Mat &in_img, const ResizeParameter &param) {
  cv::Mat out_img;

  // Reading parameters
  const int new_height = param.height();
  const int new_width = param.width();

  int pad_mode = cv::BORDER_CONSTANT;
  switch (param.pad_mode()) {
  case ResizeParameter_Pad_mode_CONSTANT:
    break;
  case ResizeParameter_Pad_mode_MIRRORED:
    pad_mode = cv::BORDER_REFLECT101;
    break;
  case ResizeParameter_Pad_mode_REPEAT_NEAREST:
    pad_mode = cv::BORDER_REPLICATE;
    break;
  default:
    LOG(FATAL) << "Unknown pad mode.";
  }

  int interp_mode = cv::INTER_LINEAR;
  int num_interp_mode = param.interp_mode_size();
  if (num_interp_mode > 0) {
    vector<float> probs(num_interp_mode, 1.f / num_interp_mode);
    int prob_num = roll_weighted_die(probs);
    switch (param.interp_mode(prob_num)) {
    case ResizeParameter_Interp_mode_AREA:
      interp_mode = cv::INTER_AREA;
      break;
    case ResizeParameter_Interp_mode_CUBIC:
      interp_mode = cv::INTER_CUBIC;
      break;
    case ResizeParameter_Interp_mode_LINEAR:
      interp_mode = cv::INTER_LINEAR;
      break;
    case ResizeParameter_Interp_mode_NEAREST:
      interp_mode = cv::INTER_NEAREST;
      break;
    case ResizeParameter_Interp_mode_LANCZOS4:
      interp_mode = cv::INTER_LANCZOS4;
      break;
    default:
      LOG(FATAL) << "Unknown interp mode.";
    }
  }

  cv::Scalar pad_val = cv::Scalar(0, 0, 0);
  const int img_channels = in_img.channels();
  if (param.pad_value_size() > 0) {
    CHECK(param.pad_value_size() == 1 || param.pad_value_size() == img_channels)
        << "Specify either 1 pad_value or as many as channels: "
        << img_channels;
    vector<float> pad_values;
    for (int i = 0; i < param.pad_value_size(); ++i) {
      pad_values.push_back(param.pad_value(i));
    }
    if (img_channels > 1 && param.pad_value_size() == 1) {
      // Replicate the pad_value for simplicity
      for (int c = 1; c < img_channels; ++c) {
        pad_values.push_back(pad_values[0]);
      }
    }
    pad_val = cv::Scalar(pad_values[0], pad_values[1], pad_values[2]);
  }

  switch (param.resize_mode()) {
  case ResizeParameter_Resize_mode_WARP:
    cv::resize(in_img, out_img, cv::Size(new_width, new_height), 0, 0,
               interp_mode);
    break;
  case ResizeParameter_Resize_mode_FIT_LARGE_SIZE_AND_PAD:
    out_img = internal::AspectKeepingResizeAndPad(
        in_img, new_width, new_height, pad_mode, pad_val, interp_mode);
    break;
  case ResizeParameter_Resize_mode_FIT_SMALL_SIZE:
    out_img = internal::AspectKeepingResizeBySmall(in_img, new_width,
                                                   new_height, interp_mode);
    break;
  default:
    LOG(INFO) << "Unknown resize mode.";
  }
  return out_img;
}

cv::Mat ApplyNoise(const cv::Mat &in_img, const NoiseParameter &param) {
  cv::Mat out_img;
  static constexpr int NOISE_NUM = 11;
  auto *prob = new float[NOISE_NUM];

  caffe_rng_uniform(NOISE_NUM, 0.f, 1.f, prob);
  if (param.decolorize() && param.decolorize_prob() > prob[0]) {
    cv::Mat grayscale_img;
    cv::cvtColor(in_img, grayscale_img, CV_BGR2GRAY);
    cv::cvtColor(grayscale_img, out_img, CV_GRAY2BGR);
  } else {
    out_img = in_img;
  }

  if (param.gauss_blur() && param.gauss_blur_prob() > prob[1]) {
    cv::GaussianBlur(out_img, out_img, cv::Size(5, 5), 1.5);
  }

  if (param.hist_eq() && param.hist_eq_prob() > prob[2]) {
    if (out_img.channels() > 1) {
      cv::Mat ycrcb_image;
      cv::cvtColor(out_img, ycrcb_image, CV_BGR2YCrCb);
      // Extract the L channel
      vector<cv::Mat> ycrcb_planes(3);
      cv::split(ycrcb_image, ycrcb_planes);
      // now we have the L image in ycrcb_planes[0]
      cv::Mat dst;
      cv::equalizeHist(ycrcb_planes[0], dst);
      ycrcb_planes[0] = dst;
      cv::merge(ycrcb_planes, ycrcb_image);
      // convert back to RGB
      cv::cvtColor(ycrcb_image, out_img, CV_YCrCb2BGR);
    } else {
      cv::Mat temp_img;
      cv::equalizeHist(out_img, temp_img);
      out_img = temp_img;
    }
  }

  if (param.clahe() && param.clahe_prob() > prob[3]) {
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(4);
    if (out_img.channels() > 1) {
      cv::Mat ycrcb_image;
      cv::cvtColor(out_img, ycrcb_image, CV_BGR2YCrCb);
      // Extract the L channel
      vector<cv::Mat> ycrcb_planes(3);
      cv::split(ycrcb_image, ycrcb_planes);
      // now we have the L image in ycrcb_planes[0]
      cv::Mat dst;
      clahe->apply(ycrcb_planes[0], dst);
      ycrcb_planes[0] = dst;
      cv::merge(ycrcb_planes, ycrcb_image);
      // convert back to RGB
      cv::cvtColor(ycrcb_image, out_img, CV_YCrCb2BGR);
    } else {
      cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
      clahe->setClipLimit(4);
      cv::Mat temp_img;
      clahe->apply(out_img, temp_img);
      out_img = temp_img;
    }
  }

  if (param.jpeg() > 0 && param.jpeg_prob() > prob[4]) {
    vector<uchar> buf;
    vector<int> params;
    params.push_back(CV_IMWRITE_JPEG_QUALITY);
    params.push_back(param.jpeg());
    cv::imencode(".jpg", out_img, buf, params);
    out_img = cv::imdecode(buf, CV_LOAD_IMAGE_COLOR);
  }

  if (param.erode() && param.erode_prob() > prob[5]) {
    cv::Mat element =
        cv::getStructuringElement(2, cv::Size(3, 3), cv::Point(1, 1));
    cv::erode(out_img, out_img, element);
  }

  if (param.posterize() && param.posterize_prob() > prob[6]) {
    cv::Mat tmp_img;
    tmp_img = internal::colorReduce(out_img);
    out_img = tmp_img;
  }

  if (param.inverse() && param.inverse_prob() > prob[7]) {
    cv::Mat tmp_img;
    cv::bitwise_not(out_img, tmp_img);
    out_img = tmp_img;
  }

  if (param.saltpepper() && param.saltpepper_prob() > prob[8]) {
    vector<uchar> noise_values;
    if (param.saltpepper_param().value_size() > 0) {
      CHECK(param.saltpepper_param().value_size() == 1 ||
            param.saltpepper_param().value_size() == out_img.channels())
          << "Specify either 1 pad_value or as many as channels: "
          << out_img.channels();

      for (int i = 0; i < param.saltpepper_param().value_size(); i++) {
        noise_values.push_back(uchar(param.saltpepper_param().value(i)));
      }
      if (out_img.channels() > 1 &&
          param.saltpepper_param().value_size() == 1) {
        // Replicate the pad_value for simplicity
        for (int c = 1; c < out_img.channels(); ++c) {
          noise_values.push_back(uchar(noise_values[0]));
        }
      }
    } else {
      for (int c = 0; c < out_img.channels(); ++c) {
        noise_values.push_back(0);
      }
    }
    const int noise_pixels_num = floor(param.saltpepper_param().fraction() *
                                       out_img.cols * out_img.rows);
    internal::constantNoise(noise_pixels_num, noise_values, &out_img);
  }

  if (param.convert_to_hsv() && param.convert_to_hsv_prob() > prob[9]) {
    cv::Mat hsv_image;
    cv::cvtColor(out_img, hsv_image, CV_BGR2HSV);
    out_img = hsv_image;
  }

  if (param.convert_to_lab() && param.convert_to_lab_prob() > prob[10]) {
    cv::Mat lab_image;
    out_img.convertTo(lab_image, CV_32F);
    lab_image *= 1.0 / 255;
    cv::cvtColor(lab_image, out_img, CV_BGR2Lab);
  }

  delete[] prob;
  return out_img;
}

cv::Mat ApplyDistort(const cv::Mat &in_img, const DistortionParameter &param) {
  cv::Mat out_img = in_img;
  float prob;
  caffe_rng_uniform(1, 0.f, 1.f, &prob);

  std::vector<std::function<void(void)>> distort_functions;
  distort_functions.reserve(5);
  distort_functions.emplace_back([&]() -> void {
    // Do random brightness distortion.
    internal::RandomBrightness(out_img, &out_img, param.brightness_prob(),
                               param.brightness_delta());
  });
  distort_functions.emplace_back([&]() -> void {
    // Do random hue distortion.
    internal::RandomHue(out_img, &out_img, param.hue_prob(), param.hue_delta());
  });

  distort_functions.emplace_back([&]() -> void {
    // Do random contrast distortion.
    internal::RandomContrast(out_img, &out_img, param.contrast_prob(),
                             param.contrast_lower(), param.contrast_upper());
  });

  distort_functions.emplace_back([&]() -> void {
    // Do random saturation distortion.
    internal::RandomSaturation(out_img, &out_img, param.saturation_prob(),
                               param.saturation_lower(),
                               param.saturation_upper());
  });

  distort_functions.emplace_back([&]() -> void {
    // Do random reordering of the channels.
    internal::RandomOrderChannels(out_img, &out_img, param.random_order_prob());
  });

  if (param.has_random_order_prob() && prob < param.random_order_prob()) {
    shuffle(distort_functions.begin(), distort_functions.end());
  }
  for (auto &f : distort_functions) {
    f();
  }
  return out_img;
}

void ApplyZoom(const cv::Mat &in_img, cv::Mat &out_img, const cv::Mat &in_lbl,
               cv::Mat &out_lbl, const ExpansionParameter &param) {
  if (param.prob() == 0.0) {
    out_img = in_img;
    out_lbl = in_lbl;
    return;
  }

  float z; // zoom factor
  caffe_rng_uniform(1, float(1.0), param.max_expand_ratio(), &z);
  int r = in_img.rows / z;
  int c = in_img.cols / z;
  float scf;
  caffe_rng_uniform(1, float(0.0), (float)(in_img.cols - c), &scf);
  float srf;
  caffe_rng_uniform(1, float(0.0), (float)(in_img.rows - r), &srf);

  cv::Rect zone = cv::Rect((int)scf, (int)srf, c, r);

  cv::resize(in_img(zone), out_img, cv::Size(in_img.cols, in_img.rows), 0, 0,
             cv::INTER_LINEAR);
  cv::resize(in_lbl(zone), out_lbl, cv::Size(in_img.cols, in_img.rows), 0, 0,
             cv::INTER_NEAREST);
}

cv::Mat ApplyGeometry(const cv::Mat &in_img, const GeometryParameter &param) {
  cv::Mat out_img;
  if (param.prob() == 0.0)
    return in_img;

  vector<float> binary_probs;
  if (param.prob() > 0.0)
    binary_probs = {1.f - param.prob(), param.prob()};

  bool persp = (roll_weighted_die(binary_probs) == 1);
  if (!persp)
    return in_img;

  cv::Mat in_img_enlarged;
  internal::getEnlargedImage(in_img, param, in_img_enlarged);

  // Input Quadilateral or Image plane coordinates
  cv::Point2f inputQuad[4];
  // Output Quadilateral or World plane coordinates
  cv::Point2f outputQuad[4];

  internal::getQuads(in_img.rows, in_img.cols, param, inputQuad, outputQuad);

  // Get the Perspective Transform Matrix i.e. lambda
  cv::Mat lambda = getPerspectiveTransform(inputQuad, outputQuad);
  // Apply the Perspective Transform just found to the src image
  //  warpPerspective(in_img,out_img,lambda,in_img.size());
  warpPerspective(in_img_enlarged, out_img, lambda, in_img.size());

  return out_img;
}

void ApplyGeometry(const cv::Mat &in_img, cv::Mat &out_img,
                   const AnnotatedDatum &anno_datum,
                   AnnotatedDatum &geom_anno_datum,
                   const GeometryParameter &param) {
  geom_anno_datum.set_type(anno_datum.type());

  bool nochange = false;
  if (param.prob() > 0.0) {
    vector<float> binary_probs;
    binary_probs = {1.f - param.prob(), param.prob()};
    nochange = (roll_weighted_die(binary_probs) != 1);
  } else
    nochange = true;

  if (nochange) {
    out_img = in_img;
    for (int g = 0; g < anno_datum.annotation_group_size(); ++g)
      geom_anno_datum.mutable_annotation_group()->Add()->CopyFrom(
          anno_datum.annotation_group(g));
    return;
  }

  if (anno_datum.type() != AnnotatedDatum_AnnotationType_BBOX) {
    LOG(ERROR) << "Unknown annotation type.";
    LOG(FATAL) << "fatal error";
  }

  for (int g = 0; g < anno_datum.annotation_group_size(); ++g) {
    std::vector<int> instances;
    std::vector<NormalizedBBox> bboxes;

    const AnnotationGroup &anno_group = anno_datum.annotation_group(g);
    AnnotationGroup geom_anno_group;
    for (int a = 0; a < anno_group.annotation_size(); ++a) {
      const Annotation &anno = anno_group.annotation(a);
      NormalizedBBox bbox = anno.bbox();
      bboxes.push_back(bbox);
      instances.push_back(anno.instance_id());
    }

    cv::Mat in_img_enlarged;
    internal::getEnlargedImage(in_img, param, in_img_enlarged);
    internal::getEnlargedBBoxes(in_img.rows, in_img.cols, param, bboxes,
                                instances);

    // Input Quadilateral or Image plane coordinates
    cv::Point2f inputQuad[4];
    // Output Quadilateral or World plane coordinates
    cv::Point2f outputQuad[4];

    internal::getQuads(in_img.rows, in_img.cols, param, inputQuad, outputQuad);

    // Get the Perspective Transform Matrix i.e. lambda
    cv::Mat lambda = getPerspectiveTransform(inputQuad, outputQuad);
    warpPerspective(in_img_enlarged, out_img, lambda, in_img.size());

    internal::warpBBoxes(bboxes, lambda, in_img.rows, in_img.cols);
    internal::filterBBoxes(bboxes, instances);

    for (int i = 0; i < bboxes.size(); ++i) {
      NormalizedBBox bb = bboxes[i];
      Annotation *geom_annot = geom_anno_group.add_annotation();
      geom_annot->set_instance_id(instances[i]);
      NormalizedBBox *geom_bbox = geom_annot->mutable_bbox();
      geom_bbox->CopyFrom(bb);
    }
    geom_anno_group.set_group_label(anno_group.group_label());
    geom_anno_datum.mutable_annotation_group()->Add()->CopyFrom(
        geom_anno_group);
  }
}

void ApplyGeometry(const cv::Mat &in_img, cv::Mat &out_img,
                   const cv::Mat &in_lbl, cv::Mat &out_lbl,
                   const GeometryParameter &param) {

  if (param.prob() == 0.0) {
    out_img = in_img;
    out_lbl = in_lbl;
    return;
  }

  vector<float> binary_probs;
  if (param.prob() > 0.0)
    binary_probs = {1.f - param.prob(), param.prob()};

  bool persp = (roll_weighted_die(binary_probs) == 1);
  if (!persp) {
    out_img = in_img;
    out_lbl = in_lbl;
    return;
  }

  cv::Mat in_img_enlarged;
  internal::getEnlargedImage(in_img, param, in_img_enlarged);
  cv::Mat in_lbl_enlarged;
  internal::getEnlargedImage(in_lbl, param, in_lbl_enlarged);

  // Input Quadilateral or Image plane coordinates
  cv::Point2f inputQuad[4];
  // Output Quadilateral or World plane coordinates
  cv::Point2f outputQuad[4];

  internal::getQuads(in_img.rows, in_img.cols, param, inputQuad, outputQuad);

  // Get the Perspective Transform Matrix i.e. lambda
  cv::Mat lambda = getPerspectiveTransform(inputQuad, outputQuad);
  // Apply the Perspective Transform just found to the src image
  //  warpPerspective(in_img,out_img,lambda,in_img.size());
  warpPerspective(in_img_enlarged, out_img, lambda, in_img.size());
  warpPerspective(in_lbl_enlarged, out_lbl, lambda, in_lbl.size(),
                  cv::INTER_NEAREST);
}
#endif // USE_OPENCV

} // namespace caffe
