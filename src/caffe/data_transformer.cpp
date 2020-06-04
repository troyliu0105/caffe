#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif // USE_OPENCV

#include <string>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/util/bbox_util.hpp"
#include "caffe/util/im_transforms.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
const double prob_eps = 0.01;

#ifdef DEBUG
static int iter_count = 0;
#endif

namespace caffe {

#pragma region helpers
static void rotate(cv::Mat &src, int angle) {
  // get rotation matrix for rotating the image around its center
  cv::Point2f center(src.cols / 2.0, src.rows / 2.0);
  cv::Mat rot = cv::getRotationMatrix2D(center, angle, 1.0);
  // determine bounding rectangle
  cv::Rect bbox = cv::RotatedRect(center, src.size(), angle).boundingRect();
  // adjust transformation matrix
  rot.at<double>(0, 2) += bbox.width / 2.0 - center.x;
  rot.at<double>(1, 2) += bbox.height / 2.0 - center.y;
  cv::warpAffine(src, src, rot, bbox.size());
}

static void resize(cv::Mat &cv_img, int smallest_side) {
  int cur_width = cv_img.cols;
  int cur_height = cv_img.rows;
  cv::Size dsize;
  if (cur_height <= cur_width) {
    double k = ((double)cur_height) / smallest_side;
    int new_size = (int)ceil(cur_width / k);
    dsize = cv::Size(new_size, smallest_side);
  } else {
    double k = ((double)cur_width) / smallest_side;
    int new_size = (int)ceil(cur_height / k);
    dsize = cv::Size(smallest_side, new_size);
  }
  cv::resize(cv_img, cv_img, dsize);
}

template <typename T>
static std::tuple<int, int> get_cv_type(int channels) {
  int type, ctype;
  switch (sizeof(T)) {
  case 4:
    ctype = CV_32FC(channels);
    type = CV_32FC1;
    break;
  case 8:
    ctype = CV_64FC(channels);
    type = CV_64FC1;
    break;
  default:
    LOG(FATAL) << "DataTransformer Dtype can only be float or double";
    break;
  }
  return std::make_tuple(type, ctype);
}
#pragma endregion

template <typename Dtype>
DataTransformer<Dtype>::DataTransformer(const TransformationParameter &param,
                                        Phase phase)
    : param_(param), phase_(phase), mirror_param_(false) {
  // check if we want to use mean_file
  if (param_.has_mean_file()) {
    CHECK_EQ(param_.mean_value_size(), 0)
        << "Cannot specify mean_file and mean_value at the same time";
    const string &mean_file = param.mean_file();
    if (Caffe::root_solver()) {
      LOG(INFO) << "Loading mean file from: " << mean_file;
    }
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
    data_mean_.FromProto(blob_proto);
    cv_mean_.reset(new cv::Mat());
    Dtype *mean_data = data_mean_.mutable_cpu_data();
    auto [type, ctype] = get_cv_type<Dtype>(blob_proto.channels());
    vector<cv::Mat> mats;
    mats.reserve(blob_proto.channels());
    for (int c = 0; c < blob_proto.channels(); ++c) {
      mats.emplace_back(blob_proto.height(), blob_proto.width(), type,
                        mean_data +
                            c * blob_proto.height() * blob_proto.width());
    }
    cv::merge(mats, *cv_mean_);
  }
  // check if we want to use mean_value
  if (param_.mean_value_size() > 0) {
    CHECK(!param_.has_mean_file())
        << "Cannot specify mean_file and mean_value at the same time";
    for (int c = 0; c < param_.mean_value_size(); ++c) {
      mean_values_.push_back(param_.mean_value(c));
    }
    auto [type, ctype] = get_cv_type<Dtype>(mean_values_.size());
    cv_mean_.reset(new cv::Mat(1, 1, type, mean_values_.data()));
  }
  // if (param_.has_resize_param()) {
  //  CHECK_GT(param_.resize_param().height(), 0);
  //  CHECK_GT(param_.resize_param().width(), 0);
  //}
  if (param_.has_expand_param()) {
    CHECK_GT(param_.expand_param().max_expand_ratio(), 1.);
  }
}

#pragma region datum_trans
////////////////////////////////////////////////////////////////////////////////
///////////////////////////// Datum ↓↓↓ Transforms /////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template <typename Dtype>
void DataTransformer<Dtype>::Transform(const Datum &datum,
                                       Dtype *transformed_data,
                                       NormalizedBBox *crop_bbox,
                                       bool *do_mirror) {

  const string &data = datum.data();
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();

  const int crop_size = param_.crop_size();
  const Dtype scale = param_.scale();
  *do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_uint8 = !data.empty();
  const bool has_mean_values = !mean_values_.empty();

  CHECK_GT(datum_channels, 0);
  CHECK_GE(datum_height, crop_size);
  CHECK_GE(datum_width, crop_size);
  auto [type, ctype] = get_cv_type<Dtype>(datum_channels);
  //  Dtype *mean = nullptr;
  if (has_mean_file) {
    //    CHECK_EQ(datum_channels, data_mean_.channels());
    //    CHECK_EQ(datum_height, data_mean_.height());
    //    CHECK_EQ(datum_width, data_mean_.width());
    //    mean = data_mean_.mutable_cpu_data();
    CHECK_EQ(datum_channels, cv_mean_->channels());
    CHECK_EQ(datum_height, cv_mean_->rows);
    CHECK_EQ(datum_width, cv_mean_->cols);
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == datum_channels)
        << "Specify either 1 mean_value or as many as channels: "
        << datum_channels;
    if (datum_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < datum_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
    cv_mean_.reset(new cv::Mat(1, 1, ctype, mean_values_.data()));
  }

  int height = datum_height;
  int width = datum_width;

  int h_off = 0;
  int w_off = 0;
  if (crop_size) {
    height = crop_size;
    width = crop_size;
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = Rand(datum_height - crop_size + 1);
      w_off = Rand(datum_width - crop_size + 1);
    } else {
      h_off = (datum_height - crop_size) / 2;
      w_off = (datum_width - crop_size) / 2;
    }
  }
  // LOG(INFO) << width << "," << height;
  // Return the normalized crop bbox.
  crop_bbox->set_xmin(Dtype(w_off) / datum_width);
  crop_bbox->set_ymin(Dtype(h_off) / datum_height);
  crop_bbox->set_xmax(Dtype(w_off + width) / datum_width);
  crop_bbox->set_ymax(Dtype(h_off + height) / datum_height);

  cv::Mat cv_img = DatumToCVMat(datum);
  cv_img.convertTo(cv_img, ctype);

  vector<cv::Mat> transformed_mats;
  transformed_mats.reserve(datum_channels);
  for (int c = 0; c < datum_channels; ++c) {
    transformed_mats.emplace_back(height, width, type,
                                  transformed_data + c * height * width);
  }

  cv::Rect roi(crop_bbox->xmin() * datum_width,
               crop_bbox->ymin() * datum_height,
               (crop_bbox->xmax() - crop_bbox->xmin()) * datum_width,
               (crop_bbox->ymax() - crop_bbox->ymin()) * datum_height);
  cv_img = cv_img(roi);
  if (*do_mirror) {
    cv::flip(cv_img, cv_img, 1);
  }

  if (has_mean_file || has_mean_values) {
    if (cv_img.size != cv_mean_->size) {
      cv::resize(*cv_mean_, *cv_mean_, cv::Size(cv_img.cols, cv_img.rows));
    }
    cv_img -= *cv_mean_;
  }
  cv_img *= scale;
  cv::split(cv_img, transformed_mats);
}

template <typename Dtype>
void DataTransformer<Dtype>::Transform(const Datum &datum,
                                       Dtype *transformed_data) {

  NormalizedBBox crop_bbox;
  bool do_mirror;
  Transform(datum, transformed_data, &crop_bbox, &do_mirror);
}

template <typename Dtype>
void DataTransformer<Dtype>::Transform(const Datum &datum,
                                       Blob<Dtype> *transformed_blob,
                                       NormalizedBBox *crop_bbox,
                                       bool *do_mirror, int policy_num) {
  // LOG(INFO) << policy_num;
  // If datum is encoded, decoded and transform the cv::image.
  CHECK(!(param_.force_color() && param_.force_gray()))
      << "cannot set both force_color and force_gray";

#ifdef USE_OPENCV
  cv::Mat cv_img;

  if (datum.encoded()) {
    if (param_.force_color() || param_.force_gray()) {
      // If force_color then decode in color otherwise decode in gray.
      cv_img = DecodeDatumToCVMat(datum, param_.force_color());
    } else {
      cv_img = DecodeDatumToCVMatNative(datum);
    }
  } else {
    cv_img = DatumToCVMat(datum);
    if (param_.force_gray() && cv_img.channels() == 3) {
      cv::cvtColor(cv_img, cv_img, cv::COLOR_BGR2GRAY);
    }
    if (param_.force_color() && cv_img.channels() == 1) {
      cv::cvtColor(cv_img, cv_img, cv::COLOR_GRAY2BGR);
    }
  }

  if (param_.enable_classf_aug()) {
    return Transform3(cv_img, transformed_blob);
  } else {
    return Transform(cv_img, transformed_blob, crop_bbox, do_mirror,
                     policy_num);
  }
#else
  LOG(FATAL) << "Encoded datum requires OpenCV; compile with USE_OPENCV.";
#endif // USE_OPENCV
  // LOG(INFO) << "test";
  //  const int crop_size = param_.crop_size();
  //  const int datum_channels = datum.channels();
  //  const int datum_height = datum.height();
  //  const int datum_width = datum.width();
  //
  //  // Check dimensions.
  //  const int num = transformed_blob->num();
  //  const int channels = transformed_blob->channels();
  //  const int height = transformed_blob->height();
  //  const int width = transformed_blob->width();
  //
  //  CHECK_EQ(channels, datum_channels);
  //  CHECK_LE(height, datum_height);
  //  CHECK_LE(width, datum_width);
  //  CHECK_GE(num, 1);
  //
  //  if (crop_size) {
  //    CHECK_EQ(crop_size, height);
  //    CHECK_EQ(crop_size, width);
  //  } else {
  //    CHECK_EQ(datum_height, height);
  //    CHECK_EQ(datum_width, width);
  //  }
  //
  //  Dtype *transformed_data = transformed_blob->mutable_cpu_data();
  //  Transform(datum, transformed_data, crop_bbox, do_mirror);
}

template <typename Dtype>
void DataTransformer<Dtype>::Transform(const Datum &datum,
                                       Blob<Dtype> *transformed_blob,
                                       int policy_num) {
  NormalizedBBox crop_bbox;
  bool do_mirror;
  Transform(datum, transformed_blob, &crop_bbox, &do_mirror, policy_num);
  // entry point 1
}

template <typename Dtype>
void DataTransformer<Dtype>::Transform(const vector<Datum> &datum_vector,
                                       Blob<Dtype> *transformed_blob) {
  const int datum_num = datum_vector.size();
  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();

  CHECK_GT(datum_num, 0) << "There is no datum to add";
  CHECK_LE(datum_num, num) << "The size of datum_vector must be no greater "
                              "than transformed_blob->num()";
  Blob<Dtype> uni_blob(1, channels, height, width);
  for (int item_id = 0; item_id < datum_num; ++item_id) {
    int offset = transformed_blob->offset(item_id);
    uni_blob.set_cpu_data(transformed_blob->mutable_cpu_data() + offset);
    Transform(datum_vector[item_id], &uni_blob);
  }
}
////////////////////////////////////////////////////////////////////////////////
#pragma endregion

#pragma region mat_trans
////////////////////////////////////////////////////////////////////////////////
///////////////////////////// cvMat ↓↓↓ Transforms /////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template <typename Dtype>
void DataTransformer<Dtype>::Transform(const vector<cv::Mat> &mat_vector,
                                       Blob<Dtype> *transformed_blob) {
  const int mat_num = mat_vector.size();
  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();

  CHECK_GT(mat_num, 0) << "There is no MAT to add";
  CHECK_EQ(mat_num, num)
      << "The size of mat_vector must be equals to transformed_blob->num()";
  Blob<Dtype> uni_blob(1, channels, height, width);
  for (int item_id = 0; item_id < mat_num; ++item_id) {
    int offset = transformed_blob->offset(item_id);
    uni_blob.set_cpu_data(transformed_blob->mutable_cpu_data() + offset);
    Transform(mat_vector[item_id], &uni_blob);
  }
}

template <typename Dtype>
void DataTransformer<Dtype>::Transform3(const cv::Mat &img,
                                        Blob<Dtype> *transformed_blob) {

  // Copy from https://github.com/twtygqyy/caffe-augmentation

  const int min_side = param_.min_side();
  const int min_side_min = param_.min_side_min();
  const int min_side_max = param_.min_side_max();
  const int crop_size = param_.crop_size();
  const int rotation_angle = param_.max_rotation_angle();
  const float min_contrast = param_.min_contrast();
  const float max_contrast = param_.max_contrast();
  const int max_brightness_shift = param_.max_brightness_shift();
  const float max_smooth = param_.max_smooth();
  const int max_color_shift = param_.max_color_shift();
  const float apply_prob = 1.f - param_.apply_probability();
  const bool debug_params = param_.debug_params();

  // Check dimensions.
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int num = transformed_blob->num();

  const Dtype scale = param_.scale();
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = !mean_values_.empty();

  float current_prob;

  const bool do_rotation = rotation_angle > 0 && phase_ == TRAIN;

  const bool do_resize_to_min_side = min_side > 0;
  const bool do_resize_to_min_side_min = min_side_min > 0;
  const bool do_resize_to_min_side_max = min_side_max > 0;

  const bool do_mirror = param_.mirror() && phase_ == TRAIN && Rand(2);

  caffe_rng_uniform(1, 0.f, 1.f, &current_prob);
  const bool do_brightness = param_.contrast_brightness_adjustment() &&
                             phase_ == TRAIN && current_prob > apply_prob;

  caffe_rng_uniform(1, 0.f, 1.f, &current_prob);
  const bool do_smooth = param_.smooth_filtering() && phase_ == TRAIN &&
                         max_smooth > 1 && current_prob > apply_prob;

  caffe_rng_uniform(1, 0.f, 1.f, &current_prob);
  const bool do_color_shift =
      max_color_shift > 0 && phase_ == TRAIN && current_prob > apply_prob;

  cv::Mat cv_img = img;

  int current_angle = 0;
  if (do_rotation) {
    current_angle = Rand(rotation_angle * 2 + 1) - rotation_angle;
    if (current_angle)
      rotate(cv_img, current_angle);
  }

  // resizing and crop according to min side, preserving aspect ratio
  if (do_resize_to_min_side) {
    resize(cv_img, min_side);
    // random_crop(cv_img, min_side);
  }

  if (do_resize_to_min_side_min && do_resize_to_min_side_max) {
    // std::cout << min_side_min << " "<<min_side_max<<std::endl;
    int min_side_length = min_side_min + Rand(min_side_max - min_side_min + 1);
    resize(cv_img, min_side_length);
    // crop_center(cv_img, min_side, min_side);
    // random_crop(cv_img, min_side_length);
  }

  // apply color shift
  if (do_color_shift) {
    int b = Rand(max_color_shift + 1);
    int g = Rand(max_color_shift + 1);
    int r = Rand(max_color_shift + 1);
    int sign = Rand(2);

    cv::Mat shiftArr = cv_img.clone();
    shiftArr.setTo(cv::Scalar(b, g, r));

    if (sign == 1) {
      cv_img -= shiftArr;
    } else {
      cv_img += shiftArr;
    }
  }

  // set contrast and brightness
  float alpha;
  int beta;
  if (do_brightness) {
    caffe_rng_uniform(1, min_contrast, max_contrast, &alpha);
    beta = Rand(max_brightness_shift * 2 + 1) - max_brightness_shift;
    cv_img.convertTo(cv_img, -1, alpha, beta);
  }

  // set smoothness
  int smooth_param = 0;
  int smooth_type = 0;
  if (do_smooth) {
    smooth_type = Rand(4);
    smooth_param = 1 + 2 * Rand(max_smooth / 2);
    switch (smooth_type) {
    case 0:
      cv::GaussianBlur(cv_img, cv_img, cv::Size(smooth_param, smooth_param), 0);
      break;
    case 1:
      cv::blur(cv_img, cv_img, cv::Size(smooth_param, smooth_param));
      break;
    case 2:
      cv::medianBlur(cv_img, cv_img, smooth_param);
      break;
    case 3:
      cv::boxFilter(cv_img, cv_img, -1,
                    cv::Size(smooth_param * 2, smooth_param * 2));
      break;
    default:
      break;
    }
  }

  if (debug_params && phase_ == TRAIN) {
    LOG(INFO) << "----------------------------------------";

    if (do_rotation) {
      LOG(INFO) << "* parameter for rotation: ";
      LOG(INFO) << "  current rotation angle: " << current_angle;
    }
    if (do_brightness) {
      LOG(INFO) << "* parameter for contrast adjustment: ";
      LOG(INFO) << "  alpha: " << alpha << ", beta: " << beta;
    }
    if (do_smooth) {
      LOG(INFO) << "* parameter for smooth filtering: ";
      LOG(INFO) << "  smooth type: " << smooth_type
                << ", smooth param: " << smooth_param;
    }
  }

  const int img_channels = cv_img.channels();
  const int img_height = cv_img.rows;
  const int img_width = cv_img.cols;

  CHECK_GT(img_channels, 0);
  CHECK_GE(img_height, crop_size);
  CHECK_GE(img_width, crop_size);

  CHECK_EQ(channels, img_channels);
  CHECK_LE(height, img_height);
  CHECK_LE(width, img_width);
  CHECK_GE(num, 1);

  CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";

  auto [type, ctype] = get_cv_type<Dtype>(img_channels);

  //  Dtype *mean = nullptr;
  if (has_mean_file) {
    //    CHECK_EQ(img_channels, data_mean_.channels());
    //    CHECK_EQ(img_height, data_mean_.height());
    //    CHECK_EQ(img_width, data_mean_.width());
    //    mean = data_mean_.mutable_cpu_data();
    CHECK_EQ(img_channels, cv_mean_->channels());
    CHECK_EQ(height, cv_mean_->rows);
    CHECK_EQ(width, cv_mean_->cols);
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == img_channels)
        << "Specify either 1 mean_value or as many as channels: "
        << img_channels;
    if (img_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < img_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
    cv_mean_.reset(new cv::Mat(1, 1, ctype, mean_values_.data()));
  }

  int h_off = 0;
  int w_off = 0;
  cv::Mat cv_cropped_img = cv_img;
  if (crop_size) {
    CHECK_EQ(crop_size, height);
    CHECK_EQ(crop_size, width);
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = Rand(img_height - crop_size + 1);
      w_off = Rand(img_width - crop_size + 1);
    } else {
      h_off = (img_height - crop_size) / 2;
      w_off = (img_width - crop_size) / 2;
    }
    cv::Rect roi(w_off, h_off, crop_size, crop_size);
    cv_cropped_img = cv_img(roi);
  } else {
    // CHECK_EQ(img_height, height);
    // CHECK_EQ(img_width, width);
  }

  CHECK(cv_cropped_img.data);

  if (do_mirror) {
    cv::flip(cv_cropped_img, cv_cropped_img, 1);
  }
  cv_cropped_img.convertTo(cv_cropped_img, ctype);
  if (has_mean_file || has_mean_values) {
    if (cv_cropped_img.size != cv_mean_->size) {
      cv::resize(*cv_mean_, *cv_mean_,
                 cv::Size(cv_cropped_img.cols, cv_cropped_img.rows));
    }
    cv_cropped_img -= *cv_mean_;
  }
  cv_cropped_img *= scale;

  Dtype *transformed_data = transformed_blob->mutable_cpu_data();
  vector<cv::Mat> transformed_mats;
  transformed_mats.reserve(img_channels);
  for (int c = 0; c < img_channels; ++c) {
    transformed_mats.emplace_back(height, width, type,
                                  transformed_data + c * height * width);
  }
  cv::split(cv_cropped_img, transformed_mats);
}

template <typename Dtype>
void DataTransformer<Dtype>::Transform2(const std::vector<cv::Mat> &cv_imgs,
                                        Blob<Dtype> *transformed_blob,
                                        bool preserve_pixel_vals) {

  // Check dimensions.
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int num = transformed_blob->num();
  // LOG(INFO) << img_channels;
  // CHECK_EQ(channels, img_channels);

  // CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";
  const int crop_size = param_.crop_size();
  const float scale = 1 / 255.0F;
  const bool do_mirror = param_.mirror() && Rand(2);

  // LOG(INFO) << scale << ","<< mean_values_[0] << ","<< mean_values_[1];
  Dtype *transformed_data = transformed_blob->mutable_cpu_data();
  for (int i = 0; i < cv_imgs.size(); i++) {
    // LOG(INFO)<<i;
    auto cv_img = cv_imgs[i];
    const int img_channels = cv_img.channels();
    const int img_height = cv_img.rows;
    const int img_width = cv_img.cols;
    CHECK_LE(height, img_height);
    CHECK_LE(width, img_width);
    CHECK_GE(num, 1);
    CHECK_GT(img_channels, 0);
    // LOG(INFO) << do_mirror;
    int maxima = 0;
    int top_index;
    for (int h = 0; h < height; ++h) {
      const uchar *ptr = cv_img.ptr<uchar>(h);
      int img_index = 0;
      for (int w = 0; w < width; ++w) {
        // for (int c = 0; c < img_channels; ++c) {
        int c = i;
        if (mirror_param_) {
          top_index = (c * height + h) * width + (width - 1 - w);
        } else {
          top_index = (c * height + h) * width + w;
        }
        // LOG(INFO) << top_index;
        // int top_index = (c * height + h) * width + w;
        auto pixel = static_cast<Dtype>(ptr[img_index++]);
        // if(pixel>0)
        //  LOG(INFO) << pixel;
        transformed_data[top_index] = pixel * scale;
        // LOG(INFO) << transformed_data[top_index];
        if (top_index > maxima)
          maxima = top_index;
        //}
      }
    }
    // LOG(INFO)<<maxima;
  }
}
template <typename Dtype>
void DataTransformer<Dtype>::Transform(const cv::Mat &cv_img,
                                       Blob<Dtype> *transformed_blob,
                                       NormalizedBBox *crop_bbox,
                                       bool *do_mirror, int policy_num) {
  // LOG(INFO) << policy_num;
  // Check dimensions.
  const int img_channels = cv_img.channels();
  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const bool cvt = param_.cvt_bgr2rgb();
  CHECK_GT(img_channels, 0);
  CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";
  CHECK_EQ(channels, img_channels);
  CHECK_GE(num, 1);
  if (cvt) {
    cv::cvtColor(cv_img, cv_img, cv::COLOR_BGR2RGB);
  }
  const int crop_size = param_.crop_size();
  const Dtype scale = param_.scale();
  *do_mirror = param_.mirror() && Rand(2);
  mirror_param_ = *do_mirror;
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = !mean_values_.empty();

  const int num_resize_policies = param_.resize_param_size();

  auto [type, ctype] = get_cv_type<Dtype>(img_channels);
  //  Dtype *mean = nullptr;
  if (has_mean_file) {
    //    CHECK_EQ(img_channels, data_mean_.channels());
    //    mean = data_mean_.mutable_cpu_data();
    CHECK_EQ(img_channels, cv_mean_->channels());
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == img_channels)
        << "Specify either 1 mean_value or as many as channels: "
        << img_channels;
    if (img_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < img_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
    cv_mean_.reset(new cv::Mat(1, 1, ctype, mean_values_.data()));
  }

  int crop_h = param_.crop_h();
  int crop_w = param_.crop_w();
  if (crop_size) {
    crop_h = crop_size;
    crop_w = crop_size;
  }

  cv::Mat cv_resized_image, cv_noised_image, cv_cropped_image;
  if (param_.resize_param_size()) {
    cv_resized_image = ApplyResize(cv_img, param_.resize_param(policy_num));
    // LOG(INFO) << *do_mirror << ",data";
  } else {
    cv_resized_image = cv_img;
  }
  if (param_.has_noise_param()) {
    cv_noised_image = ApplyNoise(cv_resized_image, param_.noise_param());
  } else {
    cv_noised_image = cv_resized_image;
  }
  int img_height = cv_noised_image.rows;
  int img_width = cv_noised_image.cols;

  CHECK_GE(img_height, crop_h);
  CHECK_GE(img_width, crop_w);
  // LOG(INFO)<<img_width<<","<<img_height;
  int h_off = 0;
  int w_off = 0;
  if ((crop_h > 0) && (crop_w > 0)) {
    CHECK_EQ(crop_h, height);
    CHECK_EQ(crop_w, width);
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = Rand(img_height - crop_h + 1);
      w_off = Rand(img_width - crop_w + 1);
    } else {
      h_off = (img_height - crop_h) / 2;
      w_off = (img_width - crop_w) / 2;
    }
    cv::Rect roi(w_off, h_off, crop_w, crop_h);
    cv_cropped_image = cv_noised_image(roi);
  } else {
    cv_cropped_image = cv_noised_image;
  }

#ifdef DEBUG
  char buf[1000];
  sprintf(buf, "input/input_%05d.jpg", iter_count++);
  if (*do_mirror) {
    cv::flip(cv_cropped_image, cv_cropped_image, 1);
  }
  cv::imwrite(buf, cv_resized_image);
#endif

  // Return the normalized crop bbox.
  crop_bbox->set_xmin(Dtype(w_off) / img_width);
  crop_bbox->set_ymin(Dtype(h_off) / img_height);
  crop_bbox->set_xmax(Dtype(w_off + width) / img_width);
  crop_bbox->set_ymax(Dtype(h_off + height) / img_height);
  // LOG(INFO)<<width <<","<<height;
  //  if (has_mean_file) {
  //    CHECK_EQ(cv_cropped_image.rows, data_mean_.height());
  //    CHECK_EQ(cv_cropped_image.cols, data_mean_.width());
  //  }
  CHECK(cv_cropped_image.data);

  if (*do_mirror) {
    cv::flip(cv_cropped_image, cv_cropped_image, 1);
  }
  cv_cropped_image.convertTo(cv_cropped_image, ctype);
  if (has_mean_file || has_mean_values) {
    if (cv_cropped_image.size != cv_mean_->size) {
      cv::resize(*cv_mean_, *cv_mean_,
                 cv::Size(cv_cropped_image.cols, cv_cropped_image.rows));
    }
    cv_cropped_image -= *cv_mean_;
  }
  cv_cropped_image *= scale;

  Dtype *transformed_data = transformed_blob->mutable_cpu_data();
  vector<cv::Mat> transformed_mats;
  transformed_mats.reserve(img_channels);
  for (int c = 0; c < img_channels; ++c) {
    transformed_mats.emplace_back(height, width, type,
                                  transformed_data + c * height * width);
  }

  cv::split(cv_cropped_image, transformed_mats);
}

template <typename Dtype>
void DataTransformer<Dtype>::Transform(const cv::Mat &cv_img,
                                       Blob<Dtype> *transformed_blob) {
  NormalizedBBox crop_bbox;
  bool do_mirror;
  Transform(cv_img, transformed_blob, &crop_bbox, &do_mirror);
}

////////////////////////////////////////////////////////////////////////////////
#pragma endregion

#pragma region annotation_trans
////////////////////////////////////////////////////////////////////////////////
////////////////////////// Annotation ↓↓↓ Transforms ///////////////////////////
////////////////////////////////////////////////////////////////////////////////

template <typename Dtype>
void DataTransformer<Dtype>::Transform(
    const AnnotatedDatum &anno_datum, Blob<Dtype> *transformed_blob,
    RepeatedPtrField<AnnotationGroup> *transformed_anno_group_all,
    bool *do_mirror, int policy_num) {
  // Transform datum.
  const Datum &datum = anno_datum.datum();
  NormalizedBBox crop_bbox;

  Transform(datum, transformed_blob, &crop_bbox, do_mirror, policy_num);

  // Transform annotation.
  const bool do_resize = true;
  TransformAnnotation(anno_datum, do_resize, crop_bbox, *do_mirror,
                      transformed_anno_group_all, policy_num);
}

template <typename Dtype>
void DataTransformer<Dtype>::Transform(
    const AnnotatedDatum &anno_datum, Blob<Dtype> *transformed_blob,
    RepeatedPtrField<AnnotationGroup> *transformed_anno_group_all,
    int policy_num) {
  bool do_mirror;

  Transform(anno_datum, transformed_blob, transformed_anno_group_all,
            &do_mirror, policy_num);
}

template <typename Dtype>
void DataTransformer<Dtype>::Transform(
    const AnnotatedDatum &anno_datum, Blob<Dtype> *transformed_blob,
    vector<AnnotationGroup> *transformed_anno_vec, bool *do_mirror,
    int policy_num) {
  // LOG(INFO) << policy_num;
  RepeatedPtrField<AnnotationGroup> transformed_anno_group_all;
  Transform(anno_datum, transformed_blob, &transformed_anno_group_all,
            do_mirror, policy_num);
  for (int g = 0; g < transformed_anno_group_all.size(); ++g) {
    transformed_anno_vec->push_back(transformed_anno_group_all.Get(g));
  }
}

template <typename Dtype>
void DataTransformer<Dtype>::Transform(
    const AnnotatedDatum &anno_datum, Blob<Dtype> *transformed_blob,
    vector<AnnotationGroup> *transformed_anno_vec, int policy_num) {
  bool do_mirror;
  // LOG(INFO) << policy_num;
  Transform(anno_datum, transformed_blob, transformed_anno_vec, &do_mirror,
            policy_num);
}

template <typename Dtype>
void DataTransformer<Dtype>::TransformAnnotation(
    const AnnotatedDatum &anno_datum, const bool do_resize,
    const NormalizedBBox &crop_bbox, const bool do_mirror,
    RepeatedPtrField<AnnotationGroup> *transformed_anno_group_all,
    int policy_num) {
  const int img_height = anno_datum.datum().height();
  const int img_width = anno_datum.datum().width();
  const int num_resize_policies = param_.resize_param_size();
  // LOG(INFO) << policy_num;
  // LOG(INFO) << img_width << "," << img_height;
  if (anno_datum.type() == AnnotatedDatum_AnnotationType_BBOX ||
      anno_datum.type() == AnnotatedDatum_AnnotationType_BBOXandSeg) {
    // Go through each AnnotationGroup.
    for (int g = 0; g < anno_datum.annotation_group_size(); ++g) {
      const AnnotationGroup &anno_group = anno_datum.annotation_group(g);
      AnnotationGroup transformed_anno_group;
      // Go through each Annotation.
      bool has_valid_annotation = false;
      for (int a = 0; a < anno_group.annotation_size(); ++a) {
        const Annotation &anno = anno_group.annotation(a);
        const NormalizedBBox &bbox = anno.bbox();
        // Adjust bounding box annotation.
        NormalizedBBox resize_bbox = bbox;
        if (do_resize && param_.resize_param_size()) {
          CHECK_GT(img_height, 0);
          CHECK_GT(img_width, 0);
          UpdateBBoxByResizePolicy(param_.resize_param(policy_num), img_width,
                                   img_height, &resize_bbox);
        }
        if (param_.has_emit_constraint() &&
            !MeetEmitConstraint(crop_bbox, resize_bbox,
                                param_.emit_constraint())) {
          continue;
        }
        NormalizedBBox proj_bbox;
        if (ProjectBBox(crop_bbox, resize_bbox, &proj_bbox)) {
          has_valid_annotation = true;
          Annotation *transformed_anno =
              transformed_anno_group.add_annotation();
          transformed_anno->set_instance_id(anno.instance_id());
          NormalizedBBox *transformed_bbox = transformed_anno->mutable_bbox();
          transformed_bbox->CopyFrom(proj_bbox);
          if (do_mirror) {
            Dtype temp = transformed_bbox->xmin();
            transformed_bbox->set_xmin(1 - transformed_bbox->xmax());
            transformed_bbox->set_xmax(1 - temp);
          }
          if (do_resize && param_.resize_param_size()) {
            ExtrapolateBBox(param_.resize_param(policy_num), img_height,
                            img_width, crop_bbox, transformed_bbox);
          }
        }
        // LOG(INFO) << num_resize_policies;
      }
      // Save for output.
      if (has_valid_annotation) {
        transformed_anno_group.set_group_label(anno_group.group_label());
        transformed_anno_group_all->Add()->CopyFrom(transformed_anno_group);
      }
    }
  } else {
    LOG(FATAL) << "Unknown annotation type.";
  }
}

template <typename Dtype>
void DataTransformer<Dtype>::CropImage(const AnnotatedDatum &anno_datum,
                                       const NormalizedBBox &bbox,
                                       AnnotatedDatum *cropped_anno_datum,
                                       bool has_anno) {
  // Crop the datum.
  CropImage(anno_datum.datum(), bbox, cropped_anno_datum->mutable_datum());
  if (has_anno) {
    cropped_anno_datum->set_type(anno_datum.type());

    // Transform the annotation according to crop_bbox.
    const bool do_resize = false;
    const bool do_mirror = false;
    NormalizedBBox crop_bbox;
    ClipBBox(bbox, &crop_bbox);
    TransformAnnotation(anno_datum, do_resize, crop_bbox, do_mirror,
                        cropped_anno_datum->mutable_annotation_group());
  }
}

template <typename Dtype>
void DataTransformer<Dtype>::ExpandImage(const AnnotatedDatum &anno_datum,
                                         AnnotatedDatum *expanded_anno_datum) {
  if (!param_.has_expand_param()) {
    expanded_anno_datum->CopyFrom(anno_datum);
    return;
  }
  const ExpansionParameter &expand_param = param_.expand_param();
  const float expand_prob = expand_param.prob();
  float prob;
  caffe_rng_uniform(1, 0.f, 1.f, &prob);
  if (prob > expand_prob) {
    expanded_anno_datum->CopyFrom(anno_datum);
    return;
  }
  const float max_expand_ratio = expand_param.max_expand_ratio();
  if (fabs(max_expand_ratio - 1.) < 1e-2) {
    expanded_anno_datum->CopyFrom(anno_datum);
    return;
  }
  float expand_ratio;
  caffe_rng_uniform(1, 1.f, max_expand_ratio, &expand_ratio);
  // Expand the datum.
  NormalizedBBox expand_bbox;
  ExpandImage(anno_datum.datum(), expand_ratio, &expand_bbox,
              expanded_anno_datum->mutable_datum());
  expanded_anno_datum->set_type(anno_datum.type());

  // Transform the annotation according to crop_bbox.
  const bool do_resize = false;
  const bool do_mirror = false;
  TransformAnnotation(anno_datum, do_resize, expand_bbox, do_mirror,
                      expanded_anno_datum->mutable_annotation_group());
}

////////////////////////////////////////////////////////////////////////////////
#pragma endregion

template <typename Dtype>
void DataTransformer<Dtype>::CropImage(const Datum &datum,
                                       const NormalizedBBox &bbox,
                                       Datum *crop_datum) {
  // If datum is encoded, decode and crop the cv::image.
#ifdef USE_OPENCV
  cv::Mat cv_img, crop_img;

  if (datum.encoded()) {
    CHECK(!(param_.force_color() && param_.force_gray()))
        << "cannot set both force_color and force_gray";
    if (param_.force_color() || param_.force_gray()) {
      // If force_color then decode in color otherwise decode in gray.
      cv_img = DecodeDatumToCVMat(datum, param_.force_color());
    } else {
      cv_img = DecodeDatumToCVMatNative(datum);
    }
    // Crop the image.
    CropImage(cv_img, bbox, &crop_img);
    // Save the image into datum.
    EncodeCVMatToDatum(crop_img, "jpg", crop_datum);
  } else {
    //    if (param_.force_color() || param_.force_gray()) {
    //      LOG(ERROR) << "force_color and force_gray only for encoded datum";
    //    }
    cv_img = DatumToCVMat(datum);
    if (param_.force_gray() && cv_img.channels() == 3) {
      cv::cvtColor(cv_img, cv_img, cv::COLOR_BGR2GRAY);
    }
    if (param_.force_color() && cv_img.channels() == 1) {
      cv::cvtColor(cv_img, cv_img, cv::COLOR_GRAY2BGR);
    }
    CropImage(cv_img, bbox, &crop_img);
    CVMatToDatum(crop_img, crop_datum);
  }
  crop_datum->set_label(datum.label());
#else
  LOG(FATAL) << "Encoded datum requires OpenCV; compile with USE_OPENCV.";
#endif // USE_OPENCV
  //  const int datum_channels = datum.channels();
  //  const int datum_height = datum.height();
  //  const int datum_width = datum.width();
  //
  //  // Get the bbox dimension.
  //  NormalizedBBox clipped_bbox;
  //  ClipBBox(bbox, &clipped_bbox);
  //  NormalizedBBox scaled_bbox;
  //  ScaleBBox(clipped_bbox, datum_height, datum_width, &scaled_bbox);
  //  const int w_off = static_cast<int>(scaled_bbox.xmin());
  //  const int h_off = static_cast<int>(scaled_bbox.ymin());
  //  const int width = static_cast<int>(scaled_bbox.xmax() -
  //  scaled_bbox.xmin()); const int height =
  //  static_cast<int>(scaled_bbox.ymax() - scaled_bbox.ymin());
  //
  //  // Crop the image using bbox.
  //  crop_datum->set_channels(datum_channels);
  //  crop_datum->set_height(height);
  //  crop_datum->set_width(width);
  //  crop_datum->set_label(datum.label());
  //  crop_datum->clear_data();
  //  crop_datum->clear_float_data();
  //  crop_datum->set_encoded(false);
  //
  //  const int crop_datum_size = datum_channels * height * width;
  //  std::string buffer(crop_datum_size, ' ');
  //  vector<cv::Mat> crop_mats;
  //  for (int c = 0; c < datum_channels; ++c) {
  //    crop_mats.emplace_back(height, width, CV_8UC1,
  //                           &buffer[0] + c * height * width);
  //  }
  //
  //  cv::Mat datum_mat = DatumToCVMat(datum);
  //  datum_mat = datum_mat(cv::Rect(w_off, h_off, width, height));
  //
  //  cv::split(datum_mat, crop_mats);
  //
  //  crop_datum->set_data(buffer);
}

template <typename Dtype>
void DataTransformer<Dtype>::ExpandImage(const Datum &datum,
                                         const float expand_ratio,
                                         NormalizedBBox *expand_bbox,
                                         Datum *expand_datum) {
  // If datum is encoded, decode and crop the cv::image.
#ifdef USE_OPENCV
  cv::Mat cv_img, expand_img;
  if (datum.encoded()) {
    CHECK(!(param_.force_color() && param_.force_gray()))
        << "cannot set both force_color and force_gray";
    if (param_.force_color() || param_.force_gray()) {
      // If force_color then decode in color otherwise decode in gray.
      cv_img = DecodeDatumToCVMat(datum, param_.force_color());
    } else {
      cv_img = DecodeDatumToCVMatNative(datum);
    }
    // Expand the image.
    ExpandImage(cv_img, expand_ratio, expand_bbox, &expand_img);
    // Save the image into datum.
    EncodeCVMatToDatum(expand_img, "jpg", expand_datum);
  } else {
    //    if (param_.force_color() || param_.force_gray()) {
    //      LOG(ERROR) << "force_color and force_gray only for encoded datum";
    //    }
    cv_img = DatumToCVMat(datum);
    if (param_.force_gray() && cv_img.channels() == 3) {
      cv::cvtColor(cv_img, cv_img, cv::COLOR_BGR2GRAY);
    }
    if (param_.force_color() && cv_img.channels() == 1) {
      cv::cvtColor(cv_img, cv_img, cv::COLOR_GRAY2BGR);
    }
    ExpandImage(cv_img, expand_ratio, expand_bbox, &expand_img);
    CVMatToDatum(expand_img, expand_datum);
  }
  expand_datum->set_label(datum.label());
#else
  LOG(FATAL) << "Encoded datum requires OpenCV; compile with USE_OPENCV.";
#endif // USE_OPENCV
  //  const int datum_channels = datum.channels();
  //  const int datum_height = datum.height();
  //  const int datum_width = datum.width();
  //
  //  // Get the bbox dimension.
  //  int height = static_cast<int>(datum_height * expand_ratio);
  //  int width = static_cast<int>(datum_width * expand_ratio);
  //  float h_off, w_off;
  //  caffe_rng_uniform(1, 0.f, static_cast<float>(height - datum_height),
  //  &h_off); caffe_rng_uniform(1, 0.f, static_cast<float>(width -
  //  datum_width), &w_off); h_off = floor(h_off); w_off = floor(w_off);
  //  expand_bbox->set_xmin(-w_off / datum_width);
  //  expand_bbox->set_ymin(-h_off / datum_height);
  //  expand_bbox->set_xmax((width - w_off) / datum_width);
  //  expand_bbox->set_ymax((height - h_off) / datum_height);
  //
  //  // Crop the image using bbox.
  //  expand_datum->set_channels(datum_channels);
  //  expand_datum->set_height(height);
  //  expand_datum->set_width(width);
  //  expand_datum->set_label(datum.label());
  //  expand_datum->clear_data();
  //  expand_datum->clear_float_data();
  //  expand_datum->set_encoded(false);
  //
  //  const int expand_datum_size = datum_channels * height * width;
  //  std::string buffer(expand_datum_size, ' ');
  //  vector<cv::Mat> expand_mats;
  //  cv::Mat expand_mat;
  //  for (int c = 0; c < datum_channels; ++c) {
  //    expand_mats.emplace_back(height, width, CV_8UC1,
  //                             &buffer[0] + c * height * width);
  //  }
  //  cv::merge(expand_mats, expand_mat);
  //
  //  cv::Mat datum_mat = DatumToCVMat(datum);
  //  datum_mat.copyTo(
  //      expand_mat(cv::Rect(w_off, h_off, datum_width, datum_height)));
  //  cv::split(expand_mat, expand_mats);
  //  expand_datum->set_data(buffer);
}

template <typename Dtype>
void DataTransformer<Dtype>::DistortImage(const Datum &datum,
                                          Datum *distort_datum) {
  if (!param_.has_distort_param()) {
    distort_datum->CopyFrom(datum);
    return;
  }
#ifdef USE_OPENCV
  cv::Mat cv_img, distort_img;
  // If datum is encoded, decode and crop the cv::image.
  if (datum.encoded()) {
    CHECK(!(param_.force_color() && param_.force_gray()))
        << "cannot set both force_color and force_gray";
    if (param_.force_color() || param_.force_gray()) {
      // If force_color then decode in color otherwise decode in gray.
      cv_img = DecodeDatumToCVMat(datum, param_.force_color());
    } else {
      cv_img = DecodeDatumToCVMatNative(datum);
    }
    // Distort the image.
    distort_img = ApplyDistort(cv_img, param_.distort_param());
    // Save the image into datum.
    EncodeCVMatToDatum(distort_img, "jpg", distort_datum);
  } else {
    cv_img = DatumToCVMat(datum);
    distort_img = ApplyDistort(cv_img, param_.distort_param());
    CVMatToDatum(distort_img, distort_datum);
  }
  distort_datum->set_label(datum.label());
#else
  LOG(FATAL) << "Encoded datum requires OpenCV; compile with USE_OPENCV.";
#endif // USE_OPENCV
}

template <typename Dtype>
void DataTransformer<Dtype>::NoiseImage(const Datum &datum,
                                        Datum *noise_datum) {
  if (!param_.has_noise_param()) {
    noise_datum->CopyFrom(datum);
    return;
  }
#ifdef USE_OPENCV
  cv::Mat cv_img, noise_img;
  // If datum is encoded, decode and crop the cv::image.
  if (datum.encoded()) {
    CHECK(!(param_.force_color() && param_.force_gray()))
        << "cannot set both force_color and force_gray";
    if (param_.force_color() || param_.force_gray()) {
      // If force_color then decode in color otherwise decode in gray.
      cv_img = DecodeDatumToCVMat(datum, param_.force_color());
    } else {
      cv_img = DecodeDatumToCVMatNative(datum);
    }
    // Noise the image.
    noise_img = ApplyNoise(cv_img, param_.noise_param());
    // Save the image into datum.
    EncodeCVMatToDatum(noise_img, "jpg", noise_datum);
    noise_datum->set_label(datum.label());
    return;
  } else {
    cv_img = DatumToCVMat(datum);
    noise_img = ApplyNoise(cv_img, param_.noise_param());
    CVMatToDatum(noise_img, noise_datum);
    noise_datum->set_label(datum.label());
  }
#else
  LOG(FATAL) << "Encoded datum requires OpenCV; compile with USE_OPENCV.";
#endif // USE_OPENCV
}

#ifdef USE_OPENCV

template <typename Dtype>
void DataTransformer<Dtype>::TransformInv(const Dtype *data, cv::Mat *cv_img,
                                          const int height, const int width,
                                          const int channels) {
  const Dtype scale = param_.scale();
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = !mean_values_.empty();
  LOG(INFO) << "test";

  auto [type, ctype] = get_cv_type<Dtype>(channels);
  //  Dtype *mean = nullptr;
  if (has_mean_file) {
    //    CHECK_EQ(channels, data_mean_.channels());
    //    CHECK_EQ(height, data_mean_.height());
    //    CHECK_EQ(width, data_mean_.width());
    //    mean = data_mean_.mutable_cpu_data();
    CHECK_EQ(channels, cv_mean_->channels());
    CHECK_EQ(height, cv_mean_->rows);
    CHECK_EQ(width, cv_mean_->cols);
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == channels)
        << "Specify either 1 mean_value or as many as channels: " << channels;
    if (channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
    cv_mean_.reset(new cv::Mat(1, 1, ctype, mean_values_.data()));
  }

  const int img_type = channels == 3 ? CV_8UC3 : CV_8UC1;
  cv::Mat orig_img(height, width, img_type, cv::Scalar(0, 0, 0));

  cv::Mat transformed_mat;
  vector<cv::Mat> transformed_mats;
  transformed_mats.reserve(channels);
  for (int c = 0; c < channels; ++c) {
    transformed_mats.emplace_back(
        height, width, type, const_cast<Dtype *>(data + c * height * width));
  }
  cv::merge(transformed_mats, transformed_mat);

  transformed_mat /= scale;
  if (has_mean_file || has_mean_values) {
    if (transformed_mat.size != cv_mean_->size) {
      cv::resize(*cv_mean_, *cv_mean_, cv::Size(orig_img.cols, orig_img.rows));
    }
    transformed_mat += *cv_mean_;
  }
  transformed_mat.convertTo(orig_img, img_type);

  if (param_.resize_param_size()) {
    *cv_img = ApplyResize(orig_img, param_.resize_param(0));
  } else {
    *cv_img = orig_img;
  }
}

template <typename Dtype>
void DataTransformer<Dtype>::TransformInv(const Blob<Dtype> *blob,
                                          vector<cv::Mat> *cv_imgs) {
  const int channels = blob->channels();
  const int height = blob->height();
  const int width = blob->width();
  const int num = blob->num();
  CHECK_GE(num, 1);
  const Dtype *image_data = blob->cpu_data();

  for (int i = 0; i < num; ++i) {
    cv::Mat cv_img;
    TransformInv(image_data, &cv_img, height, width, channels);
    cv_imgs->push_back(cv_img);
    image_data += blob->offset(1);
  }
}

template <typename Dtype>
void DataTransformer<Dtype>::CropImage(const cv::Mat &img,
                                       const NormalizedBBox &bbox,
                                       cv::Mat *crop_img) {
  const int img_height = img.rows;
  const int img_width = img.cols;

  // Get the bbox dimension.
  NormalizedBBox clipped_bbox;
  ClipBBox(bbox, &clipped_bbox);
  NormalizedBBox scaled_bbox;
  ScaleBBox(clipped_bbox, img_height, img_width, &scaled_bbox);

  // Crop the image using bbox.
  int w_off = static_cast<int>(scaled_bbox.xmin());
  int h_off = static_cast<int>(scaled_bbox.ymin());
  int width = static_cast<int>(scaled_bbox.xmax() - scaled_bbox.xmin());
  int height = static_cast<int>(scaled_bbox.ymax() - scaled_bbox.ymin());
  cv::Rect bbox_roi(w_off, h_off, width, height);

  img(bbox_roi).copyTo(*crop_img);
}

template <typename Dtype>
void DataTransformer<Dtype>::ExpandImage(const cv::Mat &img,
                                         const float expand_ratio,
                                         NormalizedBBox *expand_bbox,
                                         cv::Mat *expand_img) {
  const int img_height = img.rows;
  const int img_width = img.cols;
  const int img_channels = img.channels();
  // LOG(INFO)<<img_width <<","<<img_height;
  // Get the bbox dimension.
  int height = static_cast<int>(img_height * expand_ratio);
  int width = static_cast<int>(img_width * expand_ratio);
  float h_off, w_off;
  caffe_rng_uniform(1, 0.f, static_cast<float>(height - img_height), &h_off);
  caffe_rng_uniform(1, 0.f, static_cast<float>(width - img_width), &w_off);
  h_off = floor(h_off);
  w_off = floor(w_off);
  expand_bbox->set_xmin(-w_off / img_width);
  expand_bbox->set_ymin(-h_off / img_height);
  expand_bbox->set_xmax((width - w_off) / img_width);
  expand_bbox->set_ymax((height - h_off) / img_height);

  expand_img->create(height, width, img.type());
  expand_img->setTo(cv::Scalar(0));
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = !mean_values_.empty();

  auto [type, ctype] = get_cv_type<Dtype>(img_channels);
  if (has_mean_file) {
    //    CHECK_EQ(img_channels, data_mean_.channels());
    //    CHECK_EQ(height, data_mean_.height());
    //    CHECK_EQ(width, data_mean_.width());
    CHECK_EQ(img_channels, cv_mean_->channels());
    CHECK_EQ(height, cv_mean_->rows);
    CHECK_EQ(width, cv_mean_->cols);
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == img_channels)
        << "Specify either 1 mean_value or as many as channels: "
        << img_channels;
    if (img_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < img_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
      cv_mean_.reset(new cv::Mat(1, 1, ctype, mean_values_.data()));
    }
  }
  cv_mean_->convertTo(*expand_img, ctype);
  cv::Rect bbox_roi(w_off, h_off, img_width, img_height);
  img.copyTo((*expand_img)(bbox_roi));
}

#endif // USE_OPENCV

template <typename Dtype>
void DataTransformer<Dtype>::Transform(Blob<Dtype> *input_blob,
                                       Blob<Dtype> *transformed_blob) {
  const int crop_size = param_.crop_size();
  const int input_num = input_blob->num();
  const int input_channels = input_blob->channels();
  const int input_height = input_blob->height();
  const int input_width = input_blob->width();

  if (transformed_blob->count() == 0) {
    // Initialize transformed_blob with the right shape.
    if (crop_size) {
      transformed_blob->Reshape(input_num, input_channels, crop_size,
                                crop_size);
    } else {
      transformed_blob->Reshape(input_num, input_channels, input_height,
                                input_width);
    }
  }

  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int size = transformed_blob->count();
  LOG(INFO) << width << "," << height;
  CHECK_LE(input_num, num);
  CHECK_EQ(input_channels, channels);
  CHECK_GE(input_height, height);
  CHECK_GE(input_width, width);

  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = !mean_values_.empty();

  int h_off = 0;
  int w_off = 0;
  if (crop_size) {
    CHECK_EQ(crop_size, height);
    CHECK_EQ(crop_size, width);
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = Rand(input_height - crop_size + 1);
      w_off = Rand(input_width - crop_size + 1);
    } else {
      h_off = (input_height - crop_size) / 2;
      w_off = (input_width - crop_size) / 2;
    }
  } else {
    CHECK_EQ(input_height, height);
    CHECK_EQ(input_width, width);
  }

  Dtype *input_data = input_blob->mutable_cpu_data();
  if (has_mean_file) {
    CHECK_EQ(input_channels, data_mean_.channels());
    CHECK_EQ(input_height, data_mean_.height());
    CHECK_EQ(input_width, data_mean_.width());
    for (int n = 0; n < input_num; ++n) {
      int offset = input_blob->offset(n);
      caffe_sub(data_mean_.count(), input_data + offset, data_mean_.cpu_data(),
                input_data + offset);
    }
  }

  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == input_channels)
        << "Specify either 1 mean_value or as many as channels: "
        << input_channels;
    if (mean_values_.size() == 1) {
      caffe_add_scalar(input_blob->count(), -(mean_values_[0]), input_data);
    } else {
      for (int n = 0; n < input_num; ++n) {
        for (int c = 0; c < input_channels; ++c) {
          int offset = input_blob->offset(n, c);
          caffe_sub_scalar(input_height * input_width, mean_values_[c],
                           input_data + offset);
        }
      }
    }
  }

  Dtype *transformed_data = transformed_blob->mutable_cpu_data();

  for (int n = 0; n < input_num; ++n) {
    int top_index_n = n * channels;
    int data_index_n = n * channels;
    for (int c = 0; c < channels; ++c) {
      int top_index_c = (top_index_n + c) * height;
      int data_index_c = (data_index_n + c) * input_height + h_off;
      for (int h = 0; h < height; ++h) {
        int top_index_h = (top_index_c + h) * width;
        int data_index_h = (data_index_c + h) * input_width + w_off;
        if (do_mirror) {
          int top_index_w = top_index_h + width - 1;
          for (int w = 0; w < width; ++w) {
            transformed_data[top_index_w - w] = input_data[data_index_h + w];
          }
        } else {
          for (int w = 0; w < width; ++w) {
            transformed_data[top_index_h + w] = input_data[data_index_h + w];
          }
        }
      }
    }
  }
  if (scale != Dtype(1)) {
    DLOG(INFO) << "Scale: " << scale;
    caffe_blas_scal(size, scale, transformed_data);
  }
}

template <typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(const Datum &datum,
                                                   int policy_num) {
  if (datum.encoded()) {
#ifdef USE_OPENCV
    CHECK(!(param_.force_color() && param_.force_gray()))
        << "cannot set both force_color and force_gray";
    cv::Mat cv_img;
    if (param_.force_color() || param_.force_gray()) {
      // If force_color then decode in color otherwise decode in gray.
      cv_img = DecodeDatumToCVMat(datum, param_.force_color());
    } else {
      cv_img = DecodeDatumToCVMatNative(datum);
    }
    // InferBlobShape using the cv::image.
    return InferBlobShape(cv_img, policy_num);
#else
    LOG(FATAL) << "Encoded datum requires OpenCV; compile with USE_OPENCV.";
#endif // USE_OPENCV
  }

  const int crop_size = param_.crop_size();
  int crop_h = param_.crop_h();
  int crop_w = param_.crop_w();
  if (crop_size) {
    crop_h = crop_size;
    crop_w = crop_size;
  }
  const int datum_channels = datum.channels();
  int datum_height = datum.height();
  int datum_width = datum.width();

  // Check dimensions.
  CHECK_GT(datum_channels, 0);
  if (param_.resize_param_size()) {
    InferNewSize(param_.resize_param(policy_num), datum_width, datum_height,
                 &datum_width, &datum_height);
  }
  CHECK_GE(datum_height, crop_h);
  CHECK_GE(datum_width, crop_w);

  // Build BlobShape.
  vector<int> shape(4);
  shape[0] = 1;
  shape[1] = datum_channels;
  shape[2] = (crop_h) ? crop_h : datum_height;
  shape[3] = (crop_w) ? crop_w : datum_width;
  return shape;
}

template <typename Dtype>
vector<int>
DataTransformer<Dtype>::InferBlobShape(const vector<Datum> &datum_vector) {
  const int num = datum_vector.size();
  CHECK_GT(num, 0) << "There is no datum to in the vector";
  // Use first datum in the vector to InferBlobShape.
  vector<int> shape = InferBlobShape(datum_vector[0]);
  // Adjust num to the size of the vector.
  shape[0] = num;
  return shape;
}

#ifdef USE_OPENCV
template <typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(const cv::Mat &cv_img,
                                                   int policy_num) {
  const int crop_size = param_.crop_size();
  int crop_h = param_.crop_h();
  int crop_w = param_.crop_w();
  if (crop_size) {
    crop_h = crop_size;
    crop_w = crop_size;
  }
  const int img_channels = cv_img.channels();
  int img_height = cv_img.rows;
  int img_width = cv_img.cols;
  // Check dimensions.
  CHECK_GT(img_channels, 0);
  // LOG(INFO) << policy_num;
  if (param_.resize_param_size()) {
    InferNewSize(param_.resize_param(policy_num), img_width, img_height,
                 &img_width, &img_height);
  }
  CHECK_GE(img_height, crop_h);
  CHECK_GE(img_width, crop_w);

  // Build BlobShape.
  vector<int> shape(4);
  shape[0] = 1;
  shape[1] = img_channels;
  shape[2] = (crop_h) ? crop_h : img_height;
  shape[3] = (crop_w) ? crop_w : img_width;
  return shape;
}

template <typename Dtype>
vector<int>
DataTransformer<Dtype>::InferBlobShape(const vector<cv::Mat> &mat_vector) {
  const int num = mat_vector.size();
  CHECK_GT(num, 0) << "There is no cv_img to in the vector";
  // Use first cv_img in the vector to InferBlobShape.
  vector<int> shape = InferBlobShape(mat_vector[0]);
  // Adjust num to the size of the vector.
  shape[0] = num;
  return shape;
}
#endif // USE_OPENCV

template <typename Dtype>
void DataTransformer<Dtype>::InitRand() {
  const bool needs_rand =
      param_.mirror() || (phase_ == TRAIN && param_.crop_size());
  if (needs_rand) {
    const unsigned int rng_seed = caffe_rng_rand();
    rng_.reset(new Caffe::RNG(rng_seed));
  } else {
    rng_.reset();
  }
}

template <typename Dtype>
int DataTransformer<Dtype>::Rand(int n) {
  CHECK(rng_);
  CHECK_GT(n, 0);
  auto *rng = static_cast<caffe::rng_t *>(rng_->generator());
  return ((*rng)() % n);
}

INSTANTIATE_CLASS(DataTransformer);

} // namespace caffe
