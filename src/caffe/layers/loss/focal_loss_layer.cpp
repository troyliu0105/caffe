#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/loss/focal_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void FocalLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                       const vector<Blob<Dtype> *> &top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  LayerParameter activate_param(this->layer_param_);
  activate_type_ = this->layer_param_.focal_loss_param().activate();
  if (activate_type_ == FocalLossParameter_ActivateType_SOFTMAX) {
    activate_param.set_type("Softmax");
  } else {
    activate_param.set_type("Sigmoid");
  }
  activate_layer_ = LayerRegistry<Dtype>::CreateLayer(activate_param);
  activate_bottom_vec_.clear();
  activate_bottom_vec_.push_back(bottom[0]);
  activate_top_vec_.clear();
  activate_top_vec_.push_back(&prob_);
  activate_layer_->SetUp(activate_bottom_vec_, activate_top_vec_);

  alpha_ = this->layer_param_.focal_loss_param().alpha();
  gamma_ = this->layer_param_.focal_loss_param().gamma();
  has_ignore_label_ = this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  if (!this->layer_param_.loss_param().has_normalization() &&
      this->layer_param_.loss_param().has_normalize()) {
    normalization_ = this->layer_param_.loss_param().normalize()
                         ? LossParameter_NormalizationMode_VALID
                         : LossParameter_NormalizationMode_BATCH_SIZE;
  } else {
    normalization_ = this->layer_param_.loss_param().normalization();
  }
}

template <typename Dtype>
void FocalLossLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                    const vector<Blob<Dtype> *> &top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  activate_layer_->Reshape(activate_bottom_vec_, activate_top_vec_);
  if (activate_type_ == FocalLossParameter_ActivateType_SOFTMAX) {
    activate_axis_ = bottom[0]->CanonicalAxisIndex(
        this->layer_param_.focal_loss_param().axis());

    outer_num_ = bottom[0]->count(0, activate_axis_);  // n
    inner_num_ = bottom[0]->count(activate_axis_ + 1); // h * w
    CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
        << "Number of labels must match number of predictions; "
        << "e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), "
        << "label count (number of labels) must be N*H*W, "
        << "with integer values in {0, 1, ..., C-1}.";
  } else {
    outer_num_ = bottom[0]->shape(0);
    inner_num_ = bottom[0]->count(1);
    CHECK_EQ(bottom[0]->count(), bottom[1]->count());
    scale_.ReshapeLike(*bottom[0]);
  }
  if (top.size() >= 2) {
    // softmax output
    top[1]->ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
void FocalLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                        const vector<Blob<Dtype> *> &top) {
  // The forward pass computes the softmax prob values.
  activate_layer_->Forward(activate_bottom_vec_, activate_top_vec_);
  const int count = bottom[0]->count();
  const Dtype *input_data = bottom[0]->cpu_data();
  const Dtype *prob_data = prob_.cpu_data();
  const Dtype *label = bottom[1]->cpu_data();
  Dtype loss = 0;
  int valid_num = 0;
  if (activate_type_ == FocalLossParameter_ActivateType_SOFTMAX) {
    int dim = prob_.count() / outer_num_;
    Dtype pt = 0;
    for (int i = 0; i < outer_num_; ++i) {
      for (int j = 0; j < inner_num_; j++) {
        const int label_value = static_cast<int>(label[i * inner_num_ + j]);
        if (has_ignore_label_ && label_value == ignore_label_) {
          continue;
        }
        DCHECK_GE(label_value, 0);
        DCHECK_LT(label_value, prob_.shape(activate_axis_));
        // loss -= log(std::max(prob_data[i * dim + label_value * inner_num_ +
        // j],
        //                     Dtype(FLT_MIN)));
        pt = prob_data[i * dim + label_value * inner_num_ + j];
        loss -=
            alpha_ * pow(1.0 - pt, gamma_) * log(std::max(pt, Dtype(FLT_MIN)));
        ++valid_num;
      }
    }
  } else {
    Dtype *scale = scale_.mutable_cpu_data();
    Dtype *oriloss = scale_.mutable_cpu_diff();
    Dtype target;
    for (size_t i = 0; i < count; i++) {
      target = label[i];
      if (has_ignore_label_ && target == ignore_label_) {
        scale[i] = 0;
        oriloss[i] = 0;
        continue;
      } else {
        valid_num += 1;
        scale[i] =
            (target == 1 ? alpha_ : 1 - alpha_) *
            pow(1 - (target == 1 ? prob_data[i] : (1 - prob_data[i])), gamma_);
        oriloss[i] = -(input_data[i] * (label[i] - (input_data[i] >= 0)) -
                       log(1 + exp(input_data[i] -
                                   2 * input_data[i] * (input_data[i] >= 0))));
        loss += scale[i] * oriloss[i];
      }
    }
  }
  Dtype normalizer = LossLayer<Dtype>::GetNormalizer(normalization_, outer_num_,
                                                     inner_num_, valid_num);
  top[0]->mutable_cpu_data()[0] = loss / normalizer;
  normalizer_ = normalizer;
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
void FocalLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *> &top,
                                         const vector<bool> &propagate_down,
                                         const vector<Blob<Dtype> *> &bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype *bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype *input_data = bottom[0]->cpu_data();
    const int count = bottom[0]->count();
    const Dtype *prob_data = prob_.cpu_data();
    // caffe_copy(prob_.count(), prob_data, bottom_diff);
    const Dtype *label = bottom[1]->cpu_data();
    if (activate_type_ == FocalLossParameter_ActivateType_SOFTMAX) {
      int dim = prob_.count() / outer_num_;
      int valid_num = 0;
      Dtype focal_diff = 0;
      Dtype pt = 0;
      Dtype pc = 0;
      for (int i = 0; i < outer_num_; ++i) {
        for (int j = 0; j < inner_num_; ++j) {
          const int label_value = static_cast<int>(label[i * inner_num_ + j]);
          if (has_ignore_label_ && label_value == ignore_label_) {
            for (int c = 0; c < bottom[0]->shape(activate_axis_); ++c) {
              bottom_diff[i * dim + c * inner_num_ + j] = 0;
            }
          } else {
            // bottom_diff[i * dim + label_value * inner_num_ + j] -= 1;
            pt = prob_data[i * dim + label_value * inner_num_ + j];
            for (int c = 0; c < bottom[0]->shape(activate_axis_); ++c) {
              pc = prob_data[i * dim + c * inner_num_ + j];
              if (c == label_value) {
                focal_diff =
                    alpha_ * pow(1 - pt, gamma_) *
                    (gamma_ * pt * log(std::max(pt, Dtype(FLT_MIN))) + pt - 1);
              } else {
                focal_diff =
                    alpha_ * (pow(1 - pt, gamma_ - 1) *
                                  (-gamma_ * log(std::max(pt, Dtype(FLT_MIN))) *
                                   pt * pc) +
                              pow(1 - pt, gamma_) * pc);
              }
              bottom_diff[i * dim + c * inner_num_ + j] = focal_diff;
            }
            ++valid_num;
          }
        }
      }
    } else {
      //---------- d(oriloss) * scale
      caffe_copy(count, prob_data, bottom_diff);
      // sigmoid(x) - y
      caffe_blas_axpy(count, Dtype(-1), label, bottom_diff);
      caffe_mul(count, scale_.cpu_data(), bottom[0]->cpu_diff(), bottom_diff);

      //---------- oriloss * d(scale)
      // save result in scaler_.data
      Dtype *scale_diff = scale_.mutable_cpu_diff(); // origin loss
      Dtype *scale_data = scale_.mutable_cpu_data(); // scale, but useless now
      FOR_LOOP_WITH_PREPARE(
          count, i,
          {
            target = label[i];
            expabsx = exp(input_data[i] > 0 ? -input_data[i] : input_data[i]);
            scale_data[i] =
                (target == 1 ? alpha_ : 1 - alpha_) * gamma_ *
                pow(1 - (target == 1 ? prob_data[i] : (1 - prob_data[i])),
                    gamma_ - 1) *
                expabsx / (pow(expabsx, 2) + 2 * expabsx + 1) *
                (target == 1 ? -1 : 1);
          },
          Dtype expabsx;
          Dtype target)
      caffe_mul(count, scale_data, scale_diff, scale_data);
      caffe_add(count, scale_data, bottom_diff, bottom_diff);
      if (has_ignore_label_) {
        FOR_LOOP(count, i, {
          if (label[i] == ignore_label_) {
            bottom_diff[i] = 0;
          }
        })
      }
    }
    // Scale gradient
    Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer_;
    caffe_blas_scal(count, loss_weight, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(FocalLossLayer);
#endif

INSTANTIATE_CLASS(FocalLossLayer);
REGISTER_LAYER_CLASS(FocalLoss);

} // namespace caffe