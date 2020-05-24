#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/focal_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void FocalLossSoftmaxForwardGPU(
    const int nthreads, const Dtype *prob_data, const Dtype *label, Dtype *loss,
    const int num, const int dim, const int spatial_dim,
    const bool has_ignore_label_, const int ignore_label_, Dtype *counts,
    float alpha_, float gamma_) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);
    if (has_ignore_label_ && label_value == ignore_label_) {
      loss[index] = 0;
      counts[index] = 0;
    } else {
      // loss[index] = -log(max(prob_data[n * dim + label_value * spatial_dim +
      // s],
      //                Dtype(FLT_MIN)));
      Dtype pt = prob_data[n * dim + label_value * spatial_dim + s];
      loss[index] =
          -alpha_ * powf(1 - pt, gamma_) * log(max(pt, Dtype(FLT_MIN)));
      counts[index] = 1;
    }
  }
}

template <typename Dtype>
__global__ void FocalLossSigmoidForwardGPU(
    const int nthreads, const Dtype *input_data, const Dtype *sigmoid_data,
    const Dtype *target, Dtype *scale, Dtype *oriloss,
    const bool has_ignore_label_, const int ignore_label_, Dtype *counts,
    float alpha, float gamma) {
  CUDA_KERNEL_LOOP(i, nthreads) {
    const int target_value = static_cast<int>(target[i]);
    if (has_ignore_label_ && target_value == ignore_label_) {
      scale[i] = 0;
      oriloss[i] = 0;
      counts[i] = 0;
    } else {
      scale[i] = (target_value == 1 ? alpha : 1 - alpha) *
                 powf(1 - (target_value == 1 ? sigmoid_data[i]
                                             : (1 - sigmoid_data[i])),
                      gamma);
      oriloss[i] = -(input_data[i] * (target[i] - (input_data[i] >= 0)) -
                     log(1 + exp(input_data[i] -
                                 2 * input_data[i] * (input_data[i] >= 0))));
      counts[i] = 1;
    }
  }
}

template <typename Dtype>
void FocalLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                                        const vector<Blob<Dtype> *> &top) {
  activate_layer_->Forward(activate_bottom_vec_, activate_top_vec_);
  const Dtype *prob_data = prob_.gpu_data();
  const Dtype *label = bottom[1]->gpu_data();
  const int dim = prob_.count() / outer_num_;
  const int nthreads = outer_num_ * inner_num_;
  // Since this memory is not used for anything until it is overwritten
  // on the backward pass, we use it here to avoid having to allocate new GPU
  // memory to accumulate intermediate results in the kernel.
  Dtype *loss_data = bottom[0]->mutable_gpu_diff();
  // Similarly, this memory is never used elsewhere, and thus we can use it
  // to avoid having to allocate additional GPU memory.
  Dtype *counts = prob_.mutable_gpu_diff();
  if (activate_type_ == FocalLossParameter_ActivateType_SOFTMAX) {
    // NOLINT_NEXT_LINE(whitespace/operators)
    FocalLossSoftmaxForwardGPU<Dtype>
        <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
            nthreads, prob_data, label, loss_data, outer_num_, dim, inner_num_,
            has_ignore_label_, ignore_label_, counts, alpha_, gamma_);
  } else {
    FocalLossSigmoidForwardGPU<Dtype>
        <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
            nthreads, bottom[0]->gpu_data(), prob_data, label,
            scale_.mutable_gpu_data(), scale_.mutable_gpu_diff(),
            has_ignore_label_, ignore_label_, counts, alpha_, gamma_);
    caffe_gpu_mul(nthreads, scale_.gpu_data(), scale_.gpu_diff(), loss_data);
  }

  Dtype loss;
  caffe_gpu_asum(nthreads, loss_data, &loss);
  Dtype valid_count = -1;
  // Only launch another CUDA kernel if we actually need the count of valid
  // outputs.
  if (normalization_ == LossParameter_NormalizationMode_VALID &&
      has_ignore_label_) {
    caffe_gpu_asum(nthreads, counts, &valid_count);
  } else {
    valid_count = nthreads;
  }
  Dtype normalizer = LossLayer<Dtype>::GetNormalizer(normalization_, outer_num_,
                                                     inner_num_, valid_count);
  top[0]->mutable_cpu_data()[0] = loss / normalizer;
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
__global__ void FocalLossSoftmaxBackwardGPU(
    const int nthreads, const Dtype *top, const Dtype *label,
    Dtype *bottom_diff, const int num, const int dim, const int spatial_dim,
    const bool has_ignore_label_, const int ignore_label_, Dtype *counts,
    float alpha_, float gamma_) {
  const int channels = dim / spatial_dim;

  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);

    if (has_ignore_label_ && label_value == ignore_label_) {
      for (int c = 0; c < channels; ++c) {
        bottom_diff[n * dim + c * spatial_dim + s] = 0;
      }
      counts[index] = 0;
    } else {
      Dtype pt = bottom_diff[n * dim + label_value * spatial_dim + s];
      for (int c = 0; c < channels; ++c) {
        if (c == label_value) {
          bottom_diff[n * dim + c * spatial_dim + s] =
              alpha_ * powf(1 - pt, gamma_) *
              (gamma_ * pt * log(max(pt, Dtype(FLT_MIN))) + pt - 1);
        } else {
          Dtype pc = bottom_diff[n * dim + c * spatial_dim + s];
          bottom_diff[n * dim + c * spatial_dim + s] =
              alpha_ * (powf(1 - pt, gamma_ - 1) *
                            (-gamma_ * log(max(pt, Dtype(FLT_MIN))) * pt * pc) +
                        powf(1 - pt, gamma_) * pc);
        }
      }
      counts[index] = 1;
    }
  }
}

template <typename Dtype>
__global__ void FocalLossSigmoidBackwardSecondItemGPU(
    const int nthreads, const Dtype *input_data, const Dtype *sigmoid_data,
    const Dtype *target, float alpha, float gamma, Dtype *secondItem) {
  CUDA_KERNEL_LOOP(i, nthreads) {
    const int target_value = static_cast<int>(target[i]);
    {
      Dtype expabsx = expf(input_data[i] > 0 ? -input_data[i] : input_data[i]);
      secondItem[i] = (target_value == 1 ? alpha : 1 - alpha) * gamma *
                      powf(1 - (target_value == 1 ? sigmoid_data[i]
                                                  : (1 - sigmoid_data[i])),
                           gamma - 1) *
                      expabsx / (powf(expabsx, 2) + 2 * expabsx + 1) *
                      (target_value == 1 ? -1 : 1);
    }
  }
}

template <typename Dtype>
__global__ void
FocalLossSigmoidIgnoreDiffGPU(const int count, const int ignore_label,
                              const Dtype *target, Dtype *diff) {
  CUDA_KERNEL_LOOP(i, count) {
    const int target_value = static_cast<int>(target[i]);
    if (target_value == ignore_label) {
      diff[i] = 0;
    }
  }
}

template <typename Dtype>
void FocalLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype> *> &top,
                                         const vector<bool> &propagate_down,
                                         const vector<Blob<Dtype> *> &bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype *bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype *prob_data = prob_.gpu_data();
    const Dtype *top_data = top[0]->gpu_data();
    caffe_gpu_memcpy(prob_.count() * sizeof(Dtype), prob_data, bottom_diff);
    const Dtype *label = bottom[1]->gpu_data();
    const int dim = prob_.count() / outer_num_;
    const int nthreads = outer_num_ * inner_num_;
    // Since this memory is never used for anything else,
    // we use to to avoid allocating new GPU memory.
    Dtype *counts = prob_.mutable_gpu_diff();
    if (activate_type_ == FocalLossParameter_ActivateType_SOFTMAX) {
      // NOLINT_NEXT_LINE(whitespace/operators)
      FocalLossSoftmaxBackwardGPU<Dtype>
          <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
              nthreads, top_data, label, bottom_diff, outer_num_, dim,
              inner_num_, has_ignore_label_, ignore_label_, counts, alpha_,
              gamma_);
    } else {
      // First item: d(oriloss)*scale
      caffe_copy(nthreads, prob_data, bottom_diff);
      caffe_gpu_axpy(nthreads, Dtype(-1), label, bottom_diff);
      caffe_gpu_mul(nthreads, scale_.gpu_data(), bottom[0]->gpu_diff(),
                    bottom_diff);
      // Second item: oriloss*d(scale)
      // save result in scaler_.data
      FocalLossSigmoidBackwardSecondItemGPU<Dtype>
          <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
              nthreads, bottom[0]->gpu_data(), prob_data, label, alpha_, gamma_,
              scale_.mutable_gpu_data());
      caffe_gpu_mul(nthreads, scale_.gpu_data(), scale_.gpu_diff(),
                    scale_.mutable_gpu_data());
      caffe_gpu_add(nthreads, scale_.gpu_data(), bottom[0]->gpu_diff(),
                    bottom_diff);
      if (has_ignore_label_) {
        FocalLossSigmoidIgnoreDiffGPU<Dtype>
            <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
                nthreads, ignore_label_, label, bottom_diff);
      }
    }
    Dtype valid_count = -1;
    // Only launch another CUDA kernel if we actually need the count of valid
    // outputs.
    if (normalization_ == LossParameter_NormalizationMode_VALID &&
        has_ignore_label_) {
      caffe_gpu_asum(nthreads, counts, &valid_count);
    }
    const Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer_;
    caffe_gpu_scal(prob_.count(), loss_weight, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(FocalLossLayer);

} // namespace caffe