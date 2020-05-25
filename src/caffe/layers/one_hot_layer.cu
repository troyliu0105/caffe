//
// Created by troyl on 5/24/2020.
//

#include "caffe/layers/one_hot_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void kernel_one_hot(int outer_num, int inner_num, int dim,
                               const Dtype *bottom_data, Dtype *top_data) {
  CUDA_KERNEL_LOOP(index, outer_num * inner_num) {
    int i = index / inner_num;
    int j = index % inner_num;
    int label = static_cast<int>(bottom_data[i * inner_num + j]);
    top_data[i * dim + label * inner_num + j] = 1;
  }
}

template <typename Dtype>
void OneHotLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                                     const vector<Blob<Dtype> *> &top) {
  const int nthreads = outer_num_ * inner_num_;
  const int dim = top[0]->count() / outer_num_;
  const Dtype *bottom_data = bottom[0]->gpu_data();
  Dtype *top_data = top[0]->mutable_gpu_data();
  caffe_gpu_set(top[0]->count(), Dtype(0), top_data);
  kernel_one_hot<<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
      outer_num_, inner_num_, dim, bottom_data, top_data);
}

template <typename Dtype>
void OneHotLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype> *> &top,
                                      const vector<bool> &propagate_down,
                                      const vector<Blob<Dtype> *> &bottom) {
  if (propagate_down[0]) {
    NOT_IMPLEMENTED;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(OneHotLayer);

} // namespace caffe