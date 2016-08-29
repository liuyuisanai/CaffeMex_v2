// This file modifies the ROI pooling layer of Fast R-CNN such that it implements the 
// the rectangular rings regions that were introduced on:
// "Object detection via a multi-region & semantic segmentation-aware CNN model"
// Those rectangular ring regions are defined in terms of an inner rectangle and 
// and an outer rectangle. During the region pooling operation, both the inner and 
// the outer rectangles are projected on the activation maps and the activations 
// that lay inside the inner rectangle are ignored during the adaptive max
// pooling operation.
// 
// AUTORIGHTS
// --------------------------------------------------------
// Copyright (c) 2015 Spyros Gidaris
//
// "Object detection via a multi-region & semantic segmentation-aware CNN model"
// Technical report: http://arxiv.org/abs/1505.01749
// ---------------------------------------------------------
// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// ------------------------------------------------------------------

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/psroi_region_pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {

template <typename Dtype>
void PSROIRegionPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  PSROIRegionPoolingParameter pool_param = this->layer_param_.psroi_region_pooling_param();
  CHECK_GT(pool_param.pooled_h(), 0)
      << "pooled_h must be > 0";
  CHECK_GT(pool_param.pooled_w(), 0)
      << "pooled_w must be > 0";
  pooled_height_ = pool_param.pooled_h();
  pooled_width_  = pool_param.pooled_w();
  spatial_scale_ = pool_param.spatial_scale();
  LOG(INFO) << "Spatial scale: " << spatial_scale_;
  offset_ 	 = pool_param.offset();
  extend_scale_ = pool_param.extend_scale();
  shrink_scale_ = pool_param.shrink_scale();
  CHECK_GT(extend_scale_, shrink_scale_);
  CHECK_GE(shrink_scale_, 0);
  output_dim_ = pool_param.output_dim();
  group_size_ = pool_param.group_size();
}

template <typename Dtype>
void PSROIRegionPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  channels_ = bottom[0]->channels();
  CHECK_EQ(channels_, output_dim_*group_size_*group_size_)
      << "input channel number does not match layer parameters";
  height_   = bottom[0]->height();
  width_    = bottom[0]->width();
  top[0]->Reshape(bottom[1]->num(), output_dim_, pooled_height_,  pooled_width_);
  mapping_channel_.Reshape(bottom[1]->num(), output_dim_, pooled_height_, pooled_width_);
}

template <typename Dtype>
void PSROIRegionPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    NOT_IMPLEMENTED;
}

template <typename Dtype>
void PSROIRegionPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(PSROIRegionPoolingLayer);
#endif

INSTANTIATE_CLASS(PSROIRegionPoolingLayer);
REGISTER_LAYER_CLASS(PSROIRegionPooling);

}  // namespace caffe
