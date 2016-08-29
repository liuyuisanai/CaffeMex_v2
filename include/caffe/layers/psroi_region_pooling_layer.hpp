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

#ifndef CAFFE_PSROIREGION_POOLING_LAYER_HPP_
#define CAFFE_PSROIREGION_POOLING_LAYER_HPP_

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {


template <typename Dtype>
class PSROIRegionPoolingLayer : public Layer<Dtype> {
 public:
  explicit PSROIRegionPoolingLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "PSROIRegionPooling"; }

  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int MaxBottomBlobs() const { return 2; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int channels_;
  int height_;
  int width_;
  int pooled_height_;
  int pooled_width_;
  int output_dim_;
  int group_size_;
  Dtype extend_scale_;
  Dtype shrink_scale_;
  Dtype spatial_scale_;
  Dtype offset_;
  
  Blob<int> mapping_channel_;
};

}  // namespace caffe

#endif  // CAFFE_PSROIREGION_POOLING_LAYER_HPP_
