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
#include "caffe/util/gpu_util.cuh"
#include "caffe/util/math_functions.hpp"

using std::max;
using std::min;

namespace caffe {

template <typename Dtype>
__global__ void PSROIRegionPoolForward(const int nthreads, const Dtype* bottom_data,
    const Dtype spatial_scale, const Dtype offset, 
    const int channels, const int height, const int width, 
    const int pooled_height, const int pooled_width,
    const Dtype* bottom_rois, Dtype* top_data, 
    const int output_dim, const int group_size, 
    int* mapping_channel, 
    Dtype extend_scale, Dtype shrink_scale) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int ctop = (index / pooled_width / pooled_height) % output_dim;
    // int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / output_dim;

  // For each ROI R = [batch_index, x_outer_1, y_outer_1, x_outer_2, y_outer_2, x_inner_1, y_inner_1, x_inner_2, y_inner_2]: 
  // where R_outer = [x_outer_1, y_outer_1, x_outer_2, y_outer_2] is the outer rectangle of the region and 
  // R_inner = [x_inner_1, y_inner_1, x_inner_2, y_inner_2] is the inner rectangle of the region
  // max pooler over R by ignoring (setting to zero) the activations that lay inside the inner rectangle R_inner

    bottom_rois += n * 5;
    int roi_batch_ind = bottom_rois[0];
    Dtype center_x = (bottom_rois[1] + bottom_rois[3]) / 2.;
    Dtype center_y = (bottom_rois[2] + bottom_rois[4]) / 2.;
    Dtype extend_w = (bottom_rois[3] - bottom_rois[1] + 1) * extend_scale;
    Dtype extend_h = (bottom_rois[4] - bottom_rois[2] + 1) * extend_scale;
    Dtype shrink_w = (bottom_rois[3] - bottom_rois[1] + 1) * shrink_scale;
    Dtype shrink_h = (bottom_rois[4] - bottom_rois[2] + 1) * shrink_scale;
    
    // outer rectangle of the region
    int roi_start_w   = min(max(0, (int)floor((center_x - extend_w / 2. + 1 + offset) * spatial_scale + 0.5)), width);
    int roi_start_h   = min(max(0, (int)floor((center_y - extend_h / 2. + 1 + offset) * spatial_scale + 0.5)), height);
    int roi_end_w     = min(max(0, (int)ceil( (center_x + extend_w / 2. + 1 - offset) * spatial_scale - 0.5)), width);
    int roi_end_h     = min(max(0, (int)ceil( (center_y + extend_h / 2. + 1 - offset) * spatial_scale - 0.5)), height);

    // inner rectangle of the region
    int roi_start_w_in = min(max(0, (int)floor((center_x - shrink_w / 2. + 1 + offset) * spatial_scale + 0.5)), width);
    int roi_start_h_in = min(max(0, (int)floor((center_y - shrink_h / 2. + 1 + offset) * spatial_scale + 0.5)), height);
    int roi_end_w_in   = min(max(0, (int)ceil( (center_x + shrink_w / 2. + 1 - offset) * spatial_scale - 0.5)), width);
    int roi_end_h_in   = min(max(0, (int)ceil( (center_y + shrink_h / 2. + 1 - offset) * spatial_scale - 0.5)), height);

    if (roi_start_w > roi_end_w) {
    	roi_start_w = (roi_start_w + roi_end_w) / 2;
	    roi_end_w   = roi_start_w;
    }
    if (roi_start_h > roi_end_h) {
	    roi_start_h = (roi_start_h + roi_end_h) / 2;
	    roi_end_h   = roi_start_h;
    }  
    if (roi_start_w_in > roi_end_w_in) {
	    roi_start_w_in = (roi_start_w_in + roi_end_w_in) / 2;
	    roi_end_w_in   = roi_start_w_in;
    }
    if (roi_start_h_in > roi_end_h_in) {
	    roi_start_h_in = (roi_start_h_in + roi_end_h_in) / 2;
	    roi_end_h_in   = roi_start_h_in;
    } 

    // Force malformed ROIs to be 1x1
    int roi_width  = max(roi_end_w - roi_start_w + 1, 1);
    int roi_height = max(roi_end_h - roi_start_h + 1, 1);
    Dtype bin_size_h = static_cast<Dtype>(roi_height) / static_cast<Dtype>(pooled_height);
    Dtype bin_size_w = static_cast<Dtype>(roi_width)  / static_cast<Dtype>(pooled_width);

    const int hstart = min(height, max(0, static_cast<int>(floor(static_cast<Dtype>(ph)   * bin_size_h)) + roi_start_h));
    const int hend   = min(height, max(0, static_cast<int>(ceil( static_cast<Dtype>(ph+1) * bin_size_h)) + roi_start_h));
    const int wstart = min(width,  max(0, static_cast<int>(floor(static_cast<Dtype>(pw)   * bin_size_w)) + roi_start_w));
    const int wend   = min(width,  max(0, static_cast<int>(ceil( static_cast<Dtype>(pw+1) * bin_size_w)) + roi_start_w));
    bool is_empty = (hend <= hstart) || (wend <= wstart);

    int gw = pw;
    int gh = ph;
    int c = (ctop * group_size + gh) * group_size + gw;
    
    Dtype out_sum = 0;
    bottom_data += (roi_batch_ind * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
	    if (!(w > roi_start_w_in && w < roi_end_w_in && h > roi_start_h_in && h < roi_end_h_in)) {
          // if it is not inside the inner rectangle of the region
          int bottom_index = h * width + w;
          out_sum += bottom_data[bottom_index];
	    }
      }
    }
    Dtype bin_area = (hend - hstart) * (wend - wstart);
    top_data[index] = is_empty ? 0. : out_sum/bin_area;
    mapping_channel[index] = c;
  }
}

template <typename Dtype>
void PSROIRegionPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_rois = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int* mapping_channel_ptr = mapping_channel_.mutable_gpu_data();
  int count = top[0]->count();
  caffe_gpu_set(count, Dtype(0), top_data);
  caffe_gpu_set(count, -1, mapping_channel_ptr);
  // NOLINT_NEXT_LINE(whitespace/operators)
  PSROIRegionPoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, spatial_scale_, offset_, channels_, height_, width_,
      pooled_height_, pooled_width_, bottom_rois, top_data, output_dim_, group_size_,
      mapping_channel_ptr, extend_scale_, shrink_scale_);
  CUDA_POST_KERNEL_CHECK;
}

  template <typename Dtype>
  __global__ void PSROIRegionPoolBackwardAtomic(
    const int nthreads,
    const Dtype* top_diff,
    const int* mapping_channel,
    const int num_rois,
    const Dtype spatial_scale,
    const Dtype offset,
    const int channels,
    const int height, const int width,
    const int pooled_height, const int pooled_width,
    Dtype* bottom_diff,
    const Dtype* bottom_rois,
    const int output_dim, 
    const Dtype extend_scale, const Dtype shrink_scale) {
    CUDA_KERNEL_LOOP(index, nthreads) {
      // The output is in order (n, ctop, ph, pw)
      int pw = index % pooled_width;
      int ph = (index / pooled_width) % pooled_height;
      int n = index / pooled_width / pooled_height / output_dim;
      
      // [start, end) interval for spatial sampling
      bottom_rois += n * 5;
      int roi_batch_ind = bottom_rois[0];
      Dtype center_x = (bottom_rois[1] + bottom_rois[3]) / 2.;
      Dtype center_y = (bottom_rois[2] + bottom_rois[4]) / 2.;
      Dtype extend_w = (bottom_rois[3] - bottom_rois[1] + 1) * extend_scale;
      Dtype extend_h = (bottom_rois[4] - bottom_rois[2] + 1) * extend_scale;
      Dtype shrink_w = (bottom_rois[3] - bottom_rois[1] + 1) * shrink_scale;
      Dtype shrink_h = (bottom_rois[4] - bottom_rois[2] + 1) * shrink_scale;
      
      // outer rectangle of the region
      int roi_start_w   = min(max(0, (int)floor((center_x - extend_w / 2. + 1 + offset) * spatial_scale + 0.5)), width);
      int roi_start_h   = min(max(0, (int)floor((center_y - extend_h / 2. + 1 + offset) * spatial_scale + 0.5)), height);
      int roi_end_w     = min(max(0, (int)ceil( (center_x + extend_w / 2. + 1 - offset) * spatial_scale - 0.5)), width);
      int roi_end_h     = min(max(0, (int)ceil( (center_y + extend_h / 2. + 1 - offset) * spatial_scale - 0.5)), height);
  
      // inner rectangle of the region
      int roi_start_w_in = min(max(0, (int)floor((center_x - shrink_w / 2. + 1 + offset) * spatial_scale + 0.5)), width);
      int roi_start_h_in = min(max(0, (int)floor((center_y - shrink_h / 2. + 1 + offset) * spatial_scale + 0.5)), height);
      int roi_end_w_in   = min(max(0, (int)ceil( (center_x + shrink_w / 2. + 1 - offset) * spatial_scale - 0.5)), width);
      int roi_end_h_in   = min(max(0, (int)ceil( (center_y + shrink_h / 2. + 1 - offset) * spatial_scale - 0.5)), height);
      
      if (roi_start_w > roi_end_w) {
      	roi_start_w = (roi_start_w + roi_end_w) / 2;
  	    roi_end_w   = roi_start_w;
      }
      if (roi_start_h > roi_end_h) {
  	    roi_start_h = (roi_start_h + roi_end_h) / 2;
  	    roi_end_h   = roi_start_h;
      }  
      if (roi_start_w_in > roi_end_w_in) {
  	    roi_start_w_in = (roi_start_w_in + roi_end_w_in) / 2;
  	    roi_end_w_in   = roi_start_w_in;
      }
      if (roi_start_h_in > roi_end_h_in) {
  	    roi_start_h_in = (roi_start_h_in + roi_end_h_in) / 2;
  	    roi_end_h_in   = roi_start_h_in;
      } 
  
      // Force malformed ROIs to be 1x1
      int roi_width  = max(roi_end_w - roi_start_w + 1, 1);
      int roi_height = max(roi_end_h - roi_start_h + 1, 1);
      Dtype bin_size_h = static_cast<Dtype>(roi_height) / static_cast<Dtype>(pooled_height);
      Dtype bin_size_w = static_cast<Dtype>(roi_width)  / static_cast<Dtype>(pooled_width);
  
      const int hstart = min(height, max(0, static_cast<int>(floor(static_cast<Dtype>(ph)   * bin_size_h)) + roi_start_h));
      const int hend   = min(height, max(0, static_cast<int>(ceil( static_cast<Dtype>(ph+1) * bin_size_h)) + roi_start_h));
      const int wstart = min(width,  max(0, static_cast<int>(floor(static_cast<Dtype>(pw)   * bin_size_w)) + roi_start_w));
      const int wend   = min(width,  max(0, static_cast<int>(ceil( static_cast<Dtype>(pw+1) * bin_size_w)) + roi_start_w));
      bool is_empty = (hend <= hstart) || (wend <= wstart);

      // Compute c at bottom
      int c = mapping_channel[index];
      Dtype* offset_bottom_diff = bottom_diff + (roi_batch_ind * channels + c) * height * width;
      Dtype bin_area = (hend - hstart)*(wend - wstart);
      Dtype diff_val = is_empty ? 0. : top_diff[index] / bin_area;
      for (int h = hstart; h < hend; ++h){
        for (int w = wstart; w < wend; ++w)
	    if (!(w > roi_start_w_in && w < roi_end_w_in && h > roi_start_h_in && h < roi_end_h_in)) {
          int bottom_index = h*width + w;
          caffe_gpu_atomic_add(diff_val, offset_bottom_diff + bottom_index);
        
        }
      }
    }
  }
  
template <typename Dtype>
void PSROIRegionPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* bottom_rois = bottom[1]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int bottom_count = bottom[0]->count();
  const int* mapping_channel_ptr = mapping_channel_.gpu_data();
  caffe_gpu_set(bottom[1]->count(), Dtype(0), bottom[1]->mutable_gpu_diff());
  caffe_gpu_set(bottom_count, Dtype(0), bottom_diff);
  const int count = top[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  PSROIRegionPoolBackwardAtomic<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, top_diff, mapping_channel_ptr, top[0]->num(), spatial_scale_, offset_, channels_,
      height_, width_, pooled_height_, pooled_width_, bottom_diff, bottom_rois, output_dim_, extend_scale_, shrink_scale_);
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(PSROIRegionPoolingLayer);

}  // namespace caffe
