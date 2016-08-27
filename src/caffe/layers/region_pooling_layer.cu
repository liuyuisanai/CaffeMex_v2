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

#include "caffe/layers/region_pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;
using std::min;

namespace caffe {

template <typename Dtype>
__global__ void RegionPoolForward(const int nthreads, const Dtype* bottom_data,
    const Dtype spatial_scale, const Dtype offset, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const Dtype* bottom_rois, Dtype* top_data, int* argmax_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

  // For each ROI R = [batch_index, x_outer_1, y_outer_1, x_outer_2, y_outer_2, x_inner_1, y_inner_1, x_inner_2, y_inner_2]: 
  // where R_outer = [x_outer_1, y_outer_1, x_outer_2, y_outer_2] is the outer rectangle of the region and 
  // R_inner = [x_inner_1, y_inner_1, x_inner_2, y_inner_2] is the inner rectangle of the region
  // max pooler over R by ignoring (setting to zero) the activations that lay inside the inner rectangle R_inner

    bottom_rois += n * 9;
    int roi_batch_ind = bottom_rois[0];


    // outer rectangle of the region
    int roi_start_w   = floor((bottom_rois[1] + 1 + offset) * spatial_scale + 0.5);
    int roi_start_h   = floor((bottom_rois[2] + 1 + offset) * spatial_scale + 0.5);
    int roi_end_w     = ceil( (bottom_rois[3] + 1 - offset) * spatial_scale - 0.5);
    int roi_end_h     = ceil( (bottom_rois[4] + 1 - offset) * spatial_scale - 0.5);

    // inner rectangle of the region
    int roi_start_w_in = floor((bottom_rois[5] + 1 + offset) * spatial_scale + 0.5);
    int roi_start_h_in = floor((bottom_rois[6] + 1 + offset) * spatial_scale + 0.5);
    int roi_end_w_in   = ceil( (bottom_rois[7] + 1 - offset) * spatial_scale - 0.5);
    int roi_end_h_in   = ceil( (bottom_rois[8] + 1 - offset) * spatial_scale - 0.5);

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

    Dtype maxval = 0; 

    int maxidx = -1;
    bottom_data += (roi_batch_ind * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
	if (!(w > roi_start_w_in && w < roi_end_w_in && h > roi_start_h_in && h < roi_end_h_in)) {
          // if it is not inside the inner rectangle of the region
          int bottom_index = h * width + w;
          if (bottom_data[bottom_index] > maxval) {
            maxval = bottom_data[bottom_index];
            maxidx = bottom_index;
          }
	}
      }
    }
    top_data[index] = maxval;
    argmax_data[index] = maxidx;
  }
}

template <typename Dtype>
void RegionPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_rois = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int* argmax_data = max_idx_.mutable_gpu_data();
  int count = top[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  RegionPoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, spatial_scale_, offset_, channels_, height_, width_,
      pooled_height_, pooled_width_, bottom_rois, top_data, argmax_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void RegionPoolBackward(const int nthreads, const Dtype* top_diff,
    const int* argmax_data, const int num_rois, const Dtype spatial_scale,
    const Dtype offset, const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, Dtype* bottom_diff,
    const Dtype* bottom_rois) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, h, w) coords in bottom data
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;

    Dtype gradient = 0;
    // Accumulate gradient over all ROIs that pooled this element
    for (int roi_n = 0; roi_n < num_rois; ++roi_n) {
      const Dtype* offset_bottom_rois = bottom_rois + roi_n * 9;
      int roi_batch_ind = offset_bottom_rois[0];
      // Skip if ROI's batch index doesn't match n
      if (n != roi_batch_ind) {
        continue;
      }


      int roi_start_w = floor((offset_bottom_rois[1] + 1 + offset) * spatial_scale + 0.5);
      int roi_start_h = floor((offset_bottom_rois[2] + 1 + offset) * spatial_scale + 0.5);
      int roi_end_w   = ceil( (offset_bottom_rois[3] + 1 - offset) * spatial_scale - 0.5);
      int roi_end_h   = ceil( (offset_bottom_rois[4] + 1 - offset) * spatial_scale - 0.5);

      int roi_start_w_in = floor((offset_bottom_rois[5] + 1 + offset) * spatial_scale + 0.5);
      int roi_start_h_in = floor((offset_bottom_rois[6] + 1 + offset) * spatial_scale + 0.5);
      int roi_end_w_in   = ceil( (offset_bottom_rois[7] + 1 - offset) * spatial_scale - 0.5);
      int roi_end_h_in   = ceil( (offset_bottom_rois[8] + 1 - offset) * spatial_scale - 0.5);

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

      // Skip if ROI doesn't include (h, w)
      const bool in_roi =  (w >= roi_start_w && w <= roi_end_w &&
                            h >= roi_start_h && h <= roi_end_h) && 
		          !(w > roi_start_w_in && w < roi_end_w_in && 
			    h > roi_start_h_in && h < roi_end_h_in);

      if (!in_roi) {
        continue;
      }

      int top_offset = (roi_n * channels + c) * pooled_height * pooled_width;
      const Dtype* offset_top_diff = top_diff + top_offset;
      const int* offset_argmax_data = argmax_data + top_offset;

      // Compute feasible set of pooled units that could have pooled
      // this bottom unit

      // Force malformed ROIs to be 1x1
      int roi_width = max(roi_end_w - roi_start_w + 1, 1);
      int roi_height = max(roi_end_h - roi_start_h + 1, 1);

      Dtype bin_size_h = static_cast<Dtype>(roi_height) / static_cast<Dtype>(pooled_height);
      Dtype bin_size_w = static_cast<Dtype>(roi_width)  / static_cast<Dtype>(pooled_width);

      int phstart = floor(static_cast<Dtype>(h - roi_start_h) / bin_size_h);
      int phend = ceil(static_cast<Dtype>(h - roi_start_h + 1) / bin_size_h);
      int pwstart = floor(static_cast<Dtype>(w - roi_start_w) / bin_size_w);
      int pwend = ceil(static_cast<Dtype>(w - roi_start_w + 1) / bin_size_w);

      phstart = min(max(phstart, 0), pooled_height);
      phend = min(max(phend, 0), pooled_height);
      pwstart = min(max(pwstart, 0), pooled_width);
      pwend = min(max(pwend, 0), pooled_width);

      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          if (offset_argmax_data[ph * pooled_width + pw] == (h * width + w)) {
            gradient += offset_top_diff[ph * pooled_width + pw];
          }
        }
      }
    }
    bottom_diff[index] = gradient;
  }
}

template <typename Dtype>
void RegionPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* bottom_rois = bottom[1]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);
  const int* argmax_data = max_idx_.gpu_data();
  // NOLINT_NEXT_LINE(whitespace/operators)
  RegionPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, top_diff, argmax_data, top[0]->num(), spatial_scale_, offset_, channels_,
      height_, width_, pooled_height_, pooled_width_, bottom_diff, bottom_rois);
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(RegionPoolingLayer);

}  // namespace caffe
