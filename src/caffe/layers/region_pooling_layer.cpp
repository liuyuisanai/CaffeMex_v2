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
using std::floor;
using std::ceil;

namespace caffe {

template <typename Dtype>
void RegionPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  RegionPoolingParameter region_pool_param = this->layer_param_.region_pooling_param();
  CHECK_GT(region_pool_param.pooled_h(), 0)
      << "pooled_h must be > 0";
  CHECK_GT(region_pool_param.pooled_w(), 0)
      << "pooled_w must be > 0";
  pooled_height_ = region_pool_param.pooled_h();
  pooled_width_  = region_pool_param.pooled_w();
  spatial_scale_ = region_pool_param.spatial_scale();
  offset_ 	 = region_pool_param.offset();
  LOG(INFO) << "Spatial scale: " << spatial_scale_;
}

template <typename Dtype>
void RegionPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  channels_ = bottom[0]->channels();
  height_   = bottom[0]->height();
  width_    = bottom[0]->width();
  top[0]->Reshape(bottom[1]->num(), channels_, pooled_height_,  pooled_width_);
  max_idx_.Reshape(bottom[1]->num(), channels_, pooled_height_, pooled_width_);
}

template <typename Dtype>
void RegionPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_rois = bottom[1]->cpu_data();
  // Number of ROIs
  int num_rois = bottom[1]->num();
  int batch_size = bottom[0]->num();
  int top_count = top[0]->count();
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_set(top_count, Dtype(-FLT_MAX), top_data);
  int* argmax_data = max_idx_.mutable_cpu_data();
  caffe_set(top_count, -1, argmax_data);

  // For each ROI R = [batch_index, x_outer_1, y_outer_1, x_outer_2, y_outer_2, x_inner_1, y_inner_1, x_inner_2, y_inner_2]: 
  // where R_outer = [x_outer_1, y_outer_1, x_outer_2, y_outer_2] is the outer rectangle of the region and 
  // R_inner = [x_inner_1, y_inner_1, x_inner_2, y_inner_2] is the inner rectangle of the region
  // max pooler over R by ignoring (setting to zero) the activations that lay inside the inner rectangle R_inner
  for (int n = 0; n < num_rois; ++n) {
    int roi_batch_ind  = bottom_rois[0];
    // outer rectangle of the region
    int roi_start_w    = static_cast<int>(floor(((bottom_rois[1] + 1 + offset_) * spatial_scale_) + 0.5));
    int roi_start_h    = static_cast<int>(floor(((bottom_rois[2] + 1 + offset_) * spatial_scale_) + 0.5));
    int roi_end_w      = static_cast<int>(ceil( ((bottom_rois[3] + 1 - offset_) * spatial_scale_) - 0.5));
    int roi_end_h      = static_cast<int>(ceil( ((bottom_rois[4] + 1 - offset_) * spatial_scale_) - 0.5));
	
    // inner rectangle of the region
    int roi_start_w_in = static_cast<int>(floor(((bottom_rois[5] + 1 + offset_) * spatial_scale_) + 0.5));
    int roi_start_h_in = static_cast<int>(floor(((bottom_rois[6] + 1 + offset_) * spatial_scale_) + 0.5));
    int roi_end_w_in   = static_cast<int>(ceil( ((bottom_rois[7] + 1 - offset_) * spatial_scale_) - 0.5));
    int roi_end_h_in   = static_cast<int>(ceil( ((bottom_rois[8] + 1 - offset_) * spatial_scale_) - 0.5));


	if (roi_start_w > roi_end_w)
	{
		roi_start_w = (roi_start_w + roi_end_w) / 2;
		roi_end_w   = roi_start_w;
	}
	if (roi_start_h > roi_end_h)
	{
		roi_start_h = (roi_start_h + roi_end_h) / 2;
		roi_end_h   = roi_start_h;
	}  
	if (roi_start_w_in > roi_end_w_in)
	{
		roi_start_w_in = (roi_start_w_in + roi_end_w_in) / 2;
		roi_end_w_in   = roi_start_w_in;
	}
	if (roi_start_h_in > roi_end_h_in)
	{
		roi_start_h_in = (roi_start_h_in + roi_end_h_in) / 2;
		roi_end_h_in   = roi_start_h_in;
	} 

    CHECK_GE(roi_batch_ind, 0);
    CHECK_LT(roi_batch_ind, batch_size);

    const int roi_height = max(roi_end_h - roi_start_h + 1, 1);
    const int roi_width  = max(roi_end_w - roi_start_w + 1, 1);
    const Dtype bin_size_h = static_cast<Dtype>(roi_height) / static_cast<Dtype>(pooled_height_);
    const Dtype bin_size_w = static_cast<Dtype>(roi_width)  / static_cast<Dtype>(pooled_width_);
   
    const Dtype* batch_data = bottom_data + bottom[0]->offset(roi_batch_ind);

    for (int c = 0; c < channels_; ++c) {
      for (int ph = 0; ph < pooled_height_; ++ph) {
        for (int pw = 0; pw < pooled_width_; ++pw) {
          // Compute pooling region for this output unit:
          //  start (included) = floor(ph * roi_height / pooled_height_)
          //  end (excluded) = ceil((ph + 1) * roi_height / pooled_height_)
          
          const int hstart = min(height_, max(0, static_cast<int>(floor(static_cast<Dtype>(ph)   * bin_size_h)) + roi_start_h));
	  const int hend   = min(height_, max(0, static_cast<int>(ceil( static_cast<Dtype>(ph+1) * bin_size_h)) + roi_start_h));
          const int wstart = min(width_,  max(0, static_cast<int>(floor(static_cast<Dtype>(pw)   * bin_size_w)) + roi_start_w));
          const int wend   = min(width_,  max(0, static_cast<int>(ceil( static_cast<Dtype>(pw+1) * bin_size_w)) + roi_start_w));

          const int pool_index = ph * pooled_width_ + pw;
          top_data[pool_index] = 0;
          argmax_data[pool_index] = -1;


          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
	      if (!(w > roi_start_w_in && w < roi_end_w_in && h > roi_start_h_in && h < roi_end_h_in)) { 
                // if it is not inside the inner rectangle of the region
		const int index = h * width_ + w;
		if (batch_data[index] > top_data[pool_index]) {
		  top_data[pool_index] = batch_data[index];
		  argmax_data[pool_index] = index;
		}					
	      }
            }
          }
        }
      }
      // Increment all data pointers by one channel
      batch_data += bottom[0]->offset(0, 1);
      top_data += top[0]->offset(0, 1);
      argmax_data += max_idx_.offset(0, 1);
    }
    // Increment ROI data pointer
    bottom_rois += bottom[1]->offset(1);
  }
}

template <typename Dtype>
void RegionPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	if (!propagate_down[0]) { return; }
  	const Dtype* bottom_rois = bottom[1]->cpu_data();
  	const Dtype* top_diff = top[0]->cpu_diff();
  	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  	const int* argmax_data = max_idx_.cpu_data();
	const int num_rois = bottom[1]->num();
	
  	for(int n = 0; n < top[0]->num(); ++n) {
		for(int c = 0; c < channels_; ++c) {
			for(int h = 0; h < height_; ++h) {
				for(int w = 0; w < width_; ++w) {
					Dtype gradient = 0;
					// Accumulate gradient over all ROIs that pooled this element
					for (int roi_n = 0; roi_n < num_rois; ++roi_n) {
						const Dtype* offset_bottom_rois = bottom_rois + roi_n * 9;
						int roi_batch_ind = offset_bottom_rois[0];
						// Skip if ROI's batch index doesn't match n
						if (n != roi_batch_ind) { continue; }

						int roi_start_w = floor((offset_bottom_rois[1] + 1 + offset_) * spatial_scale_ + 0.5);
						int roi_start_h = floor((offset_bottom_rois[2] + 1 + offset_) * spatial_scale_ + 0.5);
						int roi_end_w   = ceil( (offset_bottom_rois[3] + 1 - offset_) * spatial_scale_ - 0.5);
						int roi_end_h   = ceil( (offset_bottom_rois[4] + 1 - offset_) * spatial_scale_ - 0.5);

						int roi_start_w_in = floor((offset_bottom_rois[5] + 1 + offset_) * spatial_scale_ + 0.5);
						int roi_start_h_in = floor((offset_bottom_rois[6] + 1 + offset_) * spatial_scale_ + 0.5);
						int roi_end_w_in   = ceil( (offset_bottom_rois[7] + 1 - offset_) * spatial_scale_ - 0.5);
						int roi_end_h_in   = ceil( (offset_bottom_rois[8] + 1 - offset_) * spatial_scale_ - 0.5);

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

					    if (!in_roi) { continue; }

					    int top_offset = (roi_n * channels_ + c) * pooled_height_ * pooled_width_;
					    const Dtype* offset_top_diff = top_diff + top_offset;
					    const int* offset_argmax_data = argmax_data + top_offset;

					    // Compute feasible set of pooled units that could have pooled
					    // this bottom unit

				            // Force malformed ROIs to be 1x1
					    int roi_width  = max(roi_end_w - roi_start_w + 1, 1);
					    int roi_height = max(roi_end_h - roi_start_h + 1, 1);

					    Dtype bin_size_h = static_cast<Dtype>(roi_height) / static_cast<Dtype>(pooled_height_);
					    Dtype bin_size_w = static_cast<Dtype>(roi_width)  / static_cast<Dtype>(pooled_width_);

					    int phstart = floor(static_cast<Dtype>(h - roi_start_h)     / bin_size_h);
					    int phend   =  ceil(static_cast<Dtype>(h - roi_start_h + 1) / bin_size_h);
					    int pwstart = floor(static_cast<Dtype>(w - roi_start_w)     / bin_size_w);
					    int pwend   =  ceil(static_cast<Dtype>(w - roi_start_w + 1) / bin_size_w);

					    phstart = min(max(phstart, 0), pooled_height_);
					    phend   = min(max(phend,   0), pooled_height_);
					    pwstart = min(max(pwstart, 0), pooled_width_);
					    pwend   = min(max(pwend,   0), pooled_width_);

						for (int ph = phstart; ph < phend; ++ph) {
							for (int pw = pwstart; pw < pwend; ++pw) {
								if (offset_argmax_data[ph * pooled_width_ + pw] == (h * width_ + w)) {
									gradient += offset_top_diff[ph * pooled_width_ + pw];
								}
							}
						}
					}
					const int index = ((n * channels_ + c) * height_ + h) * width_ + w;
					bottom_diff[index] = gradient;
				}
			}
		}
	}
}

#ifdef CPU_ONLY
STUB_GPU(RegionPoolingLayer);
#endif

INSTANTIATE_CLASS(RegionPoolingLayer);
REGISTER_LAYER_CLASS(RegionPooling);

}  // namespace caffe
