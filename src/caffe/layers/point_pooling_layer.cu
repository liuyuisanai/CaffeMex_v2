// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by LBY, LY
// ------------------------------------------------------------------

#include <cfloat>

#include "caffe/layers/point_pooling_layer.hpp"
#include "caffe/util/gpu_util.cuh"

using std::max;
using std::min;

namespace caffe {

template <typename Dtype>
__global__ void MaxPointPoolForward(const int nthreads, const Dtype* bottom_data, const int ncls, 
	int channels, const int height, const int width, const int* cls_ch, const Dtype spatial_scale,
    const Dtype* bottom_ids, const Dtype* bottom_points, const Dtype* bottom_points_conf, Dtype* top_data, int* argmax_data, const bool use_valid_channel, const Dtype conf_th) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, cls, 1, 1) is an element in the pooled output, n represents roi id
    int cls = index % ncls;
    int n = index / ncls;

    cls_ch += cls * 2;
	argmax_data += n * channels;
    bottom_ids += n;
    int roi_batch_ind = bottom_ids[0];
	int ch_len = cls_ch[1] - cls_ch[0] + 1;
    
	for (int ch = cls_ch[0]; ch <= cls_ch[1]; ch++) {
        argmax_data[ch] = -1;
		const Dtype* feat_map = bottom_data + (roi_batch_ind * channels + ch) * height * width;
		const Dtype* pnt = bottom_points + (n * channels + ch) * 4;
		const Dtype* conf = bottom_points_conf + n * channels + ch;
        if (use_valid_channel) {
            bool is_valid = conf[0] > conf_th;
            if (!is_valid) {
                ch_len--;
                continue; // the point is absent
            }
        }
        int x1 = floor(pnt[0] * spatial_scale);
        int y1 = floor(pnt[1] * spatial_scale);
        int x2 = ceil(pnt[2] * spatial_scale);
        int y2 = ceil(pnt[3] * spatial_scale);
        x1 = min(max(x1, 0), width - 1);
        y1 = min(max(y1, 0), height - 1);
        x2 = min(max(x2, 0), width - 1);
        y2 = min(max(y2, 0), height - 1);
		Dtype maxval = -FLT_MAX;
		for (int h=y1; h<=y2; h++) {
			for (int w=x1; w<=x2; w++) {
				int ind = h * width + w;
				if (feat_map[ind] > maxval) {
					maxval = feat_map[ind];
					argmax_data[ch] = ind;
				}
			}
		}
        if (!use_valid_channel) // use all channels, use confidence as weight
            maxval *= conf[0];
		top_data[index] += maxval;
	}
    if (ch_len > 0)
        top_data[index] /= ch_len;
	
  }
}

template <typename Dtype>
__global__ void AvePointPoolForward(const int nthreads, const Dtype* bottom_data, const int ncls, 
	int channels, const int height, const int width, const int* cls_ch, const Dtype spatial_scale,
    const Dtype* bottom_ids, const Dtype* bottom_points, const Dtype* bottom_points_conf, Dtype* top_data, const bool use_valid_channel, const Dtype conf_th) {

    CUDA_KERNEL_LOOP(index, nthreads) {
	// (n, cls, 1, 1) is an element in the pooled output, n represents roi id
	int cls = index % ncls;
	int n = index / ncls;

	const int* cls_channel = cls_ch + cls * 2;
	int roi_batch_ind = bottom_ids[n];
	int ch_len = cls_channel[1] - cls_channel[0] + 1;
	
	for (int ch = cls_channel[0]; ch <= cls_channel[1]; ch++) {
	    const Dtype* feat_map = bottom_data + (roi_batch_ind * channels + ch) * height * width;
	    const Dtype* pnt = bottom_points + (n * channels + ch) * 4;
	    const Dtype* conf = bottom_points_conf + n * channels + ch;
	    if (use_valid_channel) {
		if (conf[0] < conf_th) {
		    ch_len--;
		    continue; // the point is absent
		}
	    }
	    int x1 = floor(pnt[0] * spatial_scale);
	    int y1 = floor(pnt[1] * spatial_scale);
	    int x2 = ceil(pnt[2] * spatial_scale);
	    int y2 = ceil(pnt[3] * spatial_scale);
	    x1 = min(max(x1, 0), width - 1);
	    y1 = min(max(y1, 0), height - 1);
	    x2 = min(max(x2, 0), width - 1);
	    y2 = min(max(y2, 0), height - 1);

	    Dtype avgval = 0;
	    for (int h=y1; h<=y2; h++) {
		for (int w=x1; w<=x2; w++) {
		    int ind = h * width + w;
		    avgval += feat_map[ind];
		}
	    }
	    if (!use_valid_channel) { // use all channels, use confidence as weight
		avgval *= conf[0];
	    }

	    top_data[index] += avgval / (y2 - y1 + 1) / (x2 - x1 + 1);
	}
	if (ch_len > 0) {
	    top_data[index] /= ch_len;
	}
    }
}
template <typename Dtype>
void PointPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->gpu_data();
	const Dtype* bottom_ids = bottom[1]->gpu_data(); // n_roi * 1 * 1 * 1
	const Dtype* bottom_points = bottom[2]->gpu_data(); // n_roi * all_pnt_num * 4 * 1
    const Dtype* bottom_points_conf = bottom[3]->gpu_data(); // n_roi * all_pnt_num * 1 * 1
    const int* cls_ch = class_channel_.gpu_data();
	Dtype* top_data = top[0]->mutable_gpu_data();
	int* argmax_data = max_idx_.mutable_gpu_data();
	int count = top[0]->count();
    caffe_gpu_set(count, Dtype(0), top_data);
	// NOLINT_NEXT_LINE(whitespace/operators)
	if (use_maxpool_)
        MaxPointPoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
		  count, bottom_data, ncls_, channels_, height_, width_,
		  cls_ch, spatial_scale_, bottom_ids, bottom_points, bottom_points_conf, top_data, argmax_data, use_valid_channel_, conf_th_);
	else
        AvePointPoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, bottom_data, ncls_, channels_, height_, width_,
          cls_ch, spatial_scale_, bottom_ids, bottom_points, bottom_points_conf, top_data, use_valid_channel_, conf_th_);
	CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void MaxPointPoolBackward(const int nthreads, const Dtype* top_diff,
    const int* argmax_data, const int num_rois, const int ncls, const int channels,
    const int height, const int width, const int* cls_ch, const Dtype spatial_scale,
    Dtype* bottom_diff, const Dtype* bottom_ids, const Dtype* bottom_points, const Dtype* bottom_points_conf, const bool use_valid_channel, const Dtype conf_th) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, cls, 1, 1) is an element in the pooled output, n represents roi id
    int cls = index % ncls;
    int n = index / ncls;

    cls_ch += cls * 2;
	argmax_data += n * channels;
    bottom_ids += n;
    int roi_batch_ind = bottom_ids[0];
	int ch_len = cls_ch[1] - cls_ch[0] + 1;
    
    if (use_valid_channel) {
        for (int ch = cls_ch[0]; ch <= cls_ch[1]; ch++) {
            const Dtype* conf = bottom_points_conf + n * channels + ch;
            bool is_valid = conf[0] > conf_th;
            if (!is_valid) {
                ch_len--;
            }
        }
    }

    for (int ch = cls_ch[0]; ch <= cls_ch[1]; ch++) {
		Dtype* diff = bottom_diff + (roi_batch_ind * channels + ch) * height * width;
        int ind = argmax_data[ch];
        if (ind > -1) { 
            if (use_valid_channel) {
                diff[ind] += top_diff[index] / ch_len;
            }
            else {
                const Dtype* conf = bottom_points_conf + n * channels + ch;
                diff[ind] += top_diff[index] * conf[0] / ch_len;
            }
        }
	}
  }
}

template <typename Dtype>
__global__ void AvePointPoolBackward(const int nthreads, const Dtype* top_diff,
    const int num_rois, const int ncls, const int channels,
    const int height, const int width, const int* cls_ch, const Dtype spatial_scale,
    Dtype* bottom_diff, const Dtype* bottom_ids, const Dtype* bottom_points, const Dtype* bottom_points_conf, const bool use_valid_channel, const Dtype conf_th) {

    CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, cls, 1, 1) is an element in the pooled output, n represents roi id
	int cls = index % ncls;
	int n = index / ncls;

	const int* cls_channel = cls_ch + cls * 2;
	int roi_batch_ind = bottom_ids[n];
	int ch_len = cls_channel[1] - cls_channel[0] + 1;
	    
	if (use_valid_channel) {
	    for (int ch = cls_channel[0]; ch <= cls_channel[1]; ch++) {
		const Dtype* conf = bottom_points_conf + n * channels + ch;
		if (conf[0] < conf_th) {
		    ch_len--;
		}
	    }
	}
    
	for (int ch = cls_channel[0]; ch <= cls_channel[1]; ch++) {
	    Dtype* diff = bottom_diff + (roi_batch_ind * channels + ch) * height * width;
	    const Dtype* pnt = bottom_points + (n * channels + ch) * 4;
	    const Dtype* conf = bottom_points_conf + n * channels + ch;
	    if (use_valid_channel) {
		if (conf[0] < conf_th) {
		    continue; // the point is absent
		}
	    }
	    int x1 = floor(pnt[0] * spatial_scale);
	    int y1 = floor(pnt[1] * spatial_scale);
	    int x2 = ceil(pnt[2] * spatial_scale);
	    int y2 = ceil(pnt[3] * spatial_scale);
	    x1 = min(max(x1, 0), width - 1);
	    y1 = min(max(y1, 0), height - 1);
	    x2 = min(max(x2, 0), width - 1);
	    y2 = min(max(y2, 0), height - 1);
	    for (int h=y1; h<=y2; h++) {
		for (int w=x1; w<=x2; w++) {
		    int ind = h * width + w;
		    if (use_valid_channel) {
			Dtype diff_val = top_diff[index] / ch_len /(y2 - y1 + 1) / (x2 - x1 + 1);
			caffe_gpu_atomic_add(diff_val, diff + ind);
		    } else {
			Dtype diff_val = top_diff[index] * conf[0] / ch_len / (y2 - y1 + 1) / (x2 - x1 + 1);
			caffe_gpu_atomic_add(diff_val, diff + ind);
		    }
		}
	    }
	}
    }
}

template <typename Dtype>
void PointPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	if (!propagate_down[0]) {
		return;
	}
	const Dtype* bottom_ids = bottom[1]->gpu_data(); // n_roi * 1 * 1 * 1
	const Dtype* bottom_points = bottom[2]->gpu_data(); // n_roi * all_pnt_num * 4 * 1
    const Dtype* bottom_points_conf = bottom[3]->gpu_data(); // n_roi * all_pnt_num * 1 * 1
	const Dtype* top_diff = top[0]->gpu_diff();
	Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
	const int count = top[0]->count();
	caffe_gpu_set(bottom[0]->count(), Dtype(0.), bottom_diff);
	const int* argmax_data = max_idx_.gpu_data();
    const int* cls_ch = class_channel_.gpu_data();
	// NOLINT_NEXT_LINE(whitespace/operators)
	if (use_maxpool_)
		MaxPointPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
		  count, top_diff, argmax_data, top[0]->num(), ncls_, channels_, 
		  height_, width_, cls_ch, spatial_scale_, bottom_diff, bottom_ids, bottom_points, bottom_points_conf, use_valid_channel_, conf_th_);
	else
		AvePointPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
		  count, top_diff, top[0]->num(), ncls_, channels_, height_, width_, 
		  cls_ch, spatial_scale_, bottom_diff, bottom_ids, bottom_points, bottom_points_conf, use_valid_channel_, conf_th_);
	CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(PointPoolingLayer);

}  // namespace caffe
