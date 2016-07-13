// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// ------------------------------------------------------------------

#include <cfloat>

#include "caffe/frcnn/layers/point_pooling_layer.hpp"

using std::max;
using std::min;

namespace caffe {

template <typename Dtype>
__global__ void MaxPointPoolForward(const int nthreads, const Dtype* bottom_data, const int ncls, 
	int channels, const int height, const int width, const int* cls_ch,
    const Dtype* bottom_ids, const Dtype* bottom_points, Dtype* top_data, int* argmax_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, cls, 1, 1) is an element in the pooled output
    int cls = index % ncls;
    int n = index / ncls;

    cls_ch += cls * 2;
	argmax_data += n * channels;
    bottom_ids += n;
    int roi_batch_ind = bottom_ids[0];
	top_data[index] = 0;
	
	for (int ch = cls_ch[0]; ch <= cls_ch[1]; ch++) {
		const Dtype* feat_map = bottom_data + (roi_batch_ind * channels + ch) * height * width;
		const Dtype* pnt = bottom_points + (n * channels + ch) * 4;
		int x1 = round(pnt[0]);
        int y1 = round(pnt[1]);
        int x2 = round(pnt[2]);
        int y2 = round(pnt[3]);
        if (x2 == -1) {
            continue; // if set -1, the point is absent
        }
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
		top_data[index] += maxval;
	}
	top_data[index] /= (cls_ch[1] - cls_ch[0] + 1);
	
  }
}

template <typename Dtype>
__global__ void AvePointPoolForward(const int nthreads, const Dtype* bottom_data, const int ncls, 
	int channels, const int height, const int width, const int* cls_ch,
    const Dtype* bottom_ids, const Dtype* bottom_points, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, cls, 1, 1) is an element in the pooled output
    int cls = index % ncls;
    int n = index / ncls;

    cls_ch += cls * 2;
    bottom_ids += n;
    int roi_batch_ind = bottom_ids[0];
	top_data[index] = 0;
	
	for (int ch = cls_ch[0]; ch <= cls_ch[1]; ch++) {
		const Dtype* feat_map = bottom_data + (roi_batch_ind * channels + ch) * height * width;
		const Dtype* pnt = bottom_points + (n * channels + ch) * 4;
		int x1 = round(pnt[0]);
        int y1 = round(pnt[1]);
        int x2 = round(pnt[2]);
        int y2 = round(pnt[3]);
        if (x2 == -1) {
            continue; // if set -1, the point is absent
        }
		Dtype avgval = 0;
		for (int h=y1; h<=y2; h++) {
			for (int w=x1; w<=x2; w++) {
				int ind = h * width + w;
				avgval += feat_map[ind];
			}
		}
		top_data[index] += avgval / (y2 - y1 + 1) / (x2 - x1 + 1);
	}
	top_data[index] /= (cls_ch[1] - cls_ch[0] + 1);
	
  }
}
template <typename Dtype>
void PointPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->gpu_data();
	const Dtype* bottom_ids = bottom[1]->gpu_data();
	const Dtype* bottom_points = bottom[2]->gpu_data();
    const int* cls_ch = class_channel_.gpu_data();
	Dtype* top_data = top[0]->mutable_gpu_data();
	int* argmax_data = max_idx_.mutable_gpu_data();
	int count = top[0]->count();
    caffe_gpu_set(count, Dtype(0), top_data);
    int idx_count = max_idx_.count();
    caffe_gpu_set(idx_count, -1, argmax_data);
	// NOLINT_NEXT_LINE(whitespace/operators)
	if (use_maxpool_)
        MaxPointPoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
		  count, bottom_data, ncls_, channels_, height_, width_,
		  cls_ch, bottom_ids, bottom_points, top_data, argmax_data);
	else
        AvePointPoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, bottom_data, ncls_, channels_, height_, width_,
          cls_ch, bottom_ids, bottom_points, top_data);
	CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void MaxPointPoolBackward(const int nthreads, const Dtype* top_diff,
    const int* argmax_data, const int num_rois, const int ncls, const int channels,
    const int height, const int width, const int* cls_ch, 
    Dtype* bottom_diff, const Dtype* bottom_ids, const Dtype* bottom_points) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, ch, h, w) coords in bottom data
    int w = index % width;
    int h = (index / width) % height;
    int ch = (index / width / height) % channels;
    int n = index / width / height / channels;
    
    int cls = -1;
    for (int i=0; i<ncls; i++) {
        if (ch >= cls_ch[0] && ch <= cls_ch[1]) {
            cls = i;
            break;
        }
        cls_ch += 2;
    }
    int ch_len = cls_ch[1] - cls_ch[0] + 1;

    Dtype gradient = 0;
    // Accumulate gradient over all ROIs that pooled this element
    for (int roi_n = 0; roi_n < num_rois; ++roi_n) {
        const Dtype* offset_bottom_ids = bottom_ids + roi_n;
        int roi_batch_ind = offset_bottom_ids[0];
        // Skip if ROI's batch index doesn't match n
        if (n != roi_batch_ind) {
            continue;
        }
        
        const Dtype* pnt = bottom_points + (roi_n * channels + ch) * 4;
        int x1 = round(pnt[0]);
        int y1 = round(pnt[1]);
        int x2 = round(pnt[2]);
        int y2 = round(pnt[3]);
        // Skip if point area doesn't include (h, w)
        const bool in_area = (w >= x1 && w <= x2 && h >= y1 && h <= y2);
        if (!in_area) {
            continue;
        }

        const Dtype* offset_top_diff = top_diff + roi_n * ncls + cls;
        const int* offset_argmax_data = argmax_data + roi_n * channels + ch;
        if (offset_argmax_data[0] == (h * width + w)) {
            gradient += offset_top_diff[0] / ch_len;
        }
    } // for roi_n
    bottom_diff[index] = gradient;
  }
}

template <typename Dtype>
__global__ void AvePointPoolBackward(const int nthreads, const Dtype* top_diff,
    const int num_rois, const int ncls, const int channels,
    const int height, const int width, const int* cls_ch, 
    Dtype* bottom_diff, const Dtype* bottom_ids, const Dtype* bottom_points) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, ch, h, w) coords in bottom data
    int w = index % width;
    int h = (index / width) % height;
    int ch = (index / width / height) % channels;
    int n = index / width / height / channels;
    
    int cls = -1;
    for (int i=0; i<ncls; i++) {
        if (ch >= cls_ch[0] && ch <= cls_ch[1]) {
            cls = i;
            break;
        }
        cls_ch += 2;
    }
    int ch_len = cls_ch[1] - cls_ch[0] + 1;

    Dtype gradient = 0;
    // Accumulate gradient over all ROIs that pooled this element
    for (int roi_n = 0; roi_n < num_rois; ++roi_n) {
        const Dtype* offset_bottom_ids = bottom_ids + roi_n;
        int roi_batch_ind = offset_bottom_ids[0];
        // Skip if ROI's batch index doesn't match n
        if (n != roi_batch_ind) {
            continue;
        }
        
        const Dtype* pnt = bottom_points + (roi_n * channels + ch) * 4;
        int x1 = round(pnt[0]);
        int y1 = round(pnt[1]);
        int x2 = round(pnt[2]);
        int y2 = round(pnt[3]);
        // Skip if point area doesn't include (h, w)
        const bool in_area = (w >= x1 && w <= x2 && h >= y1 && h <= y2);
        if (!in_area) {
            continue;
        }

        const Dtype* offset_top_diff = top_diff + roi_n * ncls + cls;
        gradient += offset_top_diff[0] / ch_len / (x2 - x1 + 1) / (y2 - y1 + 1);
    } // for roi_n
    bottom_diff[index] = gradient;
  }
}

template <typename Dtype>
void PointPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	if (!propagate_down[0]) {
		return;
	}
	const Dtype* bottom_ids = bottom[1]->gpu_data();
	const Dtype* bottom_points = bottom[2]->gpu_data();
	const Dtype* top_diff = top[0]->gpu_diff();
	Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
	const int count = bottom[0]->count();
	caffe_gpu_set(count, Dtype(0.), bottom_diff);
	const int* argmax_data = max_idx_.gpu_data();
    const int* cls_ch = class_channel_.gpu_data();
	// NOLINT_NEXT_LINE(whitespace/operators)
	if (use_maxpool_)
		MaxPointPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
		  count, top_diff, argmax_data, top[0]->num(), ncls_, channels_, 
		  height_, width_, cls_ch, bottom_diff, bottom_ids, bottom_points);
	else
		AvePointPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
		  count, top_diff, top[0]->num(), ncls_, channels_, height_, width_, 
		  cls_ch, bottom_diff, bottom_ids, bottom_points);
	CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(PointPoolingLayer);

}  // namespace caffe
