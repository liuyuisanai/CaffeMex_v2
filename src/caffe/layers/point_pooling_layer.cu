// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// ------------------------------------------------------------------

#include <cfloat>

#include "caffe/layers/point_pooling_layer.hpp"

using std::max;
using std::min;

namespace caffe {

template <typename Dtype>
__global__ void MaxPointPoolForward(const int nthreads, const Dtype* bottom_data, const int ncls, 
	int channels, const int height, const int width, const int* cls_ch, const Dtype spatial_scale,
    const Dtype* bottom_ids, const Dtype* bottom_points, const Dtype* bottom_points_valid, Dtype* top_data, int* argmax_data) {
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
		const Dtype* valid = bottom_points_valid + n * channels + ch;
        bool is_valid = valid[0];
        if (!is_valid) {
            ch_len--;
            continue; // the point is absent
        }
        int x1 = round(pnt[0] * spatial_scale);
        int y1 = round(pnt[1] * spatial_scale);
        int x2 = round(pnt[2] * spatial_scale);
        int y2 = round(pnt[3] * spatial_scale);
        x1 = min(max(x1, 0), width);
        y1 = min(max(y1, 0), height);
        x2 = min(max(x2, 0), width);
        y2 = min(max(y2, 0), height);
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
    if (ch_len > 0)
        top_data[index] /= ch_len;
	
  }
}

template <typename Dtype>
__global__ void AvePointPoolForward(const int nthreads, const Dtype* bottom_data, const int ncls, 
	int channels, const int height, const int width, const int* cls_ch, const Dtype spatial_scale,
    const Dtype* bottom_ids, const Dtype* bottom_points, const Dtype* bottom_points_valid, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, cls, 1, 1) is an element in the pooled output, n represents roi id
    int cls = index % ncls;
    int n = index / ncls;

    cls_ch += cls * 2;
    bottom_ids += n;
    int roi_batch_ind = bottom_ids[0];
    int ch_len = cls_ch[1] - cls_ch[0] + 1;
	
	for (int ch = cls_ch[0]; ch <= cls_ch[1]; ch++) {
		const Dtype* feat_map = bottom_data + (roi_batch_ind * channels + ch) * height * width;
		const Dtype* pnt = bottom_points + (n * channels + ch) * 4;
		const Dtype* valid = bottom_points_valid + n * channels + ch;
        bool is_valid = valid[0];
        if (!is_valid) {
            ch_len--;
            continue; // the point is absent
        }
        int x1 = round(pnt[0] * spatial_scale);
        int y1 = round(pnt[1] * spatial_scale);
        int x2 = round(pnt[2] * spatial_scale);
        int y2 = round(pnt[3] * spatial_scale);
        x1 = min(max(x1, 0), width);
        y1 = min(max(y1, 0), height);
        x2 = min(max(x2, 0), width);
        y2 = min(max(y2, 0), height);
		Dtype avgval = 0;
		for (int h=y1; h<=y2; h++) {
			for (int w=x1; w<=x2; w++) {
				int ind = h * width + w;
				avgval += feat_map[ind];
			}
		}
		top_data[index] += avgval / (y2 - y1 + 1) / (x2 - x1 + 1);
	}
    if (ch_len > 0)
        top_data[index] /= ch_len;
	
  }
}
template <typename Dtype>
void PointPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->gpu_data();
	const Dtype* bottom_ids = bottom[1]->gpu_data(); // n_roi * 1 * 1 * 1
	const Dtype* bottom_points = bottom[2]->gpu_data(); // n_roi * all_pnt_num * 4 * 1
    const Dtype* bottom_points_valid = bottom[3]->gpu_data(); // n_roi * all_pnt_num * 1 * 1
    const int* cls_ch = class_channel_.gpu_data();
	Dtype* top_data = top[0]->mutable_gpu_data();
	int* argmax_data = max_idx_.mutable_gpu_data();
	int count = top[0]->count();
    caffe_gpu_set(count, Dtype(0), top_data);
	// NOLINT_NEXT_LINE(whitespace/operators)
	if (use_maxpool_)
        MaxPointPoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
		  count, bottom_data, ncls_, channels_, height_, width_,
		  cls_ch, spatial_scale_, bottom_ids, bottom_points, bottom_points_valid, top_data, argmax_data);
	else
        AvePointPoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, bottom_data, ncls_, channels_, height_, width_,
          cls_ch, spatial_scale_, bottom_ids, bottom_points, bottom_points_valid, top_data);
	CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void MaxPointPoolBackward(const int nthreads, const Dtype* top_diff,
    const int* argmax_data, const int num_rois, const int ncls, const int channels,
    const int height, const int width, const int* cls_ch, const Dtype spatial_scale,
    Dtype* bottom_diff, const Dtype* bottom_ids, const Dtype* bottom_points, const Dtype* bottom_points_valid) {
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
        const Dtype* valid = bottom_points_valid + n * channels + ch;
        bool is_valid = valid[0];
        if (!is_valid) {
            ch_len--; // the point is absent
        }
    }

    for (int ch = cls_ch[0]; ch <= cls_ch[1]; ch++) {
		Dtype* diff = bottom_diff + (roi_batch_ind * channels + ch) * height * width;
        int ind = argmax_data[ch];
        if (ind > -1)
            diff[ind] += top_diff[index] / ch_len; 
	}
  }
}

template <typename Dtype>
__global__ void AvePointPoolBackward(const int nthreads, const Dtype* top_diff,
    const int num_rois, const int ncls, const int channels,
    const int height, const int width, const int* cls_ch, const Dtype spatial_scale,
    Dtype* bottom_diff, const Dtype* bottom_ids, const Dtype* bottom_points, const Dtype* bottom_points_valid) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, cls, 1, 1) is an element in the pooled output, n represents roi id
    int cls = index % ncls;
    int n = index / ncls;

    cls_ch += cls * 2;
    bottom_ids += n;
    int roi_batch_ind = bottom_ids[0];
    int ch_len = cls_ch[1] - cls_ch[0] + 1;
	
    for (int ch = cls_ch[0]; ch <= cls_ch[1]; ch++) {
        const Dtype* valid = bottom_points_valid + n * channels + ch;
        bool is_valid = valid[0];
        if (!is_valid) {
            ch_len--; // the point is absent
        }
    }
    
	for (int ch = cls_ch[0]; ch <= cls_ch[1]; ch++) {
		Dtype* diff = bottom_diff + (roi_batch_ind * channels + ch) * height * width;
		const Dtype* pnt = bottom_points + (n * channels + ch) * 4;
		const Dtype* valid = bottom_points_valid + n * channels + ch;
        bool is_valid = valid[0];
        if (!is_valid) {
            continue; // the point is absent
        }
        int x1 = round(pnt[0] * spatial_scale);
        int y1 = round(pnt[1] * spatial_scale);
        int x2 = round(pnt[2] * spatial_scale);
        int y2 = round(pnt[3] * spatial_scale);
        x1 = min(max(x1, 0), width);
        y1 = min(max(y1, 0), height);
        x2 = min(max(x2, 0), width);
        y2 = min(max(y2, 0), height);
		for (int h=y1; h<=y2; h++) {
			for (int w=x1; w<=x2; w++) {
				int ind = h * width + w;
				diff[ind] += top_diff[index] / ch_len / (y2 - y1 + 1) / (x2 - x1 + 1);
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
    const Dtype* bottom_points_valid = bottom[3]->gpu_data(); // n_roi * all_pnt_num * 1 * 1
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
		  height_, width_, cls_ch, spatial_scale_, bottom_diff, bottom_ids, bottom_points, bottom_points_valid);
	else
		AvePointPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
		  count, top_diff, top[0]->num(), ncls_, channels_, height_, width_, 
		  cls_ch, spatial_scale_, bottom_diff, bottom_ids, bottom_points, bottom_points_valid);
	CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(PointPoolingLayer);

}  // namespace caffe
