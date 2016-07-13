// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// ------------------------------------------------------------------

#include <cfloat>
#include <fstream>
#include <string>

#include "caffe/frcnn/layers/point_pooling_layer.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {

template <typename Dtype>
void PointPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  PointPoolingParameter point_pool_param = this->layer_param_.point_pooling_param();
  if (point_pool_param.method() == PointPoolingParameter_PoolMethod_MAX)
    use_maxpool_ = true;
  else if (point_pool_param.method() == PointPoolingParameter_PoolMethod_AVE)
    use_maxpool_ = false;
  else
    LOG(FATAL) << "Unknown pooling method.";
  string config = point_pool_param.config();
  std::ifstream infile(config.c_str());
  infile >> ncls_;
  class_channel_.Reshape(ncls_, 2, 1, 1);
  int* cls_ch = class_channel_.mutable_cpu_data();
  int tmp;
  cls_ch[0] = 0;
  infile >> tmp;
  cls_ch[1] = tmp - 1;
  int pre = cls_ch[1];
  for (int i=1; i<ncls_; i++) {
    cls_ch += class_channel_.offset(1);
    cls_ch[0] = pre + 1;
    infile >> tmp;
    cls_ch[1] = pre + tmp;
    pre = cls_ch[1];
  }
  infile.close();
  for (int i=0; i<ncls_; i++) {
    cls_ch = class_channel_.mutable_cpu_data() + class_channel_.offset(i);
    CHECK_GE(cls_ch[1], cls_ch[0]) << "Channel id must be ascending order.";
  }
}

template <typename Dtype>
void PointPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  channels_ = bottom[0]->channels();
  const int* cls_ch = class_channel_.cpu_data();
  CHECK_EQ(cls_ch[ncls_*2-1], channels_-1) << "Config class channel ids don't match blob channel.";
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  top[0]->Reshape(bottom[1]->num(), ncls_, 1, 1);
  max_idx_.Reshape(bottom[1]->num(), channels_, 1, 1);
}

template <typename Dtype>
void PointPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_ids = bottom[1]->cpu_data(); // n * 1 * 1 * 1
  const Dtype* bottom_points = bottom[2]->cpu_data();
  // Number of ROIs
  int num_rois = bottom[1]->num();
  int batch_size = bottom[0]->num();
  int top_count = top[0]->count();
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_set(top_count, Dtype(0), top_data);
  int idx_count = max_idx_.count();
  int* argmax_data = max_idx_.mutable_cpu_data();
  caffe_set(idx_count, -1, argmax_data);

  // For each point P = (x, y): ave/max pool over the neighborhood of P on the corresponding channel
  for (int n = 0; n < num_rois; ++n) {
    int roi_batch_ind = bottom_ids[0];
    CHECK_GE(roi_batch_ind, 0);
    CHECK_LT(roi_batch_ind, batch_size);

	for (int i=0; i<ncls_; i++) {
        const int* cls_ch = class_channel_.cpu_data() + class_channel_.offset(i);
		for (int ch = cls_ch[0]; ch <= cls_ch[1]; ch++) {
			const Dtype* feat_map = bottom_data + bottom[0]->offset(roi_batch_ind, ch);
			const Dtype* pnt = bottom_points + bottom[2]->offset(0, ch);
			int x1 = round(pnt[0]);
			int y1 = round(pnt[1]);
			int x2 = round(pnt[2]);
			int y2 = round(pnt[3]);
            if (x2 == -1) {
                continue; // if set -1, the point is absent
            }
			if (use_maxpool_) {
				Dtype maxval = -FLT_MAX;
				for (int h=y1; h<=y2; h++) {
					for (int w=x1; w<=x2; w++) {
						int ind = h * width_ + w;
						if (feat_map[ind] > maxval) {
							maxval = feat_map[ind];
							argmax_data[ch] = ind;
						}
					}
				}
				top_data[i] += maxval;
			} 
			else {
				Dtype avgval = 0;
				for (int h=y1; h<=y2; h++) {
					for (int w=x1; w<=x2; w++) {
						int ind = h * width_ + w;
						avgval += feat_map[ind];
					}
				}
				top_data[i] += avgval / (y2 - y1 + 1) / (x2 - x1 + 1);
			}
		} // for ch
		top_data[i] /= cls_ch[1] - cls_ch[0] + 1;
	} // for i
    // Increment blob pointer
    bottom_ids += bottom[1]->offset(1);
	bottom_points += bottom[2]->offset(1);
	top_data += top[0]->offset(1);
	argmax_data += max_idx_.offset(1);
  } // for n
}

template <typename Dtype>
void PointPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}


#ifdef CPU_ONLY
STUB_GPU(PointPoolingLayer);
#endif

INSTANTIATE_CLASS(PointPoolingLayer);
REGISTER_LAYER_CLASS(PointPooling);

}  // namespace caffe
