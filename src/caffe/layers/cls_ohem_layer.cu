
#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/cls_ohem_layer.hpp"

using std::max;
using std::min;

namespace caffe {
	template <typename Dtype>
	void ClsOHEMLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		if (bottom[0]->count() < 1)
			return;
		const Dtype* bottom_labels = bottom[0]->gpu_data();
		const Dtype* bottom_loss = bottom[1]->gpu_data();
		Dtype* top_labels = top[0]->mutable_gpu_data();
		Dtype* top_loss = top[1]->mutable_gpu_data();
		caffe_gpu_set(top[0]->count(), Dtype(ignore_label_), top_labels);
		caffe_gpu_set(top[1]->count(), Dtype(0), top_loss);
		int cur_channels_ = bottom[0]->channels();
		int num_labels_ = bottom[0]->count() / cur_channels_;

		bottom_labels = bottom[0]->gpu_data();
		bottom_loss = bottom[1]->gpu_data();
		// Find rois with max loss
		vector<int> sorted_idx(num_labels_);
		for (int i = 0; i < num_labels_; i++){
			sorted_idx[i] = i;
		}
		std::sort(sorted_idx.begin(), sorted_idx.end(),
			[bottom_loss](int i1, int i2){return bottom_loss[i1] > bottom_loss[i2]; });

		// Generate output labels for scoring and loss_weights for bbox regression

		int choose = min(topk_, num_labels_);
		for (int i = 0; i < num_labels_; i++){
			if (bottom_loss[i] >= bottom_loss[sorted_idx[choose]]){
				top_labels[i] = bottom_labels[i];
			}
			top_loss[i] = bottom_loss[i]; 
		}
	}

	template <typename Dtype>
	void ClsOHEMLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		return;
	}

	INSTANTIATE_LAYER_GPU_FUNCS(ClsOHEMLayer);

}  // namespace caffe
