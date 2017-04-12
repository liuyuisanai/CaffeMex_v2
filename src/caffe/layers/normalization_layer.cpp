#include <vector>
#include <cmath>

#include "caffe/util/math_functions.hpp"
#include "caffe/layers/normalization_layer.hpp"

namespace caffe {

	// template <typename Dtype>
	// void NormalizationLayer<Dtype>::LayerSetUp(
	//   const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	//   Layer<Dtype>::LayerSetUp(bottom, top);
	// }

	template <typename Dtype>
	void NormalizationLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		// Layer<Dtype>::Reshape(bottom, top);
		top[0]->ReshapeLike(*bottom[0]);
		squared_.ReshapeLike(*bottom[0]);
		// top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(),
		//     bottom[0]->height(), bottom[0]->width());
		// squared_.Reshape(bottom[0]->num(), bottom[0]->channels(), 
		//   bottom[0]->height(), bottom[0]->width());
	}

	template <typename Dtype>
	void NormalizationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const Dtype* bottom_data = bottom[0]->cpu_data();
		Dtype* top_data = top[0]->mutable_cpu_data();
		Dtype* squared_data = squared_.mutable_cpu_data();
		int n = bottom[0]->num();
		int d = bottom[0]->count() / n;
		Dtype normabs = caffe_cpu_asum<Dtype>(n, bottom_data);
		switch (this->layer_param_.normalization_param().norm()) {
		case NormalizationParameter_Norm_L2:
			//L2

			caffe_sqr<Dtype>(n*d, bottom_data, squared_data);
			for (int i = 0; i<n; ++i) {
				Dtype normsqr = caffe_cpu_asum<Dtype>(d, squared_data + i*d);
				caffe_cpu_scale<Dtype>(d, pow(normsqr, -0.5), bottom_data + i*d, top_data + i*d);
			}
			break;
		case NormalizationParameter_Norm_L1:
			//modify L1

			caffe_cpu_scale<Dtype>(n, pow(normabs, -1), bottom_data, top_data);
			break;

		}
	}

	template <typename Dtype>
	void NormalizationLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		const Dtype* top_diff = top[0]->cpu_diff();
		const Dtype* top_data = top[0]->cpu_data();
		const Dtype* bottom_data = bottom[0]->cpu_data();
		const Dtype* bottom_data_1 = bottom[0]->cpu_data();
		const int count = bottom[0]->count();
		int channels_ = bottom[0]->channels();
		int height_ = bottom[0]->height();
		int width_ = bottom[0]->width();
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
		int n = top[0]->num();
		int d = top[0]->count() / n;
		Dtype normabs = caffe_cpu_asum<Dtype>(n, bottom_data);
		Dtype a=0;
		switch (this->layer_param_.normalization_param().norm()) {
			//L2
		case NormalizationParameter_Norm_L2:
			//modify by songgl
			for (int i = 0; i<n; ++i) {

		
				a = caffe_cpu_dot(d, top_data + i*d, top_diff + i*d);
				caffe_cpu_scale(d, a, top_data + i*d, bottom_diff + i*d);
				caffe_sub(d, top_diff + i*d, bottom_diff + i*d, bottom_diff + i*d);
				a = caffe_cpu_dot(d, bottom_data + i*d, bottom_data + i*d);
				caffe_cpu_scale(d, Dtype(pow(a, -0.5)), bottom_diff + i*d, bottom_diff + i*d);
			}
			break;
		case NormalizationParameter_Norm_L1:
			
			a = caffe_cpu_dot(n, top_data, top_diff);
		//	LOG_IF(INFO, Caffe::root_solver())
		//		<< "Creating training net from net file: " << param_.net();
		//	LOG(INFO) << "debug info here---:top_diff*top_data: " <<a;
			caffe_set(n, a, bottom_diff);
			caffe_sub(n, top_diff, bottom_diff, bottom_diff);
		//	LOG(INFO) << "debug info here---:asum: " << normabs;
		//	LOG(INFO) << "debug info here---:top_diff[0]: " << top_diff[0];
			caffe_cpu_scale(n, Dtype(pow(normabs, -1)), bottom_diff, bottom_diff);
			break;
		}
	}


#ifdef CPU_ONLY
	STUB_GPU(NormalizationLayer);
#endif

	INSTANTIATE_CLASS(NormalizationLayer);
	REGISTER_LAYER_CLASS(Normalization);

}  // namespace caffe