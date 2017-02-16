#include <algorithm>
#include <cfloat>
#include <vector>

#include "thrust/device_vector.h"

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/normalize_layer.hpp"

namespace caffe {


	template <typename Dtype>
	__global__ void NormalizeForward(const int n, const Dtype* in, Dtype* out, int length){
		CUDA_KERNEL_LOOP(index, n) {
			Dtype sum = 0;
			for ( int i = 0; i < length; i++ )
			{
				sum = sum + in[ index*length + i ] * in[ index*length + i ];
			}
			sum = sqrt(sum + 0.000000001);
			for ( int i = 0; i < length; i++ )
			{
				out[ index*length + i ] = in[ index*length + i ] / sum;
			}
		}
	}
	template <typename Dtype>
	__global__ void NormalizeBackward(const int n, const Dtype* x, const Dtype* y, const Dtype* dy, Dtype* dx, int length){
		CUDA_KERNEL_LOOP(index, n) {
			Dtype sum = 0;
			Dtype ip = 0;
			for ( int i = 0; i < length; i++ )
			{
				int ind = index*length + i;
				sum += x[ ind ] * x[ ind ];
				ip += y[ ind ] * dy[ ind ];
			}
			sum = sqrt(sum + 0.000000001);
			for ( int i = 0; i < length; i++ )
			{
				int ind = index * length + i;
				dx[ ind ] = ( dy[ ind ] - ip*y[ ind ] ) / sum;
			}
		}
	}

	template <typename Dtype>
	void NormalizeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const Dtype* bottom_data = bottom[ 0 ]->gpu_data();
		Dtype* top_data = top[ 0 ]->mutable_gpu_data();
		//Dtype* squared_data = squared_.mutable_gpu_data();
		//Dtype normsqr;
		int n = bottom[ 0 ]->num();
		int d = bottom[ 0 ]->count() / n;
		/*caffe_gpu_powx(n*d, bottom_data, Dtype(2), squared_data);
		for (int i=0; i<n; ++i) {
		caffe_gpu_asum<Dtype>(d, squared_data+i*d, &normsqr);
		DLOG(INFO) << "1. asum " << pow(normsqr/Dtype(d), 0.5);
		caffe_gpu_scale<Dtype>(d, pow(normsqr + Dtype(1e-12), -0.5), bottom_data+i*d, top_data+i*d);
		DLOG(INFO) << "2. scale";
		caffe_gpu_scale<Dtype>(d, rescale_coeff_, top_data+i*d, top_data+i*d);
		DLOG(INFO) << "3. scale";
		}*/
		NormalizeForward<Dtype> << <CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS >> >( n, bottom_data, top_data, d );
		CUDA_POST_KERNEL_CHECK;
	}

	template <typename Dtype>
	void NormalizeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		const Dtype* top_diff = top[ 0 ]->gpu_diff();
		const Dtype* top_data = top[ 0 ]->gpu_data();
		const Dtype* bottom_data = bottom[ 0 ]->gpu_data();
		Dtype* bottom_diff = bottom[ 0 ]->mutable_gpu_diff();
		int n = top[ 0 ]->num();
		int d = top[ 0 ]->count() / n;
		//Dtype a;
		//for (int i = 0; i<n; ++i) {
		//	DLOG(INFO) << "Backward start";
		//	caffe_gpu_scale<Dtype>(d, rescale_coeff_, top_diff + i*d, top_diff + i*d);
		//	caffe_gpu_scale<Dtype>(d, Dtype(1.0) / rescale_coeff_, top_data + i*d, top_data + i*d);
		//	caffe_gpu_dot<Dtype>(d, top_data + i*d, top_diff + i*d, &a);
		//	caffe_gpu_scale<Dtype>(d, a, top_data + i*d, bottom_diff + i*d);
		//	caffe_gpu_sub<Dtype>(d, top_diff + i*d, bottom_diff + i*d, bottom_diff + i*d);
		//	caffe_gpu_dot<Dtype>(d, bottom_data + i*d, bottom_data + i*d, &a);
		//	caffe_gpu_scale<Dtype>(d, Dtype(pow(a + Dtype(1e-12), -0.5)), bottom_diff + i*d, bottom_diff + i*d);
		//	//caffe_gpu_scale<Dtype>(d, rescale_coeff_, bottom_diff+i*d, bottom_diff+i*d);
		//	DLOG(INFO) << "Backward end";
		//}
		NormalizeBackward<Dtype> << <CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS >> >( n, bottom_data, top_data, top_diff, bottom_diff, d );
	}

	INSTANTIATE_LAYER_GPU_FUNCS(NormalizeLayer);


}  // namespace caffe
