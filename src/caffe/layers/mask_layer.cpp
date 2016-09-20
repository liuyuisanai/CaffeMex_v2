#include <vector>

#include "caffe/layers/mask_layer.hpp"

namespace caffe {

	template <typename Dtype>
	void MaskLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		CHECK_NE(top[ 0 ], bottom[ 0 ]) << this->type() << " Layer does not "
			"allow in-place computation.";
	}

	template <typename Dtype>
	void MaskLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		CHECK_EQ(bottom[0]->num(), bottom[1]->num()) << "num not match.";
		CHECK_EQ(bottom[ 0 ]->width(), bottom[ 1 ]->width()) << "width not match.";
		CHECK_EQ(bottom[ 0 ]->height(), bottom[ 1 ]->height()) << "height not match.";
		CHECK_EQ(bottom[ 1 ]->channels(), 1) << "mask's channel should be 1.";
		CHECK_EQ(bottom[ 1 ]->num(), 1) << "mask's num should be 1.";
		n_ = bottom[ 0 ]->num();
		c_ = bottom[ 0 ]->channels();
		w_ = bottom[ 0 ]->width();
		h_ = bottom[ 0 ]->height();
		vector<int> top_shape;
		Dtype* mask = bottom[ 1 ]->cpu_data();
		validnum_ = caffe_cpu_asum(bottom[ 1 ]->count(), mask);
		top[ 0 ]->Reshape(n_, c_, validnum_, 1);
		pass_idx_.Reshape(1, 1, validnum_, 1);
	}

	template <typename Dtype>
	void MaskLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		Dtype* mask = bottom[ 1 ]->cpu_data();
		Dtype* bottom_data = bottom[ 0 ]->cpu_data();
		Dtype* top_data = top[ 0 ]->mutable_cpu_data();
		int idx = 0;
		for ( int y = 0; y < h_; y++ ){
			for ( int x = 0; x < w_; x++ ){
				if ( mask[y*w_ + x] >= 1 )
				{
					for ( int c = 0; c < c_; c++ )
					{
						top_data[ c*validnum_ + idx ] = bottom_data[ c*w_*h_ + y*w_ + x ];
					}
				}
				else{

				}
			}
		}
	}

	template <typename Dtype>
	void MaskLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		Dtype* bottom_diff = bottom[ 0 ]->cpu_diff();
		Dtype* mask = bottom[ 1 ]->cpu_data();
		Dtype* top_diff = top[ 0 ]->cpu_diff();
		int idx = 0;
		for ( int y = 0; y < h_; y++ ){
			for ( int x = 0; x < w_; x++ ){
				if ( mask[ y*w_ + x ] >= 1 )
				{
					for ( int c = 0; c < c_; c++ )
					{
						bottom_diff[ c*w_*h_ + y*w_ + x ] = top_diff[ c*validnum_ + idx ];
					}
				}
				else{

				}
			}
		}
	}

	INSTANTIATE_CLASS(MaskLayer);
	REGISTER_LAYER_CLASS(Mask);

}  // namespace caffe
