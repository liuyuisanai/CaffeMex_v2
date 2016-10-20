#include <vector>

#include "caffe/layers/point_focus_layer.hpp"

namespace caffe {
  // bottom[0]: pts matrix
  // bottom[1]: pts score map
  // top[0]: pts score matrix

template <typename Dtype>
void PointFocusLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	stride_ = this->layer_param_.delta_to_coordinate_param().stride();
}

template <typename Dtype>
void PointFocusLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels()*2) << "Points number and channels of feature map mismatch";
  top[ 0 ]->Reshape(bottom[0]->num(), bottom[ 1 ]->channels(), 1, 1);
}

template <typename Dtype>
void PointFocusLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	if ( bottom[ 0 ]->count() < 1 )
		return;
	const Dtype* pts = bottom[ 0 ]->cpu_data();
	const Dtype* fm_in = bottom[ 1 ]->cpu_data();
	Dtype* fm_out = top[ 0 ]->mutable_cpu_data();
	int w = bottom[ 1 ]->width();
	int h = bottom[ 1 ]->height();
	for ( int i = 0; i < bottom[ 0 ]->num(); ++i ){
		for (int j = 0; j < bottom[ 1 ]->channels(); ++j){
			int x = round((pts[i*bottom[ 0 ]->channels()+2*j]-(stride_-1.0)/2.0)/stride_);
			int y = round(( pts[ i*bottom[ 0 ]->channels() + 2 * j + 1 ] - ( stride_ - 1.0 ) / 2.0 ) / stride_);
			//LOG(INFO) << "x=" << x << " y=" << y << ", stride_=" << stride_ << ", i=" << i << " j=" << j << ", j*w*h+y*w+x=" << j*w*h + y*w + x;
			x = std::max(std::min(x, w - 1), 0);
			y = std::max(std::min(y, h - 1), 0);
			fm_out[i*bottom[ 1 ]->channels()+j] = fm_in[j*w*h+y*w+x];
		}
	}
}

template <typename Dtype>
void PointFocusLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down,
	const vector<Blob<Dtype>*>& bottom) {
	if ( bottom[ 0 ]->count() < 1 )
		return;
	const Dtype* pts = bottom[ 0 ]->cpu_data();
	Dtype* fm_in_diff = bottom[ 1 ]->mutable_cpu_diff();
	const Dtype* fm_out_diff = top[ 0 ]->cpu_diff();
	caffe_set(bottom[ 1 ]->count(), Dtype(0), fm_in_diff);
	int w = bottom[ 1 ]->width();
	int h = bottom[ 1 ]->height();
	for ( int i = 0; i < bottom[ 0 ]->num(); ++i ){
		for (int j = 0; j < bottom[ 1 ]->channels(); ++j){
			int x = round((pts[i*bottom[ 0 ]->channels()+2*j]-(stride_-1.0)/2.0)/stride_);
			int y = round((pts[i*bottom[ 0 ]->channels()+2*j+1]-(stride_-1.0)/2.0)/stride_);
			x = std::max(std::min(x, w - 1), 0);
			y = std::max(std::min(y, h - 1), 0);
			fm_in_diff[ j*w*h + y*w + x ] = fm_out_diff[ i*bottom[ 1 ]->channels() + j ];
		}
	}
}

INSTANTIATE_CLASS(PointFocusLayer);
REGISTER_LAYER_CLASS(PointFocus);

}  // namespace caffe
