#include <vector>

#include "caffe/layers/delta_to_coordinate_layer.hpp"

namespace caffe {
  // bottom[0]: cls map
  // bottom[1]: delta map
  // bottom[2]: anchor rect
  // bottom[3]: anchor pts
  // bottom[4]: mask
  // bottom[5]: label map
  // top[0]: pts coordinate
  // top[1]: score
  // top[2]: label

template <typename Dtype>
void DeltaToCoordinateLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	thresh_ = this->layer_param_.delta_to_coordinate_param().threshold();
	stride_ = this->layer_param_.delta_to_coordinate_param().stride();
	anchor_bias_ = this->layer_param_.delta_to_coordinate_param().anchor_bias();
}

template <typename Dtype>
void DeltaToCoordinateLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	CHECK_EQ(bottom[ 0 ]->count() / bottom[ 0 ]->channels(), bottom[ 1 ]->count() / bottom[ 1 ]->channels());
  CHECK_EQ(bottom[ 0 ]->channels(), 1) << "Only support one channels for now";
  CHECK_EQ(bottom[ 0 ]->num(), 1) << "Only support one num for now";
  CHECK_EQ(bottom[2]->count(), 4) << "Anchor rect should has 4 elements [l u r b]";
  CHECK_EQ(bottom[3]->count(), bottom[1]->channels()) << "Anchor points should have same dimension with delta";
  CHECK_EQ(bottom[ 3 ]->count() % 2, 0) << "Points number should be even number!";
  CHECK_EQ(bottom[ 4 ]->channels(), 1) << "Mask size should be [1 1 h w]";
  CHECK_EQ(bottom[ 0 ]->count(), bottom[ 4 ]->count());
  CHECK_EQ(bottom[ 0 ]->count(), bottom[ 5 ]->count());
  const Dtype* clsmap = bottom[ 0 ]->cpu_data();
  const Dtype* mask = bottom[ 4 ]->cpu_data();
  int tot = bottom[0]->count();
  w_ = bottom[ 0 ]->width();
  h_ = bottom[ 0 ]->height();
  valid_num_ = 0;
  for (int i = 0; i < tot; ++i){
	  if ( clsmap[ i ] >= thresh_ && mask[ i ] > 0 )
		++valid_num_;
  }
  top[ 0 ]->Reshape(valid_num_, bottom[ 1 ]->channels(), 1, 1);
  top[ 1 ]->Reshape(valid_num_, 1, 1, 1);
  top[ 2 ]->Reshape(valid_num_, 1, 1, 1);

}

template <typename Dtype>
void DeltaToCoordinateLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	if ( valid_num_ < 1 )
		return;
	const Dtype* clsmap = bottom[ 0 ]->cpu_data();
	const Dtype* deltamap = bottom[ 1 ]->cpu_data();
	const Dtype* anchor_rect = bottom[ 2 ]->cpu_data();
	const Dtype* anchor_pts = bottom[ 3 ]->cpu_data();
	const Dtype* mask = bottom[ 4 ]->cpu_data();
	const Dtype* labelmap = bottom[ 5 ]->cpu_data();
	Dtype* pts_coordinate = top[ 0 ]->mutable_cpu_data();
	Dtype* score = top[ 1 ]->mutable_cpu_data();
	Dtype* label = top[ 2 ]->mutable_cpu_data();
	int cnt = 0;
	std::vector<Dtype> anchor_box_size(2);
	std::vector<Dtype> anchor_center_now(2);
	std::vector<Dtype> anchor_pts_now(bottom[ 1 ]->channels());

	anchor_box_size[ 0 ] = anchor_rect[ 2 ] - anchor_rect[ 0 ];
	anchor_box_size[ 0 ] = anchor_rect[ 3 ] - anchor_rect[ 1 ];

	for ( int i = 0; i < bottom[ 0 ]->count(); ++i ){
		if ( clsmap[ i ] >= thresh_ && mask[i] > 0){
			int y = floor(i / w_);
			int x = i % w_;
			anchor_center_now[ 0 ] = x*stride_ + anchor_bias_;
			anchor_center_now[ 1 ] = y*stride_ + anchor_bias_;
			for ( int j = 0; j < bottom[ 1 ]->channels() / 2; ++j ){
				anchor_pts_now[ j * 2 ] = anchor_pts[ j * 2 ] * anchor_box_size[ 0 ] + anchor_center_now[ 0 ];
				anchor_pts_now[ j * 2 + 1 ] = anchor_pts[ j * 2 + 1 ] * anchor_box_size[ 1 ] + anchor_center_now[ 1 ];
				pts_coordinate[ cnt*top[ 0 ]->channels() + j * 2 ] = deltamap[ j * 2 * w_*h_ + y*w_ + x ] * anchor_box_size[ 0 ] + anchor_pts_now[ j * 2 ];
				pts_coordinate[ cnt*top[ 0 ]->channels() + j * 2 + 1 ] = deltamap[ ( j * 2 + 1 ) * w_*h_ + y*w_ + x ] * anchor_box_size[ 1 ] + anchor_pts_now[ j * 2 + 1 ];
			}
			score[ cnt ] = clsmap[ i ];
			label[ cnt ] = labelmap[ i ];
			++cnt;
		}
	}
}

template <typename Dtype>
void DeltaToCoordinateLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down,
	const vector<Blob<Dtype>*>& bottom) {
	if ( valid_num_ < 1 )
		return;

}

INSTANTIATE_CLASS(DeltaToCoordinateLayer);
REGISTER_LAYER_CLASS(DeltaToCoordinate);

}  // namespace caffe
