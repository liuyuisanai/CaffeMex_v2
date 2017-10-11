#include <vector>

#include "caffe/layers/balanceable_sigmoid_cross_entropy_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void BSCELossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  b_cls_ = this->layer_param().bsce_param().balance_cls();
  b_his_ = this->layer_param().bsce_param().balance_his();
  bin_num_ = this->layer_param().bsce_param().bin_num();
  sigmoid_bottom_vec_.clear();
  sigmoid_bottom_vec_.push_back(bottom[0]);
  sigmoid_top_vec_.clear();
  sigmoid_top_vec_.push_back(sigmoid_output_.get());
  sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);
  has_ignore_label_ =
	  this->layer_param_.loss_param().has_ignore_label();
  if ( has_ignore_label_ ) {
	  ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
 else
  ignore_label_ = -1;
  valid_num_ = 0;
  his_stat_.Reshape(bin_num_, 1, 1, 1);
  cls_stat_.Reshape(2, 1, 1, 1);
  cls_grad_.Reshape(2, 1, 1, 1);

}

template <typename Dtype>
void BSCELossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
      "SIGMOID_CROSS_ENTROPY_LOSS layer inputs must have the same count.";
  sigmoid_layer_->Reshape(sigmoid_bottom_vec_, sigmoid_top_vec_);
  valid_num_ = 0;
  for ( int i = 0; i < bin_num_; i++ ){
	  his_stat_.mutable_cpu_data()[ i ] = 0;
  }
  for ( int i = 0; i < 2; i++ ){
	  cls_stat_.mutable_cpu_data()[ i ] = 0;
	  cls_grad_.mutable_cpu_data()[ i ] = 0;
  }
}

template <typename Dtype>
void BSCELossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	if ( bottom[ 0 ]->count() < 1 ){
			top[ 0 ]->mutable_cpu_data()[ 0 ] = Dtype(0);
			return;
		}
	cls_stat_.mutable_cpu_data()[ 1 ] = caffe_cpu_asum(bottom[ 1 ]->count(), bottom[ 1 ]->cpu_data());
	cls_stat_.mutable_cpu_data()[ 0 ] = bottom[ 1 ]->count() - cls_stat_.cpu_data()[ 1 ];
  // The forward pass computes the sigmoid outputs.
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  // Compute the loss (negative log likelihood)
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  // Stable version of loss computation from input data
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* target = bottom[1]->cpu_data();
  Dtype loss = 0;
  for (int i = 0; i < count; ++i) {
	  if ( target[ i ] == ignore_label_ )
		  continue;
	  else{
		  valid_num_+=1;
		  loss -= input_data[ i ] * ( target[ i ] - ( input_data[ i ] >= 0 ) ) -
			  log(1 + exp(input_data[ i ] - 2 * input_data[ i ] * ( input_data[ i ] >= 0 )));
	  }
  }
  if ( valid_num_ < 1 )
	  top[ 0 ]->mutable_cpu_data()[ 0 ] = 0;
  else
	top[ 0 ]->mutable_cpu_data()[ 0 ] = loss / valid_num_;
  if ( top.size() >= 2 ) {
	  for ( int i = 0; i < count; ++i ){
		  if ( target[ i ] == ignore_label_ ){
			  top[ 1 ]->mutable_cpu_data()[ i ] = 0;
		  }
		  else{
			  top[ 1 ]->mutable_cpu_data()[ i ] = -( input_data[ i ] * ( target[ i ] - ( input_data[ i ] >= 0 ) ) -
				  log(1 + exp(input_data[ i ] - 2 * input_data[ i ] * ( input_data[ i ] >= 0 ))) );
			  // Output per-instance loss
		  }
	  }
  }
}
template <typename Dtype>
void BSCELossLayer<Dtype>::BSCE_statistics(const Dtype*bottom_diff, int count, const Dtype* target
	) {
	Dtype sums = 0.0;
	for ( int i = 0; i < count; ++i )
	{
		his_stat_.mutable_cpu_data()[ (int)floor(fabs(bottom_diff[ i ]) * bin_num_) ] += (Dtype)( 1.0 );
		cls_grad_.mutable_cpu_data()[ (int)target[ i ] ] += fabs(bottom_diff[ i ]);
	}
	for ( int i = 0; i < bin_num_; ++i ){
		his_stat_.mutable_cpu_data()[ i ] = count / ( bin_num_ * his_stat_.cpu_data()[ i ] + 1e-7);
	}
	for ( int i = 0; i < 2; ++i ){
		cls_grad_.mutable_cpu_data()[ i ] = ( cls_grad_.cpu_data()[ 0 ] + cls_grad_.cpu_data()[ 1 ] ) / ( 2 * cls_grad_.cpu_data()[ i ] + 1e-7 );
	}
}


template <typename Dtype>
void BSCELossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
	  if ( bottom[ 0 ]->count() < 1 ){
		  return;
	  }
    // First, compute the diff
    const int count = bottom[0]->count();
    const int num = bottom[0]->num();
    const Dtype* sigmoid_output_data = sigmoid_output_->cpu_data();
    const Dtype* target = bottom[1]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_sub(count, sigmoid_output_data, target, bottom_diff);
    // Scale down gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
	caffe_scal(count, loss_weight / valid_num_, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU_BACKWARD(BSCELossLayer, Backward);
#endif

INSTANTIATE_CLASS(BSCELossLayer);
REGISTER_LAYER_CLASS(BSCELoss);

}  // namespace caffe
