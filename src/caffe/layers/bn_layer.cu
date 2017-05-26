#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/bn_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

	template <typename Dtype>
	void BNLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const Dtype* const_bottom_data = bottom[0]->gpu_data();
		const Dtype* const_top_data = top[0]->gpu_data();
		Dtype* top_data = top[0]->mutable_gpu_data();

		const Dtype* scale_data = this->blobs_[0]->gpu_data();
		const Dtype* shift_data = this->blobs_[1]->gpu_data();

		// ---------- mean subtraction ---------- //
      // statistic across spatial
      caffe_gpu_gemv<Dtype>(CblasNoTrans, num_ * channels_, height_ * width_, Dtype(1. / (height_ * width_)), const_bottom_data,
                            spatial_sum_multiplier_.gpu_data(), Dtype(0), spatial_statistic_.mutable_gpu_data());
      // statistic across batch
      caffe_gpu_gemv<Dtype>(CblasTrans, num_, channels_, Dtype(1. / num_), spatial_statistic_.gpu_data(),
		  batch_sum_multiplier_.gpu_data(), Dtype(0), ex_.mutable_gpu_data());
		if (this->phase_ == TRAIN) {
			// sync statistics
			if ( sync_forward_ ){
				// first, sync EX
				caffe_copy(channels_, ex_.gpu_data(), statistics_all_.mutable_gpu_data());
				P2PSync<Dtype>* p2p = this->callbacks()[ 0 ]->callbacks()[ 0 ]->p2p()[ 0 ];
				Blob<Dtype> statistics_child(1, channels_, 1, 1);
				for ( int i = 0; i < p2p->children().size(); ++i ){
#ifdef _WIN64
					Blob<Dtype>* s_c_ogpu = p2p->dataQueue().pop();
#else
					Blob<Dtype>* s_c_ogpu = NULL;
					while ( !p2p->dataQueue().try_pop(&s_c_ogpu) )
						;
#endif
					//Blob<Dtype>* s_c_ogpu = p2p->dataQueue().pop();
					CUDA_CHECK(cudaMemcpyAsync(statistics_child.mutable_gpu_data(), s_c_ogpu->gpu_data(), channels_*sizeof( Dtype ), cudaMemcpyDeviceToDevice, cudaStreamDefault));
					CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
					caffe_gpu_add(channels_, statistics_child.gpu_data(), statistics_all_.gpu_data(), statistics_all_.mutable_gpu_data());
				}
				if ( p2p->parent() ){
					p2p->parent()->dataQueue().push(&statistics_all_);
#ifdef _WIN64
					Blob<Dtype>* statistics_final = p2p->dataQueue().pop();
#else
					Blob<Dtype>* statistics_final = NULL;
					while ( !p2p->dataQueue().try_pop(&statistics_final) )
						;
#endif
					//Blob<Dtype>* statistics_final = p2p->dataQueue().pop();
					CUDA_CHECK(cudaMemcpyAsync(ex_.mutable_gpu_data(), statistics_final->gpu_data(), channels_*sizeof( Dtype ), cudaMemcpyDeviceToDevice, cudaStreamDefault));
					CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
				}
				else {
					caffe_gpu_scal<Dtype>(channels_, Dtype(1.0 / Caffe::solver_count()), statistics_all_.mutable_gpu_data());
					caffe_copy(channels_, statistics_all_.gpu_data(), ex_.mutable_gpu_data());
				}
				for ( int i = 0; i < p2p->children().size(); ++i ){
					p2p->children()[ i ]->dataQueue().push(&ex_);
				}
			}
			// save history mean
			caffe_gpu_axpby(ex_.count(), Dtype(1) - decay_, ex_.gpu_data(), decay_,
			    this->blobs_[2]->mutable_gpu_data());
		}
		if (this->phase_ == TEST && moving_average_) {
			// use moving average mean
			caffe_copy(ex_.count(), this->blobs_[ 2 ]->gpu_data(), ex_.mutable_gpu_data());
		}
		
		// put mean blob into buffer_blob_
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1, Dtype(1),
			batch_sum_multiplier_.gpu_data(), ex_.gpu_data(), Dtype(0),
			spatial_statistic_.mutable_gpu_data());
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * channels_, height_ * width_, 1, Dtype(-1),
			spatial_statistic_.gpu_data(), spatial_sum_multiplier_.gpu_data(), Dtype(0),
			buffer_blob_.mutable_gpu_data());
		// substract mean
		caffe_gpu_add(buffer_blob_.count(), const_bottom_data, buffer_blob_.gpu_data(), top_data);

		// ---------- variance normalization ---------- //
		// add by yu liu
		// calculate EX2
		caffe_gpu_powx(bottom[ 0 ]->count(), const_bottom_data, Dtype(2), buffer_blob_.mutable_gpu_data());
		// statistic across spatial
		caffe_gpu_gemv<Dtype>(CblasNoTrans, num_ * channels_, height_ * width_, Dtype(1. / ( height_ * width_ )), buffer_blob_.gpu_data(),
			spatial_sum_multiplier_.gpu_data(), Dtype(0), spatial_statistic_.mutable_gpu_data());
		// statistic across batch
		caffe_gpu_gemv<Dtype>(CblasTrans, num_, channels_, Dtype(1. / num_), spatial_statistic_.gpu_data(),
			batch_sum_multiplier_.gpu_data(), Dtype(0), batch_statistic_.mutable_gpu_data());
		

		// original dx
      //// put the squares of X - mean into buffer_blob_
      //caffe_gpu_powx(buffer_blob_.count(), const_top_data, Dtype(2), buffer_blob_.mutable_gpu_data());
      //// statistic across spatial
      //caffe_gpu_gemv<Dtype>(CblasNoTrans, num_ * channels_, height_ * width_, Dtype(1. / (height_ * width_)), buffer_blob_.gpu_data(),
      //                      spatial_sum_multiplier_.gpu_data(), Dtype(0), spatial_statistic_.mutable_gpu_data());
      //// statistic across batch
      //caffe_gpu_gemv<Dtype>(CblasTrans, num_, channels_, Dtype(1. / num_), spatial_statistic_.gpu_data(),
      //                      batch_sum_multiplier_.gpu_data(), Dtype(0), batch_statistic_.mutable_gpu_data());
    if (this->phase_ == TRAIN) {
		if ( sync_forward_ ){
			// second, sync EX2
			caffe_copy(channels_, batch_statistic_.gpu_data(), statistics_all_.mutable_gpu_data());
			P2PSync<Dtype>* p2p = this->callbacks()[ 0 ]->callbacks()[ 0 ]->p2p()[ 0 ];
			Blob<Dtype> statistics_child(1, channels_, 1, 1);
			for ( int i = 0; i < p2p->children().size(); ++i ){
#ifdef _WIN64
				Blob<Dtype>* s_c_ogpu = p2p->dataQueue().pop();
#else // Linux support
				Blob<Dtype>* s_c_ogpu = NULL;
				while ( !p2p->dataQueue().try_pop(&s_c_ogpu) )
					;
#endif
				//Blob<Dtype>* s_c_ogpu = p2p->dataQueue().pop();
				CUDA_CHECK(cudaMemcpyAsync(statistics_child.mutable_gpu_data(), s_c_ogpu->gpu_data(), channels_*sizeof( Dtype ), cudaMemcpyDeviceToDevice, cudaStreamDefault));
				CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
				caffe_gpu_add(channels_, statistics_child.gpu_data(), statistics_all_.gpu_data(), statistics_all_.mutable_gpu_data());
			}
			if ( p2p->parent() ){
				p2p->parent()->dataQueue().push(&statistics_all_);
#ifdef _WIN64
				Blob<Dtype>* statistics_final = p2p->dataQueue().pop();
#else // Linux support
				Blob<Dtype>* statistics_final = NULL;
				while ( !p2p->dataQueue().try_pop(&statistics_final) )
					;
#endif
				//Blob<Dtype>* statistics_final = p2p->dataQueue().pop();
				CUDA_CHECK(cudaMemcpyAsync(dx_.mutable_gpu_data(), statistics_final->gpu_data(), channels_*sizeof( Dtype ), cudaMemcpyDeviceToDevice, cudaStreamDefault));
				CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
			}
			else {
				caffe_gpu_scal<Dtype>(channels_, Dtype(1.0 / Caffe::solver_count()), statistics_all_.mutable_gpu_data());
				Blob<Dtype> e2x_(1, channels_, 1, 1);
				caffe_gpu_powx(ex_.count(), ex_.gpu_data(), Dtype(2), e2x_.mutable_gpu_data());
				caffe_gpu_sub<Dtype>(ex_.count(), statistics_all_.gpu_data(), e2x_.gpu_data(), dx_.mutable_gpu_data());
			}
			for ( int i = 0; i < p2p->children().size(); ++i ){
				p2p->children()[ i ]->dataQueue().push(&dx_);
			}
		}
		else{
			Blob<Dtype> e2x_(1, channels_, 1, 1);
			caffe_gpu_powx(ex_.count(), ex_.gpu_data(), Dtype(2), e2x_.mutable_gpu_data());
			caffe_gpu_sub<Dtype>(ex_.count(), batch_statistic_.gpu_data(), e2x_.gpu_data(), dx_.mutable_gpu_data());
		}


      // save history variance
		caffe_gpu_axpby(dx_.count(), Dtype(1) - decay_, dx_.gpu_data(), decay_,
                      this->blobs_[3]->mutable_gpu_data());
    }
	if (this->phase_ == TEST ) {
		// use moving average variance
		if ( moving_average_ )
			caffe_copy(dx_.count(), this->blobs_[ 3 ]->gpu_data(), dx_.mutable_gpu_data());
		else{
			Blob<Dtype> e2x_(1, channels_, 1, 1);
			caffe_gpu_powx(ex_.count(), ex_.gpu_data(), Dtype(2), e2x_.mutable_gpu_data());
			caffe_gpu_sub<Dtype>(ex_.count(), batch_statistic_.gpu_data(), e2x_.gpu_data(), dx_.mutable_gpu_data());
		}
	}
    
	// Until now, dx_ should be calculated

    // add eps
	caffe_gpu_add_scalar(dx_.count(), var_eps_, dx_.mutable_gpu_data());
		// std
	caffe_gpu_powx(dx_.count(), dx_.gpu_data(), Dtype(0.5),
			batch_statistic_.mutable_gpu_data());
		// put std blob into buffer_blob_
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1, Dtype(1),
			batch_sum_multiplier_.gpu_data(), batch_statistic_.gpu_data(), Dtype(0),
			spatial_statistic_.mutable_gpu_data());
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * channels_, height_ * width_, 1, Dtype(1),
			spatial_statistic_.gpu_data(), spatial_sum_multiplier_.gpu_data(), Dtype(0),
			buffer_blob_.mutable_gpu_data());
		// variance normalization
		caffe_gpu_div(buffer_blob_.count(), const_top_data, buffer_blob_.gpu_data(), top_data);

		// ---------- save x_norm and x_std ---------- //
		caffe_copy(buffer_blob_.count(), const_top_data, x_norm_.mutable_gpu_data());
		caffe_copy(batch_statistic_.count(), batch_statistic_.gpu_data(), x_std_.mutable_gpu_data());

		// ---------- scale ---------- //
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1, Dtype(1),
			batch_sum_multiplier_.gpu_data(), scale_data, Dtype(0),
			spatial_statistic_.mutable_gpu_data());
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * channels_, height_ * width_, 1, Dtype(1),
			spatial_statistic_.gpu_data(), spatial_sum_multiplier_.gpu_data(), Dtype(0),
			buffer_blob_.mutable_gpu_data());
		caffe_gpu_mul(buffer_blob_.count(), const_top_data, buffer_blob_.gpu_data(), top_data);

		// ---------- shift ---------- //
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1, Dtype(1),
			batch_sum_multiplier_.gpu_data(), shift_data, Dtype(0),
			spatial_statistic_.mutable_gpu_data());
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * channels_, height_ * width_, 1, Dtype(1),
			spatial_statistic_.gpu_data(), spatial_sum_multiplier_.gpu_data(), Dtype(0),
			buffer_blob_.mutable_gpu_data());
		caffe_gpu_add(buffer_blob_.count(), const_top_data, buffer_blob_.gpu_data(), top_data);

	}

	template <typename Dtype>
	void BNLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
		const Dtype* const_bottom_diff = bottom[0]->gpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
		const Dtype* const_top_diff = top[0]->gpu_diff();	

		Dtype* scale_diff = this->blobs_[0]->mutable_gpu_diff();
		Dtype* shift_diff = this->blobs_[1]->mutable_gpu_diff();
		const Dtype* scale_data = this->blobs_[0]->gpu_data();

		// ---------- gradient w.r.t. scale ---------- //
		caffe_gpu_mul(buffer_blob_.count(), x_norm_.gpu_data(), const_top_diff, buffer_blob_.mutable_gpu_data());
		// statistic across spatial
		caffe_gpu_gemv<Dtype>(CblasNoTrans, num_ * channels_, height_ * width_, Dtype(1), buffer_blob_.gpu_data(),
			spatial_sum_multiplier_.gpu_data(), Dtype(0), spatial_statistic_.mutable_gpu_data());
		// statistic across batch
		caffe_gpu_gemv<Dtype>(CblasTrans, num_, channels_, Dtype(1), spatial_statistic_.gpu_data(),
			batch_sum_multiplier_.gpu_data(), Dtype(0), scale_diff);

		// ---------- gradient w.r.t. shift ---------- //
		// statistic across spatial
		caffe_gpu_gemv<Dtype>(CblasNoTrans, num_ * channels_, height_ * width_, Dtype(1), const_top_diff,
			spatial_sum_multiplier_.gpu_data(), Dtype(0), spatial_statistic_.mutable_gpu_data());
		// statistic across batch
		caffe_gpu_gemv<Dtype>(CblasTrans, num_, channels_, Dtype(1), spatial_statistic_.gpu_data(),
			batch_sum_multiplier_.gpu_data(), Dtype(0), shift_diff);

		// ---------- gradient w.r.t. to bottom blob ---------- //
		// put scale * top_diff to buffer_blob_
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1, Dtype(1),
			batch_sum_multiplier_.gpu_data(), scale_data, Dtype(0),
			spatial_statistic_.mutable_gpu_data());
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * channels_, height_ * width_, 1, Dtype(1),
			spatial_statistic_.gpu_data(), spatial_sum_multiplier_.gpu_data(), Dtype(0),
			buffer_blob_.mutable_gpu_data());
		caffe_gpu_mul(buffer_blob_.count(), const_top_diff, buffer_blob_.gpu_data(), buffer_blob_.mutable_gpu_data());

    if (this->phase_ == TRAIN) {
      // use new top diff for computation
      caffe_gpu_mul(buffer_blob_.count(), x_norm_.gpu_data(), buffer_blob_.gpu_data(), bottom_diff);
      // statistic across spatial
      caffe_gpu_gemv<Dtype>(CblasNoTrans, num_ * channels_, height_ * width_, Dtype(1), const_bottom_diff,
                            spatial_sum_multiplier_.gpu_data(), Dtype(0), spatial_statistic_.mutable_gpu_data());
      // statistic across batch
      caffe_gpu_gemv<Dtype>(CblasTrans, num_, channels_, Dtype(1), spatial_statistic_.gpu_data(),
                            batch_sum_multiplier_.gpu_data(), Dtype(0), batch_statistic_.mutable_gpu_data());

      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1, Dtype(1),
                            batch_sum_multiplier_.gpu_data(), batch_statistic_.gpu_data(), Dtype(0),
                            spatial_statistic_.mutable_gpu_data());
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * channels_, height_ * width_, 1, Dtype(1),
                            spatial_statistic_.gpu_data(), spatial_sum_multiplier_.gpu_data(), Dtype(0),
                            bottom_diff);

      caffe_gpu_mul(buffer_blob_.count(), x_norm_.gpu_data(), const_bottom_diff, bottom_diff);

      // statistic across spatial
      caffe_gpu_gemv<Dtype>(CblasNoTrans, num_ * channels_, height_ * width_, Dtype(1), buffer_blob_.gpu_data(),
                            spatial_sum_multiplier_.gpu_data(), Dtype(0), spatial_statistic_.mutable_gpu_data());
      // statistic across batch
      caffe_gpu_gemv<Dtype>(CblasTrans, num_, channels_, Dtype(1), spatial_statistic_.gpu_data(),
                            batch_sum_multiplier_.gpu_data(), Dtype(0), batch_statistic_.mutable_gpu_data());

      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1, Dtype(1),
                            batch_sum_multiplier_.gpu_data(), batch_statistic_.gpu_data(), Dtype(0),
                            spatial_statistic_.mutable_gpu_data());
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * channels_, height_ * width_, 1, Dtype(1),
                            spatial_statistic_.gpu_data(), spatial_sum_multiplier_.gpu_data(), Dtype(1),
                            bottom_diff);

      caffe_gpu_axpby(buffer_blob_.count(), Dtype(1), buffer_blob_.gpu_data(), Dtype(-1. / (num_ * height_ * width_)),
                      bottom_diff);
    }
    if (this->phase_ == TEST && moving_average_) {
      // use moving average variance
      caffe_copy(buffer_blob_.count(), buffer_blob_.gpu_data(), bottom_diff);
    }
        
		// variance normalization
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1, Dtype(1),
			batch_sum_multiplier_.gpu_data(), x_std_.gpu_data(), Dtype(0),
			spatial_statistic_.mutable_gpu_data());
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * channels_, height_ * width_, 1, Dtype(1),
			spatial_statistic_.gpu_data(), spatial_sum_multiplier_.gpu_data(), Dtype(0),
			buffer_blob_.mutable_gpu_data());

		caffe_gpu_div(buffer_blob_.count(), const_bottom_diff, buffer_blob_.gpu_data(), bottom_diff);

	}

	INSTANTIATE_LAYER_GPU_FUNCS(BNLayer);

}  // namespace caffe
