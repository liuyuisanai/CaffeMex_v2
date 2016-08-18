#ifndef CAFFE_BNHK_LAYER_HPP_
#define CAFFE_BNHK_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

  template <typename Dtype>
  class BNHKLayer : public Layer<Dtype> {
   public:
    explicit BNHKLayer(const LayerParameter& param)
        : Layer<Dtype>(param) {}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
  
    virtual inline const char* type() const { return "BNHK"; }
    virtual inline int ExactNumBottomBlobs() const { return 1; }
    virtual inline int ExactNumTopBlobs() const { return 1; }
  
   protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom    );
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom    );
  
    void AverageAllExceptChannel(const Dtype* input, Dtype* output);
    void BroadcastChannel(const Dtype* input, Dtype* output);
  
    bool frozen_;
    Dtype bn_momentum_;
    Dtype bn_eps_;
  
    int num_;
    int channels_;
    int height_;
    int width_;
  
    Blob<Dtype> broadcast_buffer_;
    Blob<Dtype> spatial_statistic_;
    Blob<Dtype> batch_statistic_;
  
    Blob<Dtype> x_norm_;
    Blob<Dtype> x_inv_std_;
  
    Blob<Dtype> spatial_sum_multiplier_;
    Blob<Dtype> batch_sum_multiplier_;
  };
}

#endif  // CAFFE_BNHK_LAYER_HPP_
