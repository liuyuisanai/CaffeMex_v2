#ifndef CAFFE_NORMALIZATION_LAYER_HPP_
#define CAFFE_NORMALIZATION_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe { 


  template <typename Dtype>
  class NormalizationLayer : public Layer<Dtype> {
   public:
    explicit NormalizationLayer(const LayerParameter& param)
        : Layer<Dtype>(param) {}
    //virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    //    const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const {
      return "Normalization";
    }
    virtual inline int ExactNumBottomBlobs() const { return 1; }
    virtual inline int ExactNumTopBlobs() const { return 1; }

   protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
   // virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
   //     const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
   // virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      // const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

    Blob<Dtype> sum_multiplier_;
    Blob<Dtype> norm_;
    Blob<Dtype> squared_;
  };

}  // namespace caffe

#endif  // CAFFE_NORMALIZATION_LAYER_HPP_
