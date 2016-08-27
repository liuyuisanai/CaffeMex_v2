#ifndef CAFFE_SIGMOID_CROSS_ENTROPY_LOSS_WEIGHTED_LAYER_HPP_
#define CAFFE_SIGMOID_CROSS_ENTROPY_LOSS_WEIGHTED_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"
#include "caffe/layers/sigmoid_layer.hpp"

namespace caffe {

/**
 * @brief Computes the cross-entropy (logistic) loss where the loss that the 
 * i-th training point contributes to the total loss is weighted by the weight[i]
 * factor where weight is the 3rd input blob to the layer
 */
template <typename Dtype>
class SigmoidCrossEntropyLossWeightedLayer : public LossLayer<Dtype> {
 public:
  explicit SigmoidCrossEntropyLossWeightedLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param),
          sigmoid_layer_(new SigmoidLayer<Dtype>(param)),
          sigmoid_output_(new Blob<Dtype>()) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline int ExactNumBottomBlobs() const { return 3; }
  virtual inline const char* type() const { return "SigmoidCrossEntropyLossWeighted"; }

 protected:
  /// @copydoc SigmoidCrossEntropyLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  /// The internal SigmoidLayer used to map predictions to probabilities.
  shared_ptr<SigmoidLayer<Dtype> > sigmoid_layer_;
  /// sigmoid_output stores the output of the SigmoidLayer.
  shared_ptr<Blob<Dtype> > sigmoid_output_;
  /// bottom vector holder to call the underlying SigmoidLayer::Forward
  vector<Blob<Dtype>*> sigmoid_bottom_vec_;
  /// top vector holder to call the underlying SigmoidLayer::Forward
  vector<Blob<Dtype>*> sigmoid_top_vec_;
};

}  // namespace caffe

#endif  // CAFFE_SIGMOID_CROSS_ENTROPY_LOSS_WEIGHTED_LAYER_HPP_
