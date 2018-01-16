/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tiny_dnn/layers/layer.h"

#include "tiny_dnn/core/kernels/fully_connected_grad_op.h"
#include "tiny_dnn/core/kernels/fully_connected_op.h"

namespace tiny_dnn {

/**
 * compute fully-connected(matmul) operation
 **/
class pruned_fully_connected_layer : public layer {
public:
    /**
     * @param in_dim [in] number of elements of the input
     * @param out_dim [in] number of elements of the output
     * @param has_bias [in] whether to include additional bias to the layer
     * @param pruning_percentage [in] the pruning percentage in this layer, between 0 and 1
     **/
    pruned_fully_connected_layer(size_t in_dim,
                                 size_t out_dim,
                                 bool has_bias = true,
                                 core::backend_t backend_type = core::default_engine(),
                                 float_t pruning_percentage = 0)
            : layer(std_input_order(has_bias), {vector_type::data}) {
        std::cout << "Total number of weights: " << in_dim * out_dim
                  << ", pruned number of weights: " << std::floor(pruning_percentage * in_dim * out_dim) << std::endl;
        pruning_percentage_ = pruning_percentage_;
        set_params(in_dim, out_dim, has_bias);
        init_backend(backend_type);
        layer::set_backend_type(backend_type);
    }

    // move constructor
    pruned_fully_connected_layer(pruned_fully_connected_layer &&other)
            : layer(std::move(other)),
              params_(std::move(other.params_)),
              kernel_fwd_(std::move(other.kernel_fwd_)),
              kernel_back_(std::move(other.kernel_back_)) {
        init_backend(std::move(other.engine()));
    }

    size_t fan_in_size() const override { return params_.in_size_; }

    size_t fan_out_size() const override { return params_.out_size_; }

    std::vector <index3d<size_t>> in_shape() const override {
        if (params_.has_bias_) {
            return {index3d<size_t>(params_.in_size_, 1, 1),
                    index3d<size_t>(params_.in_size_, params_.out_size_, 1),
                    index3d<size_t>(params_.out_size_, 1, 1)};
        } else {
            return {index3d<size_t>(params_.in_size_, 1, 1),
                    index3d<size_t>(params_.in_size_, params_.out_size_, 1)};
        }
    }

    std::vector <index3d<size_t>> out_shape() const override {
        return {index3d<size_t>(params_.out_size_, 1, 1)};
    }

    void forward_propagation(const std::vector<tensor_t *> &in_data,
                             std::vector<tensor_t *> &out_data) override {
        // in_data is input vectors of this layer (data, weight, bias)
        // for details see layer.h
        // sort weights vector and set mask
        // for fully connected layer, there is only one matrix, thus it's stored in tensor_t[0]
        vec_t &weights = (*in_data[1])[0];
        set_mask(weights);
        // copy weights to masked_weights
        masked_weights_.reset(new tensor_t);
        masked_weights_->push_back(vec_t(weights));
        for (size_t k = 0; k < params_.in_size_ * params_.out_size_; k++) {
            (*masked_weights_)[0][k] = weights[k] * mask_[k];
        }

        std::vector < tensor_t * > masked_in_data;
        masked_in_data.push_back(in_data[0]);
        masked_in_data.push_back(masked_weights_.get());
        if (params_.has_bias_)
            masked_in_data.push_back(in_data[2]);

        // forward fully connected op context
        fwd_ctx_.set_in_out(masked_in_data, out_data);
        fwd_ctx_.setParallelize(layer::parallelize());
        fwd_ctx_.setEngine(layer::engine());

        // launch fully connected kernel
        kernel_fwd_->compute(fwd_ctx_);
    }

    void back_propagation(const std::vector<tensor_t *> &in_data,
                          const std::vector<tensor_t *> &out_data,
                          std::vector<tensor_t *> &out_grad,
                          std::vector<tensor_t *> &in_grad) override {
        // in_grad is output vectors of this layer (delta wrt input activations, weight gradient, bias gradient)
        // see fully_connected_grad_op
        // replace in_data with masked weights
        std::vector < tensor_t * > masked_in_data;
        masked_in_data.push_back(in_data[0]);
        masked_in_data.push_back(masked_weights_.get());
        if (params_.has_bias_)
            masked_in_data.push_back(in_data[2]);

        // backward fully connected op context
        bwd_ctx_.set_in_out(in_data, out_data, out_grad, in_grad);
        bwd_ctx_.setParallelize(layer::parallelize());
        bwd_ctx_.setEngine(layer::engine());

        // launch fully connected kernel
        kernel_back_->compute(bwd_ctx_);

        // mask out weights gradient
        // weights update should be done by optimization procedure
        tensor_t &dW = *out_grad[1];
        for (size_t k = 0; k < params_.in_size_ * params_.out_size_; k++) {
            dW[0][k] *= mask_[k];
        }
    }

    std::string layer_type() const override { return "fully-connected"; }

    friend struct serialization_buddy;

protected:
    void set_params(const size_t in_size, const size_t out_size, bool has_bias) {
        params_.in_size_ = in_size;
        params_.out_size_ = out_size;
        params_.has_bias_ = has_bias;
    }

    void init_backend(core::backend_t backend_type) {
        core::OpKernelConstruction ctx =
                core::OpKernelConstruction(layer::device(), &params_);

        if (backend_type == core::backend_t::internal ||
            backend_type == core::backend_t::avx ||
            backend_type == core::backend_t::nnpack) {
            kernel_fwd_.reset(new FullyConnectedOp(ctx));
            kernel_back_.reset(new FullyConnectedGradOp(ctx));
        } else {
            throw nn_error("Not supported engine: " + to_string(backend_type));
        }
    }

    void set_mask(const vec_t &weights) {
        size_t num_weights = params_.in_size_ * params_.out_size_;
        std::vector <std::pair<float_t, int>> indexed_weights;
        for (size_t k = 0; k < num_weights; k++) {
            indexed_weights.push_back(std::make_pair(std::abs(weights[k]), k));
            mask_.push_back(true);
        }

        // sort weights according to its absolute value
        std::partial_sort(indexed_weights.begin(),
                          indexed_weights.begin() + std::floor(pruning_percentage_ * num_weights),
                          indexed_weights.end());
        for (size_t k = 0; k < std::floor(pruning_percentage_ * num_weights); k++) {
            mask_[indexed_weights[k].second] = false;
        }
    }

private:
    // set pruning parameters
    float_t pruning_percentage_;
    std::vector<bool> mask_;
    std::shared_ptr <tensor_t> masked_weights_;

    /* The layer parameters */
    core::fully_params params_;

    /* forward op context */
    core::OpKernelContext fwd_ctx_;

    /* backward op context */
    core::OpKernelContext bwd_ctx_;

    /* Forward and backward ops */
    std::shared_ptr <core::OpKernel> kernel_fwd_;
    std::shared_ptr <core::OpKernel> kernel_back_;
};

}  // namespace tiny_dnn
