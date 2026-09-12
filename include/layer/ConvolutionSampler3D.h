#pragma once

#include "layer/Convolution3D.h"
#include "Sampler.h"
#include "Tensor.h"
#include <vector>
#include <mutex>

namespace layer {

/**
 * ConvolutionSampler3D
 * --------------------
 * Drop-in replacement for Convolution3D that overrides train()/test() to
 * parallelize the inner work across the filter index `z`.
 * Behavior is functionally identical to
 * Convolution3D when `inhibition=true` (the typical KTH setup) and
 * weights/thresholds end up bit-identical because we partition by z and
 * each thread writes only to its own z-slice -> zero race condition.
 *
 */
    class ConvolutionSampler3D : public Convolution3D {
    public:
        ConvolutionSampler3D();

        ConvolutionSampler3D(size_t filter_number, size_t filter_width, size_t filter_height, size_t filter_depth,
                             std::string model_path = "",
                             size_t stride_x = 1, size_t stride_y = 1, size_t stride_k = 1,
                             size_t padding_x = 0, size_t padding_y = 0, size_t padding_k = 0);

        Shape compute_shape(const Shape &previous_shape) override;

        void process_train_sample(const std::string &label, Tensor<float> &sample,
                                  size_t current_pass, size_t current_index, size_t number) override;

        void train(const std::string &label,
                   const std::vector<Spike> &input_spike,
                   const Tensor<Time> &input_time,
                   std::vector<Spike> &output_spike) override;

        void test(const std::string &label,
                  const std::vector<Spike> &input_spike,
                  const Tensor<Time> &input_time,
                  std::vector<Spike> &output_spike) override;

        void on_epoch_end() override;

    private:
        Sampler *_sampler;

        Tensor<float> _a_local;
        Tensor<bool>  _inh_local;

        size_t _input_depth_local;
        size_t _input_conv_depth_local;

        size_t _num_threads;

        std::string _model_path_local;
        std::string _last_label;
        bool        _weights_loaded;
        bool        _weights_saved;

        size_t _max_train_spikes;
        size_t _epoch_counter;

        size_t _fire_count_epoch;
        size_t _sample_count_epoch;

        void ensure_state_allocated();
        void try_load_weights(const std::string &label);
        void try_save_weights(const std::string &label);
    };

} // namespace layer
