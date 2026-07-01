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
 * parallelize the inner work across the filter index `z` using TBB
 * (via std::execution::par). Behavior is functionally identical to
 * Convolution3D when `inhibition=true` (the typical KTH setup) and
 * weights/thresholds end up bit-identical because we partition by z and
 * each thread writes only to its own z-slice -> zero race condition.
 *
 * Why partitioning by z is safe:
 *   - Activations  _a[x,y,z,k]   : indexed by z, disjoint per thread.
 *   - Inhibition   _inh[x,y,z,k] : same.
 *   - Weights      w[x,y,zi,z,k] : the slice for a given z is owned by
 *                                  exactly one thread.
 *   - Thresholds   th[z]         : in test, read-only. In train, the
 *                                  threshold update touches all z's, so
 *                                  the train override keeps that step
 *                                  sequential and only parallelizes the
 *                                  per-spike accumulator and weight
 *                                  update over (x,y,zi,k) for the
 *                                  winning filter.
 *   - Output spikes              : each thread appends to its own
 *                                  thread-local vector; merged at the end.
 */
    class ConvolutionSampler3D : public Convolution3D {
    public:
        ConvolutionSampler3D();

        ConvolutionSampler3D(size_t filter_number, size_t filter_width, size_t filter_height, size_t filter_depth,
                             std::string model_path = "",
                             size_t stride_x = 1, size_t stride_y = 1, size_t stride_k = 1,
                             size_t padding_x = 0, size_t padding_y = 0, size_t padding_k = 0);

        // Overrides
        Shape compute_shape(const Shape &previous_shape) override;

        // Patch extraction entry point. The base Convolution3D uses a plain
        // random patch; here we delegate the patch location to the injected
        // Sampler (HOG / Random / ...) so the sampling strategy lives with
        // this class instead of the generic convolution layer.
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
        // Patch sampling strategy, injected as the "sampler" parameter.
        // Lives here (not in the generic Convolution3D) so the base layer
        // stays sampler-agnostic.
        Sampler *_sampler;

        // Local activation / inhibition state. We can't reuse the base
        // class' (which lives in the private _priv::Convolution3DImpl).
        Tensor<float> _a_local;
        Tensor<bool>  _inh_local;

        // Cached shape info derived from compute_shape.
        size_t _input_depth_local;
        size_t _input_conv_depth_local;

        // Number of worker threads to use; lazily set from
        // std::thread::hardware_concurrency() on first call.
        size_t _num_threads;

        // Save/Load support (the base Convolution3DImpl handles these, but our
        // override bypasses it, so we replicate the minimum needed here).
        std::string _model_path_local;
        std::string _last_label;
        bool        _weights_loaded;
        bool        _weights_saved;

        // Maximum number of spikes processed during training.
        // 0 = unlimited.  Spikes are time-sorted (earliest = most
        // important in latency coding), so capping early keeps the
        // most informative ones and dramatically reduces O(S×D) cost
        // in deeper layers where input channel count explodes.
        size_t _max_train_spikes;
        size_t _epoch_counter;

        // Per-epoch activity counters (reset in on_epoch_end).
        size_t _fire_count_epoch;
        size_t _sample_count_epoch;

        void ensure_state_allocated();
        void try_load_weights(const std::string &label);
        void try_save_weights(const std::string &label);
    };

} // namespace layer
