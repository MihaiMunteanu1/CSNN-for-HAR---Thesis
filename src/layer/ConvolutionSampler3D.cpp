#include "layer/ConvolutionSampler3D.h"
#include "Stdp.h"
#include "tool/Operations.h"

#include <algorithm>
#include <execution>
#include <numeric>
#include <thread>
#include <vector>
#include <tuple>
#include <filesystem>
#include <iostream>

#include <limits>
#include <cstring>

using namespace layer;

// Register this layer type so it can be created/configured like other layers.
static RegisterClassParameter<ConvolutionSampler3D, LayerFactory> _register("ConvolutionSampler3D");

ConvolutionSampler3D::ConvolutionSampler3D()
        : Convolution3D(),
          _a_local(),
          _inh_local(),
          _input_depth_local(0),
          _input_conv_depth_local(0),
          _num_threads(0),
          _model_path_local(),
          _weights_loaded(false),
          _weights_saved(false),
          _max_train_spikes(0),
          _epoch_counter(0),
          _fire_count_epoch(0),
          _sample_count_epoch(0)
{
    add_parameter("max_train_spikes", _max_train_spikes, static_cast<size_t>(0));
}

ConvolutionSampler3D::ConvolutionSampler3D(size_t filter_number, size_t filter_width, size_t filter_height, size_t filter_depth,
                                           std::string model_path,
                                           size_t stride_x, size_t stride_y, size_t stride_k,
                                           size_t padding_x, size_t padding_y, size_t padding_k)
        // NOTE: base Convolution3D ctor signature is
        //   (filter_width, filter_height, filter_depth, filter_number, model_path, ...)
        // while OUR wrapper takes filter_number FIRST. Forward in base order.
        : Convolution3D(filter_width, filter_height, filter_depth, filter_number, model_path,
                        stride_x, stride_y, stride_k, padding_x, padding_y, padding_k),
          _a_local(),
          _inh_local(),
          _input_depth_local(0),
          _input_conv_depth_local(0),
          _num_threads(0),
          _model_path_local(model_path),
          _weights_loaded(false),
          _weights_saved(false),
          _max_train_spikes(0),
          _epoch_counter(0),
          _fire_count_epoch(0),
          _sample_count_epoch(0)
{
    add_parameter("max_train_spikes", _max_train_spikes, static_cast<size_t>(0));
}

Shape ConvolutionSampler3D::compute_shape(const Shape &previous_shape)
{
    // Let the base class do its work (sets _width, _height, _depth, _conv_depth,
    // and shapes the "w" / "th" parameter tensors).
    Shape out = Convolution3D::compute_shape(previous_shape);

    _input_depth_local      = previous_shape.dim(2);
    _input_conv_depth_local = previous_shape.number() > 3 ? previous_shape.dim(3) : 1;

    // Allocate parallel-safe state owned by THIS class.
    _a_local   = Tensor<float>(Shape({width(), height(), depth(), conv_depth()}));
    _inh_local = Tensor<bool>(Shape({width(), height(), depth(), conv_depth()}));

    if (_num_threads == 0)
    {
        unsigned hc = std::thread::hardware_concurrency();
        _num_threads = (hc == 0) ? 1u : static_cast<size_t>(hc);
    }

    // When weights are pre-loaded, training is a no-op (train() returns
    // immediately).  But the framework still runs _epoch_number training
    // passes — loading every video from disk, preprocessing, extracting a
    // patch, converting to spikes — only to throw the result away.
    // Setting epoch to 0 makes train_pass_number() return 1 (one inference
    // pass, zero training passes), eliminating all that wasted I/O.
    if (!_model_path_local.empty())
    {
        parameter<uint32_t>("epoch").set(0);
        std::cout << "ConvolutionSampler3D: weights pre-loaded, "
                  << "skipping training epochs (epoch forced to 0)."
                  << std::endl;
    }

    return out;
}

// ---------------------------------------------------------------------------
// Save / Load helpers
// ---------------------------------------------------------------------------
// Replicates the bits of _priv::Convolution3DImpl::train that handle weight
// persistence (save at end-of-training, load from JSON if model_path nonempty).
// Path layout matches the base class:
//   <cwd>/Weights/<exp_name>/<layerIndex>/<exp_name>.json
// where exp_name and layerIndex are parsed out of the framework label
// "<exp>;.<layerIndex>;.<rest>".
// ---------------------------------------------------------------------------
static bool parse_label(const std::string &label,
                        std::string &exp_name, std::string &layer_index)
{
    const std::string delim = ";.";
    auto p1 = label.find(delim);
    if (p1 == std::string::npos) return false;
    exp_name = label.substr(0, p1);
    auto rest_start = p1 + delim.size();
    auto p2 = label.find(delim, rest_start);
    if (p2 == std::string::npos) return false;
    layer_index = label.substr(rest_start, p2 - rest_start);
    return true;
}

void ConvolutionSampler3D::try_load_weights(const std::string &label)
{
    if (_weights_loaded) return;
    if (_model_path_local.empty()) { _weights_loaded = true; return; }

    std::string exp_name, layer_index;
    if (!parse_label(label, exp_name, layer_index)) return;

    // The same label-stripping the base does, so LoadWeights gets the bare
    // class label.
    std::string bare = label;
    bare.erase(0, exp_name.size() + 2);
    bare.erase(0, layer_index.size() + 2);

    Tensor<float> &w = parameter<Tensor<float>>("w").get();
    try
    {
        LoadWeights(_model_path_local, bare, w);
        std::cout << "ConvolutionSampler3D: loaded weights from "
                  << _model_path_local << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cout << "ConvolutionSampler3D: LoadWeights failed ("
                  << e.what() << "), falling back to random init."
                  << std::endl;
    }

    // Thresholds are adapted during STDP, so they must be loaded alongside
    // weights to reproduce the trained inference behavior. Stored in a
    // companion file <exp>_th.json to avoid label collisions with the
    // weight-file fallback loader.
    std::string th_path = _model_path_local;
    size_t ext_pos = th_path.rfind(".json");
    if (ext_pos != std::string::npos)
        th_path.insert(ext_pos, "_th");
    else
        th_path += "_th";

    if (std::filesystem::exists(th_path))
    {
        Tensor<float> &th = parameter<Tensor<float>>("th").get();
        try
        {
            LoadWeights(th_path, bare, th);
            std::cout << "ConvolutionSampler3D: loaded thresholds from "
                      << th_path << std::endl;
        }
        catch (const std::exception &e)
        {
            std::cout << "ConvolutionSampler3D: LoadWeights(th) failed ("
                      << e.what() << "), thresholds kept at init values."
                      << std::endl;
        }
    }
    else
    {
        std::cout << "ConvolutionSampler3D: no threshold file at "
                  << th_path << " -- retrain once to produce it, "
                  << "otherwise inference uses the initial random thresholds."
                  << std::endl;
    }

    _weights_loaded = true;
}

void ConvolutionSampler3D::try_save_weights(const std::string &label)
{
    if (_weights_saved) return;
    const bool save = parameter<bool>("save_weights").get();
    if (!save) { return; }

    std::string exp_name, layer_index;
    if (!parse_label(label, exp_name, layer_index)) return;

    std::string bare = label;
    bare.erase(0, exp_name.size() + 2);
    bare.erase(0, layer_index.size() + 2);

    std::string dir     = std::filesystem::current_path().string()
                          + "/Weights/" + exp_name + "/" + layer_index + "/";
    std::string file    = dir + exp_name + ".json";
    std::string th_file = dir + exp_name + "_th.json";
    std::filesystem::create_directories(dir);

    Tensor<float> &w  = parameter<Tensor<float>>("w").get();
    Tensor<float> &th = parameter<Tensor<float>>("th").get();
    SaveWeights(file,    bare, w);
    SaveWeights(th_file, bare, th);
    _weights_saved = true;
}

void ConvolutionSampler3D::ensure_state_allocated()
{
    // Defensive: in case compute_shape wasn't called (shouldn't happen).
    if (_a_local.shape().product() == 0)
    {
        _a_local   = Tensor<float>(Shape({width(), height(), depth(), conv_depth()}));
        _inh_local = Tensor<bool>(Shape({width(), height(), depth(), conv_depth()}));
    }
    if (_num_threads == 0)
    {
        unsigned hc = std::thread::hardware_concurrency();
        _num_threads = (hc == 0) ? 1u : static_cast<size_t>(hc);
    }
}

// ---------------------------------------------------------------------------
// TRAIN
// ---------------------------------------------------------------------------
//
// Training is fundamentally sequential per sample because of WTA + threshold
// updates that touch all filters. We keep the spike loop sequential, but for
// each spike we parallelize the per-filter accumulator update across z, then
// pick the smallest z that crossed (matching the original semantics where
// `for z=0..depth: if cross then return` selected the smallest index winner).
// When a winner fires, we sequentially update all thresholds (touches all z)
// and then parallelize the STDP weight update for that single winning z over
// (x, y, zi, k) — every write hits a unique tensor index, so it's race-free.
// ---------------------------------------------------------------------------
void ConvolutionSampler3D::train(const std::string &label,
                                 const std::vector<Spike> &input_spike,
                                 const Tensor<Time> &input_time,
                                 std::vector<Spike> &output_spike)
{
    (void)output_spike;

    ensure_state_allocated();

    // If a model_path was provided, load once and skip training entirely.
    // The framework will keep calling train() for every (epoch, sample) but
    // we just no-op so the pre-trained weights are preserved.
    if (!_model_path_local.empty())
    {
        try_load_weights(label);
        return;
    }

    _last_label = label;
    _sample_count_epoch++;

    Tensor<float> &w  = parameter<Tensor<float>>("w").get();
    Tensor<float> &th = parameter<Tensor<float>>("th").get();
    STDP          &stdp = parameter<STDP>("stdp").get();

    const float  t_obj      = parameter<float>("t_obj").get();
    const float  lr_th      = parameter<float>("lr_th").get();
    const float  min_th     = parameter<float>("min_th").get();
    const bool   inhibition = parameter<bool>("inhibition").get();

    const size_t D    = depth();                  // number of filters
    const size_t Fw   = _filter_width;
    const size_t Fh   = _filter_height;
    const size_t Fcd  = _filter_conv_depth;
    const size_t Iz   = _input_depth_local;

    // ---- #5: Spike top-K limit ----
    // input_spike is sorted by time (earliest = most important in latency
    // coding). Capping the count keeps the most informative spikes and
    // reduces the O(S × D) hotpath for deeper layers where channel count
    // (and thus spike count) explodes.
    const size_t spike_count =
        (_max_train_spikes > 0 && input_spike.size() > _max_train_spikes)
            ? _max_train_spikes
            : input_spike.size();

    // ---- #4: Contiguous accumulator + raw-pointer striding ----
    // Instead of the strided _a_local.at(0,0,z,0) calls (each going
    // through variadic template expansion + debug asserts), we use a
    // contiguous float array. th is already 1D (shape D), contiguous.
    // Weight stride between consecutive z values = Fcd.
    alignas(64) float accum[512];
    std::vector<float> accum_heap;
    float *acc;
    if (D <= 512) {
        acc = accum;
    } else {
        accum_heap.resize(D, 0.0f);
        acc = accum_heap.data();
    }
    std::memset(acc, 0, D * sizeof(float));

    float *th_data = th.ptr(static_cast<size_t>(0));

    for (size_t s = 0; s < spike_count; ++s)
    {
        const Spike &spike = input_spike[s];

        // Raw pointer to w(spike.x, spike.y, spike.z, 0, spike.k).
        // Consecutive z values sit at stride Fcd apart.
        const float *w_ptr = w.ptr(spike.x, spike.y, spike.z,
                                    static_cast<size_t>(0), spike.k);

        // Accumulate — acc is contiguous, w_ptr is strided by Fcd.
        for (size_t z = 0; z < D; ++z)
            acc[z] += w_ptr[z * Fcd];

        // Find smallest z that crossed threshold.
        size_t winner = std::numeric_limits<size_t>::max();
        for (size_t z = 0; z < D; ++z)
        {
            if (acc[z] >= th_data[z])
            {
                winner = z;
                break;
            }
        }

        if (winner == std::numeric_limits<size_t>::max())
            continue;

        _fire_count_epoch++;

        // Sequential threshold update across all filters.
        const float time_delta = lr_th * (spike.time - t_obj);
        const float loser_penalty = lr_th / static_cast<float>(D - 1);
        for (size_t z1 = 0; z1 < D; ++z1)
        {
            th_data[z1] -= time_delta;
            th_data[z1] += (z1 == winner) ? lr_th : -loser_penalty;
            if (th_data[z1] < min_th) th_data[z1] = min_th;
        }

        // Serial STDP weight update for the winning filter.
        // 26K iterations of a lightweight exp() are faster serial than
        // dispatched through TBB (parallel overhead + 208 KB heap alloc
        // dominated total cost at this granularity).
        const float post_t = spike.time;
        for (size_t x = 0; x < Fw; ++x)
            for (size_t y = 0; y < Fh; ++y)
                for (size_t zi = 0; zi < Iz; ++zi)
                    for (size_t k = 0; k < Fcd; ++k) {
                        float &wref = w.at(x, y, zi, winner, k);
                        wref = stdp.process(wref, input_time.at(x, y, zi, k), post_t);
                    }

        if (inhibition)
            return; // WTA: stop processing this sample after first fire.
    }
}

void ConvolutionSampler3D::on_epoch_end()
{
    Convolution3D::on_epoch_end();

    _epoch_counter++;
    uint32_t total = parameter<uint32_t>("epoch").get();
    double fps = _sample_count_epoch > 0
                 ? static_cast<double>(_fire_count_epoch) / _sample_count_epoch
                 : 0.0;
    std::cout << "\n[" << name() << "] Epoch " << _epoch_counter << "/" << total
              << "  samples=" << _sample_count_epoch
              << "  fires=" << _fire_count_epoch
              << "  fires/sample=" << fps << std::endl;
    _fire_count_epoch = 0;
    _sample_count_epoch = 0;

    // Persist weights at the end of every epoch (cheap, overwrites).
    // The file always reflects the latest epoch when training finishes.
    if (!_last_label.empty())
    {
        _weights_saved = false;
        try_save_weights(_last_label);
    }
}

// ---------------------------------------------------------------------------
// TEST
// ---------------------------------------------------------------------------
//
// This is where we get the big speedup. Inference walks every input spike,
// and for each spike updates a set of (x,y,k) output positions across all
// filters z. We partition the filter index z into N chunks (N = thread count)
// and run them in parallel. Each thread:
//   - sees the SAME read-only spike list
//   - writes only to _a_local[*,*,z_chunk,*] and _inh_local[*,*,z_chunk,*]
//   - appends fired spikes to its OWN local vector
// At the end we concatenate the per-thread output vectors into output_spike.
// Weights are read-only in test, so no race on w/th anywhere.
// ---------------------------------------------------------------------------
void ConvolutionSampler3D::test(const std::string &label,
                                const std::vector<Spike> &input_spike,
                                const Tensor<Time> &input_time,
                                std::vector<Spike> &output_spike)
{
    (void)input_time;

    ensure_state_allocated();

    // When epoch was forced to 0 (weights pre-loaded), train() is never
    // called, so we must load weights here on the first inference call.
    if (!_model_path_local.empty() && !_weights_loaded)
        try_load_weights(label);

    Tensor<float> &w  = parameter<Tensor<float>>("w").get();
    Tensor<float> &th = parameter<Tensor<float>>("th").get();
    const bool inhibition = parameter<bool>("inhibition").get();

    const size_t D = depth();

    // Reset state.
    std::fill(std::begin(_a_local),   std::end(_a_local),   0.0f);
    std::fill(std::begin(_inh_local), std::end(_inh_local), false);

    // Build z-chunks. Each chunk is owned by exactly one task.
    size_t nthreads = std::min<size_t>(_num_threads, D);
    if (nthreads == 0) nthreads = 1;

    std::vector<size_t> chunk_starts(nthreads + 1, 0);
    for (size_t t = 0; t <= nthreads; ++t)
        chunk_starts[t] = (D * t) / nthreads;

    // Each task index processes filters [chunk_starts[t], chunk_starts[t+1]).
    std::vector<size_t> task_idx(nthreads);
    std::iota(task_idx.begin(), task_idx.end(), 0);

    std::vector<std::vector<Spike>> per_thread_out(nthreads);

    std::for_each(std::execution::par, task_idx.begin(), task_idx.end(),
        [&](size_t t) {
            const size_t z_lo = chunk_starts[t];
            const size_t z_hi = chunk_starts[t + 1];
            std::vector<Spike> &local_out = per_thread_out[t];

            // Pre-fetch a forward-output buffer reused across spikes.
            std::vector<std::tuple<uint16_t, uint16_t, uint16_t,
                                   uint16_t, uint16_t, uint16_t>> output_positions;

            for (const Spike &spike : input_spike)
            {
                output_positions.clear();
                this->forward(spike.x, spike.y, spike.k, output_positions);

                for (const auto &entry : output_positions)
                {
                    uint16_t x   = std::get<0>(entry);
                    uint16_t y   = std::get<1>(entry);
                    uint16_t k   = std::get<2>(entry);
                    uint16_t w_x = std::get<3>(entry);
                    uint16_t w_y = std::get<4>(entry);
                    uint16_t w_k = std::get<5>(entry);

                    for (size_t z = z_lo; z < z_hi; ++z)
                    {
                        if (inhibition && _inh_local.at(x, y, z, k))
                            continue;

                        // Disjoint write: only this thread touches z in [z_lo, z_hi).
                        _a_local.at(x, y, z, k) += w.at(w_x, w_y, spike.z, z, w_k);

                        if (_a_local.at(x, y, z, k) >= th.at(z))
                        {
                            local_out.emplace_back(spike.time, x, y,
                                                   static_cast<uint16_t>(z), k);
                            _inh_local.at(x, y, z, k) = true;
                        }
                    }
                }
            }
        });

    // Merge thread-local outputs.
    size_t total = 0;
    for (auto &v : per_thread_out) total += v.size();
    output_spike.reserve(output_spike.size() + total);
    for (auto &v : per_thread_out)
        output_spike.insert(output_spike.end(), v.begin(), v.end());
}
