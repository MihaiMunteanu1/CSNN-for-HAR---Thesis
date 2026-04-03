#pragma once

#include "layer/Convolution3D.h"

namespace layer {

/**
 * ConvolutionSampler3D
 * --------------------
 * Thin wrapper around Convolution3D, introduced to keep a separate
 * semantic layer name for sampler-driven experiments (HOG/Random),
 * without modifying the original Convolution3D implementation.
 *
 * Behavior is identical to Convolution3D and supports the same parameters,
 * including:
 *   - sampler: Sampler (HOGSampler3D / RandomSampler3D)
 *   - stdp
 *   - w, th, etc.
 */
    class ConvolutionSampler3D : public Convolution3D {
    public:
        ConvolutionSampler3D();

        ConvolutionSampler3D(size_t filter_number, size_t filter_width, size_t filter_height, size_t filter_depth,
                             std::string model_path = "",
                             size_t stride_x = 1, size_t stride_y = 1, size_t stride_k = 1,
                             size_t padding_x = 0, size_t padding_y = 0, size_t padding_k = 0);
    };

} // namespace layer