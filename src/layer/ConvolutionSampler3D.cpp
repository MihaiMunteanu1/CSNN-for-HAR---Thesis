#include "layer/ConvolutionSampler3D.h"

using namespace layer;

// Register this layer type so it can be created/configured like other layers.
static RegisterClassParameter<ConvolutionSampler3D, LayerFactory> _register("ConvolutionSampler3D");

ConvolutionSampler3D::ConvolutionSampler3D()
        : Convolution3D()
{
}

ConvolutionSampler3D::ConvolutionSampler3D(size_t filter_number, size_t filter_width, size_t filter_height, size_t filter_depth,
                                           std::string model_path,
                                           size_t stride_x, size_t stride_y, size_t stride_k,
                                           size_t padding_x, size_t padding_y, size_t padding_k)
        : Convolution3D(filter_number, filter_width, filter_height, filter_depth, model_path,
                        stride_x, stride_y, stride_k, padding_x, padding_y, padding_k)
{
}