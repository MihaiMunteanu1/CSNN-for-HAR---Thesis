#ifndef _SAMPLER_H
#define _SAMPLER_H

#include "ClassParameter.h"
#include "Patch.h"
#include "Tensor.h"
#include <random>

/**
 * @brief Abstract base class for patch sampling strategies.
 *
 * A Sampler controls how patches are selected from the input during training.
 * Concrete implementations define different strategies (random, HOG-guided, etc.)
 * and are injected into the Convolution layer as a parameter, keeping
 * the convolution class generic.
 */
class Sampler : public ClassParameter
{

public:
	template <typename T, typename Factory>
	Sampler(const RegisterClassParameter<T, Factory> &registration) : ClassParameter(registration) {}

	/**
	 * @brief Sample a 3D patch location from the given input sample.
	 *
	 * @param sample The input tensor for the current training sample
	 * @param width The output width of the convolutional layer
	 * @param height The output height of the convolutional layer
	 * @param conv_depth The output temporal depth of the convolutional layer
	 * @param filter_width Width of the convolutional filter
	 * @param filter_height Height of the convolutional filter
	 * @param filter_conv_depth Temporal depth of the convolutional filter
	 * @param current_index Index of the current sample in the dataset
	 * @param rng Random number generator
	 * @return Patch3D The sampled patch location (x, y, k)
	 */
	virtual Patch3D sample(const Tensor<float> &sample,
						   size_t width, size_t height, size_t conv_depth,
						   size_t filter_width, size_t filter_height, size_t filter_conv_depth,
						   size_t current_index,
						   std::default_random_engine &rng) = 0;

	virtual void adapt_parameters(float factor) {}
};

class SamplerFactory : public ClassParameterFactory<Sampler, SamplerFactory>
{

public:
	SamplerFactory() : ClassParameterFactory<Sampler, SamplerFactory>("Sampler") {}
};

#endif
