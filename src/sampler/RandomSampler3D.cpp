#include "sampler/RandomSampler3D.h"

using namespace sampler;

static RegisterClassParameter<RandomSampler3D, SamplerFactory> _register("RandomSampler3D");

RandomSampler3D::RandomSampler3D() : Sampler(_register)
{
}

Patch3D RandomSampler3D::sample(const Tensor<float> &sample,
								size_t width, size_t height, size_t conv_depth,
								size_t filter_width, size_t filter_height, size_t filter_conv_depth,
								size_t current_index,
								std::default_random_engine &rng)
{
	size_t x = 0;
	size_t y = 0;
	size_t k = 0;

	if (filter_width < width)
	{
		std::uniform_int_distribution<size_t> rand_x(0, width - filter_width);
		x = rand_x(rng);
	}
	if (filter_height < height)
	{
		std::uniform_int_distribution<size_t> rand_y(0, height - filter_height);
		y = rand_y(rng);
	}
	if (filter_conv_depth < conv_depth)
	{
		std::uniform_int_distribution<size_t> rand_k(0, conv_depth - filter_conv_depth);
		k = rand_k(rng);
	}

	return Patch3D(x, y, k);
}
