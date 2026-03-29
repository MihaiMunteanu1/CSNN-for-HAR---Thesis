#ifndef _RANDOM_SAMPLER_3D_H
#define _RANDOM_SAMPLER_3D_H

#include "Sampler.h"

namespace sampler
{
	/**
	 * @brief Samples patches at uniformly random spatial and temporal positions.
	 * This is the default sampling strategy used in Convolution3D.
	 */
	class RandomSampler3D : public Sampler
	{

	public:
		RandomSampler3D();

		virtual Patch3D sample(const Tensor<float> &sample,
							   size_t width, size_t height, size_t conv_depth,
							   size_t filter_width, size_t filter_height, size_t filter_conv_depth,
							   size_t current_index,
							   std::default_random_engine &rng) override;
	};

}

#endif
