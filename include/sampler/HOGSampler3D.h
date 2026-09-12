#ifndef _HOG_SAMPLER_3D_H
#define _HOG_SAMPLER_3D_H

#include "Sampler.h"
#include <map>
#include <vector>
#include <string>
#include <utility>

namespace sampler
{
	/**
	 * Samples patches guided by pre-computed HOG person bounding boxes.
	 *
	 * Uses bounding box data loaded by VideoKTH_3D to sample spatial positions
	 * inside detected persons. The bounding boxes are stored in the JSON in
	 * original pixel coordinates (e.g. 80x60). For deeper layers (after
	 * convolutions / pooling) the spatial feature map is scaled down. The
	 * sampler applies a (cum_stride_x, cum_stride_y) mapping to transform
	 * bboxes from pixel space into the current layer's input space.
	 *
	 * cum_stride_x is the cumulative stride along dim(0) (rows / image y axis)
	 * from the original input up to, but not including, the current layer.
	 * cum_stride_y is the cumulative stride along dim(1) (cols / image x axis).
	 *
	 * For the first convolutional layer, both are 1. For a layer sitting after
	 * conv1(stride=1) + pool1(stride=2), both become 2, and so on.
	 */
	class HOGSampler3D : public Sampler
	{

	public:
		HOGSampler3D();
		HOGSampler3D(size_t cum_stride_x, size_t cum_stride_y);

		virtual Patch3D sample(const Tensor<float> &sample,
							   size_t width, size_t height, size_t conv_depth,
							   size_t filter_width, size_t filter_height, size_t filter_conv_depth,
							   size_t current_index,
							   std::default_random_engine &rng) override;

	private:
		struct BoundingBox
		{
			bool has_person;
			size_t min_x;
			size_t max_x;
			size_t min_y;
			size_t max_y;
		};

		std::pair<size_t, size_t> sample_point_inside_person(
			size_t W, size_t H, size_t fw, size_t fh,
			std::default_random_engine &rng, size_t current_index, size_t temporal_index);

		void ensure_cache_built();

		std::map<size_t, std::vector<BoundingBox>> _person_box_cache;
		bool _cache_built;

		// Cumulative stride from the original (pixel) input to this layer's input,
		// per spatial axis. dim(0) <-> cum_stride_x, dim(1) <-> cum_stride_y.
		size_t _cum_stride_x;
		size_t _cum_stride_y;
	};

}

#endif
