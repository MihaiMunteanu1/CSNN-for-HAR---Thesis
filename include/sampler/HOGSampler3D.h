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
	 * @brief Samples patches guided by pre-computed HOG person bounding boxes.
	 *
	 * Uses bounding box data loaded by VideoKTH_3D to sample spatial positions
	 * inside detected persons. Falls back to random sampling when no detection
	 * is available for a given frame.
	 */
	class HOGSampler3D : public Sampler
	{

	public:
		HOGSampler3D();

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
	};

}

#endif
