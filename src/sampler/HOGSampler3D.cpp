#include "sampler/HOGSampler3D.h"
#include "dataset/VideoKTH_3D.h"
#include <iostream>
#include <cmath>
#include <algorithm>

using namespace sampler;

static RegisterClassParameter<HOGSampler3D, SamplerFactory> _register("HOGSampler3D");

HOGSampler3D::HOGSampler3D() : Sampler(_register), _cache_built(false)
{
}

void HOGSampler3D::ensure_cache_built()
{
	if (_cache_built)
		return;

	const auto &hog_data = dataset::VideoKTH_3D::get_hog_data();

	if (hog_data.empty())
	{
		std::cout << "HOGSampler3D: No HOG data available, will use random sampling fallback." << std::endl;
	}
	else
	{
		std::cout << "HOGSampler3D: Bounding box data ready ("
				  << hog_data.size() << " videos with HOG data)" << std::endl;
	}

	_cache_built = true;
}

std::pair<size_t, size_t> HOGSampler3D::sample_point_inside_person(
	size_t W, size_t H, size_t fw, size_t fh,
	std::default_random_engine &rng, size_t current_index, size_t temporal_index)
{
	if (_person_box_cache.find(current_index) != _person_box_cache.end())
	{
		const std::vector<BoundingBox> &boxes = _person_box_cache[current_index];

		if (temporal_index < boxes.size() && boxes[temporal_index].has_person)
		{
			const BoundingBox &box = boxes[temporal_index];
			std::uniform_int_distribution<size_t> dist_x(box.min_x, box.max_x);
			std::uniform_int_distribution<size_t> dist_y(box.min_y, box.max_y);
			return {dist_x(rng), dist_y(rng)};
		}

		// Fallback: nearest detected frame
		int best_frame = -1;
		int min_dist = static_cast<int>(boxes.size());
		for (size_t f = 0; f < boxes.size(); f++)
		{
			if (boxes[f].has_person)
			{
				int dist = std::abs(static_cast<int>(f) - static_cast<int>(temporal_index));
				if (dist < min_dist)
				{
					min_dist = dist;
					best_frame = static_cast<int>(f);
				}
			}
		}
		if (best_frame >= 0)
		{
			const BoundingBox &box = boxes[best_frame];
			std::uniform_int_distribution<size_t> dist_x(box.min_x, box.max_x);
			std::uniform_int_distribution<size_t> dist_y(box.min_y, box.max_y);
			return {dist_x(rng), dist_y(rng)};
		}
	}
	else
	{
		// Build cache for this sample from VideoKTH_3D's HOG data
		const auto &hog_data = dataset::VideoKTH_3D::get_hog_data();

		if (!hog_data.empty())
		{
			static std::vector<std::pair<std::string, size_t>> sample_to_video_group;
			static bool mapping_built = false;

			if (!mapping_built)
			{
				for (const auto &[video_key, vdata] : hog_data)
				{
					for (size_t g = 0; g < vdata.groups.size(); g++)
					{
						sample_to_video_group.push_back({video_key, g});
					}
				}
				mapping_built = true;
			}

			if (current_index < sample_to_video_group.size())
			{
				const auto &[video_key, group_idx] = sample_to_video_group[current_index];
				const auto &vdata = hog_data.at(video_key);
				const auto &group = vdata.groups[group_idx];

				size_t total_frames = group.frames.size();
				std::vector<BoundingBox> boxes(total_frames, {false, 0, 0, 0, 0});

				for (size_t f = 0; f < total_frames; f++)
				{
					const auto &fb = group.frames[f];
					if (!fb.bboxes.empty())
					{
						auto [bx, by, bw, bh] = fb.bboxes[0];

						size_t box_width = static_cast<size_t>(bw);
						size_t box_height = static_cast<size_t>(bh);

						if (box_width >= fw && box_height >= fh)
						{
							size_t min_x = std::max<size_t>(0, static_cast<size_t>(std::max(0, bx)));
							size_t max_x = std::min<size_t>(W - fw, static_cast<size_t>(bx + bw) - fw);
							size_t min_y = std::max<size_t>(0, static_cast<size_t>(std::max(0, by)));
							size_t max_y = std::min<size_t>(H - fh, static_cast<size_t>(by + bh) - fh);

							if (max_x >= min_x && max_y >= min_y)
							{
								boxes[f] = {true, min_x, max_x, min_y, max_y};
							}
						}
					}
				}

				_person_box_cache[current_index] = boxes;

				if (temporal_index < boxes.size() && boxes[temporal_index].has_person)
				{
					const BoundingBox &box = boxes[temporal_index];
					std::uniform_int_distribution<size_t> dist_x(box.min_x, box.max_x);
					std::uniform_int_distribution<size_t> dist_y(box.min_y, box.max_y);
					return {dist_x(rng), dist_y(rng)};
				}

				// Fallback: nearest detected frame
				int best_frame = -1;
				int min_dist = static_cast<int>(boxes.size());
				for (size_t f = 0; f < boxes.size(); f++)
				{
					if (boxes[f].has_person)
					{
						int dist = std::abs(static_cast<int>(f) - static_cast<int>(temporal_index));
						if (dist < min_dist)
						{
							min_dist = dist;
							best_frame = static_cast<int>(f);
						}
					}
				}
				if (best_frame >= 0)
				{
					const BoundingBox &box = boxes[best_frame];
					std::uniform_int_distribution<size_t> dist_x(box.min_x, box.max_x);
					std::uniform_int_distribution<size_t> dist_y(box.min_y, box.max_y);
					return {dist_x(rng), dist_y(rng)};
				}
			}
		}
	}

	// Ultimate fallback: random sampling
	std::uniform_int_distribution<size_t> fallback_x(0, W - fw);
	std::uniform_int_distribution<size_t> fallback_y(0, H - fh);
	return {fallback_x(rng), fallback_y(rng)};
}

Patch3D HOGSampler3D::sample(const Tensor<float> &sample,
							  size_t width, size_t height, size_t conv_depth,
							  size_t filter_width, size_t filter_height, size_t filter_conv_depth,
							  size_t current_index,
							  std::default_random_engine &rng)
{
	ensure_cache_built();

	size_t k = 0;

	// Select temporal index randomly
	if (filter_conv_depth < conv_depth)
	{
		std::uniform_int_distribution<size_t> rand_k(0, conv_depth - filter_conv_depth);
		k = rand_k(rng);
	}

	size_t x = 0;
	size_t y = 0;

	if (filter_width < width && filter_height < height)
	{
		size_t input_depth = sample.shape().dim(2);
		if (input_depth <= 3)
		{
			// Use pre-computed bounding boxes for person-guided sampling
			auto [sample_x, sample_y] = sample_point_inside_person(
				width, height, filter_width, filter_height,
				rng, current_index, k);
			x = sample_x;
			y = sample_y;
		}
		else
		{
			// For deep feature maps, fall back to random
			std::uniform_int_distribution<size_t> rand_x(0, width - filter_width);
			std::uniform_int_distribution<size_t> rand_y(0, height - filter_height);
			x = rand_x(rng);
			y = rand_y(rng);
		}
	}

	return Patch3D(x, y, k);
}
