#include "sampler/HOGSampler3D.h"
#include "dataset/VideoKTH_3D.h"
#include <iostream>
#include <cmath>
#include <algorithm>

using namespace sampler;

static RegisterClassParameter<HOGSampler3D, SamplerFactory> _register("HOGSampler3D");

HOGSampler3D::HOGSampler3D() : Sampler(_register), _cache_built(false),
	_cum_stride_x(1), _cum_stride_y(1)
{
}

HOGSampler3D::HOGSampler3D(size_t cum_stride_x, size_t cum_stride_y)
	: Sampler(_register), _cache_built(false),
	  _cum_stride_x(cum_stride_x == 0 ? 1 : cum_stride_x),
	  _cum_stride_y(cum_stride_y == 0 ? 1 : cum_stride_y)
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
	auto cache_it = _person_box_cache.find(current_index);
	if (cache_it != _person_box_cache.end())
	{
		const std::vector<BoundingBox> &boxes = cache_it->second;

		if (temporal_index < boxes.size() && boxes[temporal_index].has_person)
		{
			const BoundingBox &box = boxes[temporal_index];
			std::uniform_int_distribution<size_t> dist_x(box.min_x, box.max_x);
			std::uniform_int_distribution<size_t> dist_y(box.min_y, box.max_y);
			return {dist_x(rng), dist_y(rng)};
		}

		int best_frame = -1;
		int min_dist = static_cast<int>(boxes.size());
		for (size_t f = 0; f < boxes.size(); f++)
		{
			if (!boxes[f].has_person) continue;
			int dist = std::abs(static_cast<int>(f) - static_cast<int>(temporal_index));
			if (dist < min_dist)
			{
				min_dist = dist;
				best_frame = static_cast<int>(f);
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
		const auto &hog_data = dataset::VideoKTH_3D::get_hog_data();
		if (!hog_data.empty())
		{
			const auto &train_map = dataset::VideoKTH_3D::get_train_sample_mapping();
			const auto &test_map  = dataset::VideoKTH_3D::get_test_sample_mapping();

			std::string video_key;
			size_t group_idx = 0;
			bool found = false;

			auto it_tr = train_map.find(current_index);
			if (it_tr != train_map.end())
			{
				video_key = it_tr->second.first;
				group_idx = it_tr->second.second;
				found = true;
			}
			else
			{
				auto it_te = test_map.find(current_index);
				if (it_te != test_map.end())
				{
					video_key = it_te->second.first;
					group_idx = it_te->second.second;
					found = true;
				}
			}

			if (found)
			{
				auto hog_it = hog_data.find(video_key);
				if (hog_it == hog_data.end() || group_idx >= hog_it->second.groups.size())
				{
					std::uniform_int_distribution<size_t> fallback_x(0, W - fw);
					std::uniform_int_distribution<size_t> fallback_y(0, H - fh);
					return {fallback_x(rng), fallback_y(rng)};
				}

				const auto &group = hog_it->second.groups[group_idx];
				std::vector<BoundingBox> boxes(group.frames.size(), {false, 0, 0, 0, 0});

				for (size_t f = 0; f < group.frames.size(); f++)
				{
					const auto &fb = group.frames[f];
					if (fb.bboxes.empty()) continue;

					auto [bx, by, bw, bh] = fb.bboxes[0];

					// JSON stores bboxes in original pixel coordinates.
					// bx, bw run along dim(1) (image x / cols) -> divide by _cum_stride_y
					// by, bh run along dim(0) (image y / rows) -> divide by _cum_stride_x
					// This transforms the bbox into the current layer's input feature-map space.
					size_t pix_min_row = static_cast<size_t>(std::max(0, by));
					size_t pix_max_row = static_cast<size_t>(std::max(0, by + bh));
					size_t pix_min_col = static_cast<size_t>(std::max(0, bx));
					size_t pix_max_col = static_cast<size_t>(std::max(0, bx + bw));

					size_t feat_min_x = pix_min_row / _cum_stride_x;
					size_t feat_max_x = pix_max_row / _cum_stride_x;
					size_t feat_min_y = pix_min_col / _cum_stride_y;
					size_t feat_max_y = pix_max_col / _cum_stride_y;

					// Skip if the transformed bbox is too small to fit the filter.
					if (feat_max_x <= feat_min_x || feat_max_y <= feat_min_y) continue;
					if ((feat_max_x - feat_min_x) < fw || (feat_max_y - feat_min_y) < fh) continue;

					// Valid top-left filter corners. W - fw / H - fh are safe here
					// because sample() only calls us when filter_width < width &&
					// filter_height < height.
					size_t min_x = feat_min_x;
					size_t max_x = std::min<size_t>(W - fw, feat_max_x - fw);
					size_t min_y = feat_min_y;
					size_t max_y = std::min<size_t>(H - fh, feat_max_y - fh);

					if (max_x >= min_x && max_y >= min_y)
						boxes[f] = {true, min_x, max_x, min_y, max_y};
				}

				_person_box_cache[current_index] = boxes;

				if (temporal_index < boxes.size() && boxes[temporal_index].has_person)
				{
					const BoundingBox &box = boxes[temporal_index];
					std::uniform_int_distribution<size_t> dist_x(box.min_x, box.max_x);
					std::uniform_int_distribution<size_t> dist_y(box.min_y, box.max_y);
					return {dist_x(rng), dist_y(rng)};
				}

				int best_frame = -1;
				int min_dist = static_cast<int>(boxes.size());
				for (size_t f = 0; f < boxes.size(); f++)
				{
					if (!boxes[f].has_person) continue;
					int dist = std::abs(static_cast<int>(f) - static_cast<int>(temporal_index));
					if (dist < min_dist)
					{
						min_dist = dist;
						best_frame = static_cast<int>(f);
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

	// Issue #4: clear per-sample bbox cache at the start of each epoch.
	// `current_index` is the per-epoch sample index (0..N-1) and resets to 0
	// every epoch. Without this clear, entries from a previous epoch would
	// shadow new sample-to-bbox mappings (and the train/test mapping counters
	// reset in VideoKTH_3D::reset() would no longer match cached entries).
	if (current_index == 0)
		_person_box_cache.clear();

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
		// Person-guided sampling at every layer. Bboxes are transformed from pixel
		// space into the current layer's feature-map space using _cum_stride_x/y.
		// If no bbox is available the inner function falls back to uniform random.
		auto [sample_x, sample_y] = sample_point_inside_person(
			width, height, filter_width, filter_height,
			rng, current_index, k);
		x = sample_x;
		y = sample_y;
	}

	return Patch3D(x, y, k);
}
