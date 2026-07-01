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

	if (dataset::VideoKTH_3D::has_bboxes())
	{
		std::cout << "HOGSampler3D: per-sample bboxes available "
				  << "(NPY cache mode, cum_stride=("
				  << _cum_stride_x << "," << _cum_stride_y << "))" << std::endl;
	}
	else
	{
		const auto &hog_data = dataset::VideoKTH_3D::get_hog_data();
		if (!hog_data.empty())
		{
			std::cout << "HOGSampler3D: legacy HOG bbox data ready ("
					  << hog_data.size() << " videos)" << std::endl;
		}
		else
		{
			std::cout << "HOGSampler3D: no bbox data found, "
					  << "uniform random sampling will be used." << std::endl;
		}
	}

	_cache_built = true;
}

std::pair<size_t, size_t> HOGSampler3D::sample_point_inside_person(
		size_t W, size_t H, size_t fw, size_t fh,
		std::default_random_engine &rng, size_t current_index, size_t temporal_index)
{

	if (dataset::VideoKTH_3D::has_bboxes())
	{
		auto [bx, by, bw, bh] = dataset::VideoKTH_3D::get_train_sample_bbox(
			current_index, temporal_index);
		if (bw > 0 && bh > 0)
		{
			//W=80,H=60, fw,fh=3, ex bbox: fm_x = 20, fm_y = 5, fm_w = 30, fm_h = 50

			// pixel-space bbox -> current feature-map space.
			size_t fm_x = static_cast<size_t>(bx) / _cum_stride_y;
			size_t fm_y = static_cast<size_t>(by) / _cum_stride_x;
			size_t fm_w = static_cast<size_t>(bw) / _cum_stride_y;
			size_t fm_h = static_cast<size_t>(bh) / _cum_stride_x;

			// valid filter origin range: [fm_x, fm_x + fm_w - fw] in cols,
			// [fm_y, fm_y + fm_h - fh] in rows
			size_t x_lo = std::min(fm_x, (W > fw) ? (W - fw) : size_t{0}); //marginea stanga
			size_t y_lo = std::min(fm_y, (H > fh) ? (H - fh) : size_t{0}); //marginea de sus
			size_t x_hi_raw = (fm_w >= fw) ? (fm_x + fm_w - fw) : fm_x;
			size_t y_hi_raw = (fm_h >= fh) ? (fm_y + fm_h - fh) : fm_y;
			size_t x_hi = std::min(x_hi_raw, (W > fw) ? (W - fw) : size_t{0}); //marginea dreapta
			size_t y_hi = std::min(y_hi_raw, (H > fh) ? (H - fh) : size_t{0}); //marginea de jos

			if (x_hi >= x_lo && y_hi >= y_lo)
			{
				std::uniform_int_distribution<size_t> dist_x(x_lo, x_hi);
				std::uniform_int_distribution<size_t> dist_y(y_lo, y_hi);
				return {dist_x(rng), dist_y(rng)};
			}
		}
	}

	std::uniform_int_distribution<size_t> dist_x(0, W - fw);
	std::uniform_int_distribution<size_t> dist_y(0, H - fh);
	return {dist_x(rng), dist_y(rng)};
}

Patch3D HOGSampler3D::sample(const Tensor<float> &sample,
							  size_t width, size_t height, size_t conv_depth,
							  size_t filter_width, size_t filter_height, size_t filter_conv_depth,
							  size_t current_index,
							  std::default_random_engine &rng)
{
	ensure_cache_built();

	if (current_index == 0)
		_person_box_cache.clear();

	size_t k = 0;

	if (filter_conv_depth < conv_depth)
	{
		std::uniform_int_distribution<size_t> rand_k(0, conv_depth - filter_conv_depth);
		k = rand_k(rng);
	}

	size_t x = 0;
	size_t y = 0;

	if (filter_width < width && filter_height < height)
	{

		auto [sample_x, sample_y] = sample_point_inside_person(
			width, height, filter_width, filter_height,
			rng, current_index, k);
		x = sample_x;
		y = sample_y;
	}

	return Patch3D(x, y, k);
}
