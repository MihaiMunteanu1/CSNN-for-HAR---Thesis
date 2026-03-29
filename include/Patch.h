#ifndef _PATCH_H
#define _PATCH_H

#include <cstddef>

/**
 * @brief Patch2D represents a 2D sampling location (x, y) for spatial data (images).
 */
struct Patch2D
{
	size_t x;
	size_t y;

	Patch2D() : x(0), y(0) {}
	Patch2D(size_t x, size_t y) : x(x), y(y) {}
};

/**
 * @brief Patch3D represents a 3D sampling location (x, y, k) for spatio-temporal data (video).
 * k is the temporal dimension index.
 */
struct Patch3D
{
	size_t x;
	size_t y;
	size_t k;

	Patch3D() : x(0), y(0), k(0) {}
	Patch3D(size_t x, size_t y, size_t k) : x(x), y(y), k(k) {}
};

#endif
