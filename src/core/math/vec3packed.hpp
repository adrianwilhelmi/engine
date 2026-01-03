#pragma once

namespace engine::math{

struct Vec3Packed{
	float x,y,z;

	constexpr Vec3Packed() : x(0), y(0), z(0) {}
	constexpr Vec3Packed(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}

	FORCE_INLINE const float& operator[](int i) const {
		if(i == 0) return x;
		else if(i == 1) return y;
		else if(i == 2) return z;
		throw std::runtime_error("index oob while indexing Vec3Packed");
	}

	FORCE_INLINE float& operator[](int i) {
		if(i == 0) return x;
		else if(i == 1) return y;
		else if(i == 2) return z;
		throw std::runtime_error("index oob while indexing Vec3Packed");
	}
};

} //namespace engine::math
