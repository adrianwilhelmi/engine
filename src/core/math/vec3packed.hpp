#pragma once

namespace engine::math{

struct Vec3Packed{
	float x,y,z;

	constexpr Vec3Packed() : x(0), y(0), z(0) {}
	constexpr Vec3Packed(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}
};

} //namespace engine::math
