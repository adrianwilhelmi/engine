#pragma once

#include<iostream>
#include<iomanip>
#include<cassert>

#include"simd_backend.hpp"
#include"vec3packed.hpp"

namespace engine::math{

struct alignas(8) Vec2{
	float x;
	float y;

};

static_assert(sizeof(Vec2) == 8, "Vec2 size must be exactly 8 bytes");
static_assert(alignof(Vec2) == 8, "Vec2 alignment must be 8 byes");
