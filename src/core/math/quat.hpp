#pragma once

#include"simd_backend.hpp"

namespace engine::math{

struct alignas(16) Quat{
	simd::Register reg;

	FORCE_INLINE Quat() : reg(simd::set(0,0,0,1)) {}
	FORCE_INLINE explicit Quat(simd::Register r) : reg(r) {}
	FORCE_INLINE Quat(float x, float y, float z, float w) : 
		reg(simd::set(x,y,z,w)) {}

	[[nodiscard]] FORCE_INLINE static Quat identity() {
		return Quat(0,0,0,1);
	}

	/*
	[[nodiscard]] FORCE_INLINE Quat conjugated() const {

	}
	*/

	[[nodiscard]] FORCE_INLINE Quat normalized() const{
		simd::Register dot = simd::dot4_splat(reg,reg);
		simd::Register inv_len = simd::rsqrt_accurate(dot);
		return Quat(simd::mul(reg, inv_len));
	}
};

} // namespace engine::math
