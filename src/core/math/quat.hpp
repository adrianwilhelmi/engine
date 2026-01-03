#pragma once

#include"simd_backend.hpp"

namespace engine::math{

struct alignas(16) Quat{
	simd::Register reg;

	FORCE_INLINE Quat() : reg(simd::set(0,0,0,1)) {}
	FORCE_INLINE explicit Quat(simd::Register r) : reg(r) {}
	FORCE_INLINE Quat(float x, float y, float z, float w) : 
		reg(simd::set(x,y,z,w)) {}
	FORCE_INLINE Quat(float val) : reg(simd::set(0,0,0,w)) {}

	[[nodiscard]] FORCE_INLINE static Quat identity() {
		return Quat();
	}

	[[nodiscard]] FORCE_INLINE Quat conjugated() const {
		return simd::quat_conjugate(reg);
	}

	[[nodiscard]] FORCE_INLINE Quat normalized() const{
		simd::Register dot = simd::dot4_splat(reg,reg);
		simd::Register inv_len = simd::rsqrt_accurate(dot);
		return Quat(simd::mul(reg, inv_len));
	}

	[[nodiscard]] FORCE_INLINE float get_x() const{
		return simd::x(reg);
	}

	[[nodiscard]] FORCE_INLINE float get_y() const{
		return simd::y(reg);
	}

	[[nodiscard]] FORCE_INLINE float get_z() const{
		return simd::z(reg);
	}

	[[nodiscard]] FORCE_INLINE float get_w() const{
		return simd::w(reg);
	}

	FORCE_INLINE void set_x(const float val){
		reg = simd::set_x(reg, val);
	}

	FORCE_INLINE void set_y(const float val){
		reg = simd::set_y(reg, val);
	}

	FORCE_INLINE void set_z(const float val){
		reg = simd::set_z(reg, val);
	}

	FORCE_INLINE void set_w(const float val){
		reg = simd::set_w(reg, val);
	}

	[[nodiscard]] FORCE_INLINE Quat operator*(const Quat& other) const{
		simd::Register oreg = other.reg;
		simd::Register wwww_a = simd::splat<3>(reg);
		simd::Register wwww_b = simd::splat<3>(oreg);

		simd::Register res = simd::mul(wwww_a, oreg);
		res = simd::fmadd(wwww_b, reg, res);

		simd::Register v_cross = simd::cross3(reg,oreg);
		res = simd::add(res, v_cross);

		float d3 = simd::dot3(reg,oreg);
		float wa = simd::w(reg);
		float wb = simd::w(oreg);
		res = simd::set_w(res, (wa*wb) - d3);

		return Quat(res);
	}

	[[nodiscard]] FORCE_INLINE Vec3 rotate(const Vec3& v) const{
		simd::Register q_w = simd::splat<3>(reg);
		simd::Register t = simd::mul(simd::cross3(reg,v.reg), simd::set1(2.0f));

		simd::Register res = fmadd(q_w, t, v.reg);
		res = simd::add(res, simd::cross3(reg,t));

		return Quat(res);
	}

};

} // namespace engine::math
