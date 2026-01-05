#pragma once

#include"simd_backend.hpp"

namespace engine::math{

struct alignas(16) Quat{
	simd::Register reg;

	FORCE_INLINE Quat() : reg(simd::set(0,0,0,1)) {}
	FORCE_INLINE explicit Quat(simd::Register r) : reg(r) {}
	FORCE_INLINE Quat(float x, float y, float z, float w) : 
		reg(simd::set(x,y,z,w)) {}
	FORCE_INLINE Quat(float val) : reg(simd::set(0,0,0,val)) {}

	[[nodiscard]] FORCE_INLINE static Quat identity() {
		return Quat();
	}

	[[nodiscard]] FORCE_INLINE static Quat from_axis_angle(
			const Vec3& axis,
			float radians){
		float half_angle = radians * 0.5f;
		float s = std::sin(half_angle);
		float c = std::cos(half_angle);
		return Quat(axis.get_x() * s, axis.get_y() * s, axis.get_z() * s, c);
	}

	[[nodiscard]] FORCE_INLINE Quat conjugated() const {
		return Quat(simd::quat_conjugate(reg));
	}

	[[nodiscard]] FORCE_INLINE float l2() const{
		return std::sqrt(simd::dot4(reg, reg));
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
		return Quat(simd::quat_mul(reg, other.reg));
	}

	[[nodiscard]] FORCE_INLINE Vec3 rotate(const Vec3& v) const{
		simd::Register q_w = simd::splat<3>(reg);
		simd::Register t = simd::mul(simd::cross3(reg,v.reg), simd::set1(2.0f));

		simd::Register res = simd::fmadd(q_w, t, v.reg);
		res = simd::add(res, simd::cross3(reg,t));

		return Vec3(res);
	}

	[[nodiscard]] FORCE_INLINE Mat4 to_mat4() const{
		simd::Register q2 = simd::add(reg,reg);

		simd::Register x_splat = simd::splat<0>(reg);
		simd::Register y_splat = simd::splat<1>(reg);
		simd::Register z_splat = simd::splat<2>(reg);
		simd::Register w_splat = simd::splat<3>(reg);

		simd::Register tmp0 = simd::mul(x_splat, q2);
		simd::Register tmp1 = simd::mul(y_splat, q2);
		simd::Register tmp2 = simd::mul(z_splat, q2);
		simd::Register tmp3 = simd::mul(w_splat, q2);

		simd::Register c0 = simd::set(
			1.0f - simd::x(simd::mul(y_splat, simd::splat<1>(q2)))
				- simd::x(simd::mul(z_splat, simd::splat<2>(q2))),
			simd::y(tmp0) + simd::z(tmp3),
			simd::z(tmp0) - simd::y(tmp3),
			0.0f
		);

		simd::Register c1 = simd::set(
			simd::y(tmp0) - simd::z(tmp3),
			1.0f - simd::x(tmp0) - simd::z(tmp2),
			simd::z(tmp1) + simd::x(tmp3),
			0.0f
		);

		simd::Register c2 = simd::set(
			simd::z(tmp0) + simd::y(tmp3),
			simd::z(tmp1) - simd::x(tmp3),
			1.0f - simd::x(tmp0) - simd::y(tmp1),
			0.0f
		);

		simd::Register c3 = simd::set(0,0,0,1);

		return Mat4(c0,c1,c2,c3);
	}

	[[nodiscard]] FORCE_INLINE static Quat slerp(
			const Quat& q1,
			const Quat& q2,
			float t){
		return Quat(simd::quat_slerp(q1.reg, q2.reg, t));
	}

	[[nodiscard]] FORCE_INLINE static Quat slerp_fast(
			const Quat& q1,
			const Quat& q2,
			float t){
		return Quat(simd::quat_fast_slerp(q1.reg, q2.reg, t));
	}

};

} // namespace engine::math
