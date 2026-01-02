#pragma once

#include<iostream>
#include<iomanip>
#include<cassert>
#include<cstdint>

#include"simd_backend.hpp"
#include"vec3packed.hpp"

namespace engine::math{

struct alignas(16) Vec3{
	union{
		simd::Register reg;

		#pragma pack(push,1)

		#if defined(__GNUC__) || defined(__clang__)
			#pragma GCC diagnostic push
			#pragma GCC diagnostic ignored "-Wpedantic"
		#endif

		struct{ float x, y, z, _padding; };
		#pragma pack(pop)

		#if defined(__GNUC__) || defined(__clang__)
			#pragma GCC diagnostic pop
		#endif
	};

	FORCE_INLINE Vec3(){
		reg = simd::set(0,0,0,0);
	}

	FORCE_INLINE explicit Vec3(const float val){
		reg = simd::set1(val);
	}

	FORCE_INLINE Vec3(float _x, float _y, float _z){
		reg = simd::set(_x,_y,_z,0.0f);
	}

	FORCE_INLINE explicit Vec3(const Vec3Packed& p){
		reg = simd::set(p.x, p.y, p.z, 0.0f);
	}

	FORCE_INLINE explicit Vec3(simd::Register r) : reg(r) {}

	[[nodiscard]] FORCE_INLINE Vec3Packed pack() const{
		return Vec3Packed(x,y,z);
	};

	[[nodiscard]] FORCE_INLINE Vec3 operator+(const Vec3& other) const{
		return Vec3(simd::add(reg, other.reg));
	}

	[[nodiscard]] FORCE_INLINE Vec3 operator-(const Vec3& other) const{
		return Vec3(simd::sub(reg, other.reg));
	}

	[[nodiscard]] FORCE_INLINE Vec3 operator-() const{
		return Vec3(simd::sub(simd::set1(0.0f), reg));
	}

	[[nodiscard]] FORCE_INLINE Vec3 operator*(const float scalar) const{
		return Vec3(simd::mul(reg, scalar));
	}

	[[nodiscard]] FORCE_INLINE Vec3 operator*(const Vec3& other) const{
		return Vec3(simd::mul(reg, other.reg));
	}

	FORCE_INLINE Vec3& operator+=(const Vec3& other){
		reg = simd::add(reg, other.reg);
		return *this;
	}

	FORCE_INLINE Vec3& operator-=(const Vec3& other){
		reg = simd::sub(reg, other.reg);
		return *this;
	}

	FORCE_INLINE Vec3& operator*=(const float scalar){
		reg = simd::mul(reg, scalar);
		return *this;
	}

	FORCE_INLINE const float& operator[](uint16_t i) const {
		assert(i < 3 && "index oob for Vec3");
		return (&x)[i];
	}

	FORCE_INLINE float& operator[](uint16_t i) {
		assert(i < 3 && "index oob for Vec3");
		return (&x)[i];
	}

	[[nodiscard]] FORCE_INLINE float dot(const Vec3& other) const{
		return simd::dot3(reg, other.reg);
	}

	[[nodiscard]] FORCE_INLINE Vec3 cross(const Vec3& other) const{
		return Vec3(simd::cross3(reg, other.reg));
	}

	[[nodiscard]] FORCE_INLINE float l2() const{
		return std::sqrt(simd::dot3(reg, reg));
	}

	[[nodiscard]] FORCE_INLINE float length_sq() const{
		return simd::dot3(reg,reg);
	}

	[[nodiscard]] FORCE_INLINE Vec3 normalized() const{
		simd::Register len_sq = simd::dot3_splat(reg,reg);
		simd::Register inv_len = simd::rsqrt_accurate(len_sq);
		return Vec3(simd::mul(reg, inv_len));
	}

	[[nodiscard]] FORCE_INLINE Vec3 normalized_fast() const{
		simd::Register len_sq = simd::dot3_splat(reg,reg);
		simd::Register inv_len = simd::rsqrt(len_sq);
		return Vec3(simd::mul(reg, inv_len));
	}

	[[nodiscard]] FORCE_INLINE Vec3 abs() const{
		return Vec3{simd::abs(reg)};
	}

	FORCE_INLINE bool operator==(const Vec3& other) const{
		return simd::equals_xyz(reg, other.reg);
	}

	FORCE_INLINE bool operator!=(const Vec3& other) const{
		return !simd::equals_xyz(reg,other.reg);
	}

	[[nodiscard]] FORCE_INLINE bool is_close(const Vec3& other, float epsilon = 1e-5f) const{
		return simd::is_close_xyz(reg, other.reg, epsilon);
	}

	[[nodiscard]] static FORCE_INLINE Vec3 lerp(
			const Vec3& a,
			const Vec3& b,
			float t){
		simd::Register t_vec = simd::set1(t);
		simd::Register diff = simd::sub(b.reg, a.reg);
		simd::Register res = simd::add(a.reg, simd::mul(diff, t_vec));
		return Vec3(res);
	}
};

static_assert(sizeof(Vec3) == 16, "Vec3 size must be exactly 16 byes");
static_assert(alignof(Vec3) == 16, "Vec3 alignment must be 16 byes");
static_assert(offsetof(Vec3,Vec3::x) == 0, "Vec3::x must be at offset 0");
static_assert(offsetof(Vec3,Vec3::reg) == 0, "Vec3::reg must be at offset 0");
static_assert(offsetof(Vec3,Vec3::y) == sizeof(float), "Vec3: Gap between x and y");
static_assert(offsetof(Vec3,Vec3::z) == 2*sizeof(float), "Vec3: Gap between y and z");

inline std::ostream& operator<<(std::ostream& os, const Vec3& v){
	os << "Vec3(\n\t" << v.x << ",\n\t" << v.y << ",\n\t" << v.z << "\n)\n";
	return os;
}

[[nodiscard]] FORCE_INLINE Vec3 operator*(float s, const Vec3& v){
	return v * s;
}

} // namespace engine::math
