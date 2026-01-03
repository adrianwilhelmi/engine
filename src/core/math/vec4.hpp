#pragma once

#include<iostream>
#include<cassert>

#include"simd_backend.hpp"
#include"vec3.hpp"
#include"vec3packed.hpp"

namespace engine::math{

struct alignas(16) Vec4{
	union{
		simd::Register reg;

		#if defined(__GNUC__) || defined(__clang__)
			#pragma GCC diagnostic push
			#pragma GCC diagnostic ignored "-Wpedantic"
		#endif

		#pragma pack(push,1)
		struct{ float x, y, z, w; };
		#pragma pack(pop)

		#if defined(__GNUC__) || defined(__clang__)
			#pragma GCC diagnostic pop
		#endif
	};

	FORCE_INLINE Vec4(){
		reg = simd::set(0,0,0,0);
	}

	FORCE_INLINE explicit Vec4(const float val){
		reg = simd::set1(val);
	}

	FORCE_INLINE Vec4(float _x, float _y, float _z, float _w){
		reg = simd::set(_x,_y,_z,_w);
	}

	FORCE_INLINE explicit Vec4(const Vec3Packed& p){
		reg = simd::set(p.x, p.y, p.z, 0.0f);
	}

	FORCE_INLINE explicit Vec4(const Vec3& p){
		reg = simd::set(p.x, p.y, p.z, 0.0f);
	}

	FORCE_INLINE explicit Vec4(const Vec3& p, const float val){
		reg = simd::set(p.x, p.y, p.z, val);
	}

	FORCE_INLINE explicit Vec4(simd::Register r) : reg(r) {}

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


	[[nodiscard]] FORCE_INLINE Vec4 operator+(const Vec4& other) const{
		return Vec4(simd::add(reg, other.reg));
	}

	[[nodiscard]] FORCE_INLINE Vec4 operator-(const Vec4& other) const{
		return Vec4(simd::sub(reg, other.reg));
	}

	[[nodiscard]] FORCE_INLINE Vec4 operator-() const{
		return Vec4(simd::sub(simd::set1(0.0f), reg));
	}

	[[nodiscard]] FORCE_INLINE Vec4 operator*(const float scalar) const{
		return Vec4(simd::mul(reg, scalar));
	}

	[[nodiscard]] FORCE_INLINE Vec4 operator*(const Vec4& other) const{
		return Vec4(simd::mul(reg, other.reg));
	}

	FORCE_INLINE Vec4& operator+=(const Vec4& other){
		reg = simd::add(reg, other.reg);
		return *this;
	}

	FORCE_INLINE Vec4& operator-=(const Vec4& other){
		reg = simd::sub(reg, other.reg);
		return *this;
	}

	FORCE_INLINE Vec4& operator*=(const float scalar){
		reg = simd::mul(reg, scalar);
		return *this;
	}

	FORCE_INLINE const float& operator[](int i) const {
		assert(i < 4 && "index oob for Vec4");
		return (&x)[i];
	}

	FORCE_INLINE float& operator[](int i) {
		assert(i < 4 && "index oob for Vec4");
		return (&x)[i];
	}

	[[nodiscard]] FORCE_INLINE float dot(const Vec4& other) const{
		return simd::dot4(reg, other.reg);
	}

	[[nodiscard]] FORCE_INLINE float l2() const{
		return std::sqrt(simd::dot4(reg, reg));
	}

	[[nodiscard]] FORCE_INLINE float length_sq() const{
		return simd::dot4(reg,reg);
	}

	[[nodiscard]] FORCE_INLINE Vec4 normalized() const{
		simd::Register len_sq = simd::dot4_splat(reg,reg);
		simd::Register inv_len = simd::rsqrt_accurate(len_sq);
		return Vec4(simd::mul(reg, inv_len));
	}

	[[nodiscard]] FORCE_INLINE Vec4 normalized_fast() const{
		simd::Register len_sq = simd::dot4_splat(reg,reg);
		simd::Register inv_len = simd::rsqrt(len_sq);
		return Vec4(simd::mul(reg, inv_len));
	}

	[[nodiscard]] FORCE_INLINE Vec4 abs() const{
		return Vec4{simd::abs(reg)};
	}

	FORCE_INLINE bool operator==(const Vec4& other) const{
		return simd::equals_all(reg, other.reg);
	}

	FORCE_INLINE bool operator!=(const Vec4& other) const{
		return !simd::equals_all(reg,other.reg);
	}

	[[nodiscard]] FORCE_INLINE bool is_close(const Vec4& other, float epsilon = 1e-5f) const{
		return simd::is_close_all(reg, other.reg, epsilon);
	}

	template<int I>
	[[nodiscard]] FORCE_INLINE Vec4 splat() const{
		return Vec4(simd::splat<I>(this->reg));
	}

	[[nodiscard]] FORCE_INLINE static Vec4 fmadd(
			const Vec4& a,
			const Vec4& b,
			const Vec4& c){
		// returns (a * b) + c
		return Vec4(simd::fmadd(a.reg, b.reg, c.reg));
	}
};

static_assert(sizeof(Vec4) == 16, "Vec4 size must be exactly 16 byes");
static_assert(alignof(Vec4) == 16, "Vec4 alignment must be 16 byes");
static_assert(offsetof(Vec4,Vec4::x) == 0, "Vec4::x must be at offset 0");
static_assert(offsetof(Vec4,Vec4::reg) == 0, "Vec4::reg must be at offset 0");
static_assert(offsetof(Vec4,Vec4::y) == sizeof(float), "Vec4: Gap between x and y");
static_assert(offsetof(Vec4,Vec4::z) == 2*sizeof(float), "Vec4: Gap between y and z");
static_assert(offsetof(Vec4,Vec4::w) == 3*sizeof(float), "Vec4: Gap between z and w");

inline std::ostream& operator<<(std::ostream& os, const Vec4& v){
	os << "Vec4(" << v.x << ", " << v.y << ", " << v.z << ", " << v.w << ")";
	return os;
}

[[nodiscard]] FORCE_INLINE Vec4 operator*(float s, const Vec4& v){
	return v * s;
}

} // namespace engine::math
