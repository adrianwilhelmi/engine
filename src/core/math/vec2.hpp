#pragma once

#include<iostream>
#include<iomanip>
#include<cassert>
#include<cmath>

#include"simd_backend.hpp"
#include"vec3packed.hpp"

namespace engine::math{

struct alignas(8) Vec2{
	float x;
	float y;

	FORCE_INLINE Vec2(){
		x = 0;
		y = 0;
	}

	FORCE_INLINE explicit Vec2(const float val){
		x = val;
		y = val;
	}

	FORCE_INLINE Vec2(float _x, float _y){
		x = _x;
		y = _y;
	}

	[[nodiscard]] FORCE_INLINE float get_x() const{
		return x;
	}

	[[nodiscard]] FORCE_INLINE float get_y() const{
		return y;
	}

	FORCE_INLINE void set_x(const float val){
		x = val;
	}

	FORCE_INLINE void set_y(const float val){
		y = val;
	}

	[[nodiscard]] FORCE_INLINE Vec2 operator+(const Vec2& other) const{
		return Vec2(x+other.x, y+other.y);
	}

	[[nodiscard]] FORCE_INLINE Vec2 operator-(const Vec2& other) const{
		return Vec2(x-other.x, y-other.y);
	}

	[[nodiscard]] FORCE_INLINE Vec2 operator-() const{
		return Vec2(-x, -y);
	}

	[[nodiscard]] FORCE_INLINE Vec2 operator*(const Vec2& other) const{
		return Vec2(x*other.x, y*other.y);
	}

	[[nodiscard]] FORCE_INLINE Vec2 operator*(const float scalar) const{
		return Vec2(x*scalar, y*scalar);
	}

	[[nodiscard]] FORCE_INLINE Vec2 operator/(const Vec2& other) const{
		return Vec2(x/other.x, y/other.y);
	}

	[[nodiscard]] FORCE_INLINE Vec2 operator/(const float scalar) const{
		return Vec2(x/scalar, y/scalar);
	}

	FORCE_INLINE Vec2& operator+=(const Vec2& other){
		x += other.x;
		y += other.y;
		return *this;
	}

	FORCE_INLINE Vec2& operator-=(const Vec2& other){
		x -= other.x;
		y -= other.y;
		return *this;
	}

	FORCE_INLINE Vec2& operator*=(const Vec2& other){
		x *= other.x;
		y *= other.y;
		return *this;
	}

	FORCE_INLINE Vec2& operator*=(const float val){
		x *= val;
		y *= val;
		return *this;
	}

	FORCE_INLINE Vec2& operator/=(const float val){
		x /= val;
		y /= val;
		return *this;
	}


	FORCE_INLINE float operator[](int i) const {
		assert(i < 2 && "index oob for Vec2");
		if(i == 0) return x;
		return y;
	}
	FORCE_INLINE float& operator[](int i) {
		assert(i < 2 && "index oob for Vec2");
		if(i == 0) return x;
		return y;
	}

	FORCE_INLINE bool operator==(const Vec2& other) const{
		return (x==other.x && y==other.y);
	}

	FORCE_INLINE bool operator!=(const Vec2& other) const{
		return !(x==other.x && y==other.y);
	}

	[[nodiscard]] FORCE_INLINE Vec2 abs() const{
		return Vec2{std::abs(x), std::abs(y)};
	}

	[[nodiscard]] FORCE_INLINE bool is_close(const Vec2& other, float epsilon = 1e-5f) const{
		return ((std::abs(x-other.x) <= epsilon) && (std::abs(y-other.y) <= epsilon));
	}

	[[nodiscard]] FORCE_INLINE float dot(const Vec2& other) const{
		return x * other.x + y * other.y;
	}

	[[nodiscard]] FORCE_INLINE float cross(const Vec2& other) const{
		return x * other.y - y * other.x;
	}

	[[nodiscard]] FORCE_INLINE float length_sq() const { return dot(*this); }
	[[nodiscard]] FORCE_INLINE float l2() const { return std::sqrt(length_sq()); }

	[[nodiscard]] FORCE_INLINE Vec2 normalized() const {
        const float len = l2();
        return len > 0.0f ? (*this / len) : Vec2(0.0f);
    }

	[[nodiscard]] static FORCE_INLINE Vec2 lerp(const Vec2& a, const Vec2&b, float t){
		return a + (b - a) * t;
	}
};

static_assert(sizeof(Vec2) == 8, "Vec2 size must be exactly 8 bytes");
static_assert(alignof(Vec2) == 8, "Vec2 alignment must be 8 byes");

inline std::ostream& operator<<(std::ostream& os, const Vec2& v){
	os << "Vec2(\n\t" << v.x << ",\n\t" << v.y << "\n)\n";
	return os;
}

[[nodiscard]] FORCE_INLINE Vec2 operator*(float s, const Vec2& v){
	return v * s;
}

} // namespace engine::math
