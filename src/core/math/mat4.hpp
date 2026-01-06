#pragma once

#include<iostream>
#include<cmath>
#include<vector>
#include<iomanip>

#include"vec4.hpp"
#include"vec3.hpp"

namespace engine::math{

struct alignas(16) Mat4{
	// column-major
	Vec4 cols[4];

	FORCE_INLINE Mat4(){
		cols[0] = Vec4{1.0f, 0.0f, 0.0f, 0.0f};
		cols[1] = Vec4{0.0f, 1.0f, 0.0f, 0.0f};
		cols[2] = Vec4{0.0f, 0.0f, 1.0f, 0.0f};
		cols[3] = Vec4{0.0f, 0.0f, 0.0f, 1.0f};
	}

	FORCE_INLINE Mat4(
			const Vec4& col0,
			const Vec4& col1,
			const Vec4& col2,
			const Vec4& col3){
		cols[0] = col0;
		cols[1] = col1;
		cols[2] = col2;
		cols[3] = col3;
	}

	FORCE_INLINE Mat4(
			const simd::Register r0,
			const simd::Register r1,
			const simd::Register r2,
			const simd::Register r3){
		cols[0] = Vec4(r0);
		cols[1] = Vec4(r1);
		cols[2] = Vec4(r2);
		cols[3] = Vec4(r3);
	}


	[[nodiscard]] FORCE_INLINE static Mat4 identity(){
		return Mat4();
	}

	FORCE_INLINE Mat4(const float* vec){
		for(int i = 0; i < 4; ++i){
			cols[i] = Vec4{
				vec[i*4 + 0],
				vec[i*4 + 1],
				vec[i*4 + 2],
				vec[i*4 + 3]
			};
		}
	}

	[[nodiscard]] FORCE_INLINE static Vec4 mul(const Mat4& m, const Vec4& v){
		Vec4 res = m.cols[0] * v.splat<0>();
		res = Vec4::fmadd(m.cols[1], v.splat<1>(), res);
		res = Vec4::fmadd(m.cols[2], v.splat<2>(), res);
		res = Vec4::fmadd(m.cols[3], v.splat<3>(), res);
		return res;
	}

	[[nodiscard]] FORCE_INLINE static Mat4 matmul(const Mat4& a, const Mat4& b){
		Mat4 res;

		simd::Register a0 = a.cols[0].reg;
		simd::Register a1 = a.cols[1].reg;
		simd::Register a2 = a.cols[2].reg;
		simd::Register a3 = a.cols[3].reg;

		for(int i = 0; i < 4; ++i){
			simd::Register b_col = b.cols[i].reg;

			simd::Register r = simd::mul(a0, simd::splat<0>(b_col));
			r = simd::fmadd(a1, simd::splat<1>(b_col), r);
			r = simd::fmadd(a2, simd::splat<2>(b_col), r);
			r = simd::fmadd(a3, simd::splat<3>(b_col), r);

			res.cols[i].reg = r;
		}
		return res;
	}

	[[nodiscard]] FORCE_INLINE Vec4 operator*(const Vec4& v) const {
		return mul(*this,v);
	}

	[[nodiscard]] FORCE_INLINE Mat4 operator*(const float val) const{
		Mat4 res = *this;
		for(int i = 0; i < 4; ++i){
			res.cols[i].reg = simd::mul(res.cols[i].reg, val);
		}
		return res;
	}

	FORCE_INLINE Mat4& operator*=(const float val){
		for(int i = 0; i < 4; ++i){
			cols[i].reg = simd::mul(cols[i].reg, val);
		}
		return *this;
	}

	[[nodiscard]] FORCE_INLINE Mat4 operator*(const Mat4& m) const {
		return matmul(*this,m);
	}

	[[nodiscard]] FORCE_INLINE Mat4 operator+(const Mat4& other) const{
		Mat4 res;
		for(int i = 0; i < 4; ++i){
			res.cols[i] = cols[i] + other.cols[i];
		}
		return res;
	}

	FORCE_INLINE const Vec4& operator[](int i) const {return cols[i]; }
	FORCE_INLINE Vec4& operator[](int i) {return cols[i]; }

	FORCE_INLINE void transpose_(){
		simd::transpose(cols[0].reg, cols[1].reg, cols[2].reg, cols[3].reg);
	}

	[[nodiscard]] FORCE_INLINE Mat4 transpose() const {
		Mat4 result = *this;
		result.transpose_();
		return result;
	}

	FORCE_INLINE void inverse_transform_no_scale_(){
		// works only if this matrix is transformation matrix with scale 1
		simd::inverse_transform_no_scale(
			cols[0].reg,
			cols[1].reg,
			cols[2].reg,
			cols[3].reg
		);
	}

	[[nodiscard]] FORCE_INLINE Mat4 inverse_transform_no_scale() const{
		// works only if this matrix is transformation matrix with scale 1
		Mat4 result = *this;
		result.inverse_transform_no_scale_();
		return result;
	}

	FORCE_INLINE void inverse_transform_(){
		// works only if this matrix is transformation matrix
		simd::inverse_transform(
			cols[0].reg,
			cols[1].reg,
			cols[2].reg,
			cols[3].reg
		);
	}

	[[nodiscard]] FORCE_INLINE Mat4 inverse_transform() const{
		// works only if this matrix is transformation matrix
		Mat4 result = *this;
		result.inverse_transform_();
		return result;
	}

	FORCE_INLINE void inverse_(){
		// works for any matrix
		simd::inverse(
			cols[0].reg,
			cols[1].reg,
			cols[2].reg,
			cols[3].reg
		);
	}

	[[nodiscard]] FORCE_INLINE Mat4 inverse() const{
		// works for any matrix
		Mat4 result = *this;
		result.inverse_();
		return result;
	}

	[[nodiscard]] FORCE_INLINE static Mat4 perspective(
			const float fov_radians,
			const float aspect,
			const float znear,
			const float zfar){
		const float h = 1.0f / std::tan(fov_radians * 0.5f);
		const float w = h / aspect;
		const float a = zfar / (znear - zfar);
		const float b = (znear * zfar) / (znear - zfar);

		return Mat4(
			Vec4(w,		0.0f,	0.0f,	0.0f),
			Vec4(0.0f,	h,		0.0f,	0.0f),
			Vec4(0.0f,	0.0f,	a,		-1.0f),
			Vec4(0.0f,	0.0f,	b,		0.0f)
		);
	}

	[[nodiscard]] FORCE_INLINE const float* data() const{
		return reinterpret_cast<const float*>(&cols[0]);
	}

	[[nodiscard]] static Mat4 look_at(
			const Vec3& eye,
			const Vec3& center,
			const Vec3& up){
		Vec3 f = (center - eye).normalized();
		Vec3 s = f.cross(up).normalized();
		Vec3 u = s.cross(f);

		Mat4 res;

		res.cols[0] = Vec4(s.get_x(), u.get_x(), -f.get_x(), 0.0f);
		res.cols[1] = Vec4(s.get_y(), u.get_y(), -f.get_y(), 0.0f);
		res.cols[2] = Vec4(s.get_z(), u.get_z(), -f.get_z(), 0.0f);

		Vec4 eye4(eye);
		Vec4 translation = mul(res, eye4);

		res.cols[3] = translation * Vec4(-1.0f, -1.0f, -1.0f, 0.0f);
		res.cols[3].set_w(1.0f);

		return res;
	}

	[[nodiscard]] FORCE_INLINE static Mat4 translate(const Vec3& v){
		Mat4 res = Mat4::identity();
		res.cols[3] = Vec4(v.get_x(), v.get_y(), v.get_z(), 1.0f);
		return res;
	}

	[[nodiscard]] FORCE_INLINE static Mat4 scale(const Vec3& v){
		return Mat4(
			Vec4(v.get_x(), 0.0f, 0.0f, 0.0f),
			Vec4(0.0f, v.get_y(), 0.0f, 0.0f),
			Vec4(0.0f, 0.0f, v.get_z() , 0.0f),
			Vec4(0.0f, 0.0f, 0.0f, 1.0f)
		);
	}

	[[nodiscard]] FORCE_INLINE static Mat4 ortho(
			const float left, const float right,
			const float bottom, const float top,
			const float znear, const float zfar){
		const float r_l = 1.0f / (right - left);
		const float t_b = 1.0f / (top - bottom);
		const float f_n = 1.0f / (zfar - znear);

		Mat4 res;	// identity
		res.cols[0].set_x(2.0f * r_l);
		res.cols[1].set_y(2.0f * t_b);
		res.cols[2].set_z(-f_n);

		res.cols[3] = Vec4(
			-(right + left) * r_l,
			-(top + bottom) * t_b,
			-znear * f_n,
			1.0f
		);

		return res;
	}

	[[nodiscard]] FORCE_INLINE static Mat4 rotate_x(float rad){
		float s = std::sin(rad);
		float c = std::cos(rad);
		return Mat4(
			Vec4(1.0f,	0.0f,	0.0f,	0.0f),
			Vec4(0.0f,	c,		s,		0.0f),
			Vec4(0.0f,	-s,		c,		0.0f),
			Vec4(0.0f,	0.0f,	0.0f,	1.0f)
		);
	}

	[[nodiscard]] FORCE_INLINE static Mat4 rotate_y(float rad){
		float s = std::sin(rad);
		float c = std::cos(rad);
		return Mat4(
			Vec4(c,		0.0f,	-s,		0.0f),
			Vec4(0.0f,	1.0f,	0.0f,	0.0f),
			Vec4(s,		0.0f,	c,		0.0f),
			Vec4(0.0f,	0.0f,	0.0f,	1.0f)
		);
	}

	[[nodiscard]] FORCE_INLINE static Mat4 rotate_z(float rad){
		float s = std::sin(rad);
		float c = std::cos(rad);
		return Mat4(
			Vec4(c,		s,		0.0f,	0.0f),
			Vec4(-s,	c,		0.0f,	0.0f),
			Vec4(0.0f,	0.0f,	1.0f,	0.0f),
			Vec4(0.0f,	0.0f,	0.0f,	1.0f)
		);
	}
};

inline std::ostream& operator <<(std::ostream& os, const Mat4& m){
	os << std::fixed << std::setprecision(3);
	os << "| ";
	for(int j = 0; j < 4; ++j){
		os << m.cols[j].get_x() << " ";
		os << m.cols[j].get_y() << " ";
		os << m.cols[j].get_z() << " ";
		os << m.cols[j].get_w();
	}
	os << " |\n";
	return os;
}

} // namespace engine::math
