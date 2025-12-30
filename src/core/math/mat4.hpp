#pragma once

#include<iostream>
#include<cmath>
#include<vector>

#include"vec4.hpp"

namespace engine::math{

struct Mat4{
	Vec4 cols[4];

	Mat4(){
		cols[0] = Vec4{1.0f, 0.0f, 0.0f, 0.0f};
		cols[1] = Vec4{0.0f, 1.0f, 0.0f, 0.0f};
		cols[2] = Vec4{0.0f, 0.0f, 1.0f, 0.0f};
		cols[3] = Vec4{0.0f, 0.0f, 0.0f, 1.0f};
	}

	Mat4(
			const Vec4& col0,
			const Vec4& col1,
			const Vec4& col2,
			const Vec4& col3){
		cols[0] = col0;
		cols[1] = col1;
		cols[2] = col2;
		cols[3] = col3;
	}

	static Mat4 identity(){
		return Mat4();
	}

	Mat4(const std::vector<float>& vec){
		for(int i = 0; i < 4; ++i){
			cols[i] = Vec4(
				vec[i*4 + 0],
				vec[i*4 + 1],
				vec[i*4 + 2],
				vec[i*4 + 3]
			);
		}
	}

	Mat4(const float* vec){
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

	Vec4 operator*(const Vec4& v) const {return mul(*this,v);}
	Mat4 operator*(const Mat4& m) const {return matmul(*this,m);}

	void transpose_(){
		simd::transpose(cols[0].reg, cols[1].reg, cols[2].reg, cols[3].reg);
	}

	[[nodiscard]] FORCE_INLINE Mat4 transpose() const {
		Mat4 result = *this;
		result.transpose_();
		return result;
	}

	void inverse_transform_no_scale_(){
		simd::inverse_transform_no_scale(
			cols[0].reg,
			cols[1].reg,
			cols[2].reg,
			cols[3].reg
		);
	}

	[[nodiscard]] FORCE_INLINE Mat4 inverse_transform_no_scale() const{
		Mat4 result = *this;
		result.inverse_transform_no_scale_();
		return result;
	}

	void inverse_transform_(){
		simd::inverse_transform(
			cols[0].reg,
			cols[1].reg,
			cols[2].reg,
			cols[3].reg
		);
	}

	[[nodiscard]] FORCE_INLINE Mat4 inverse_transform() const{
		Mat4 result = *this;
		result.inverse_transform_();
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
};

} // namespace engine::math
