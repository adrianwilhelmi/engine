#pragma once

#include<iostream>

#include"vec4.hpp"

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

	FORCE_INLINE static Vec4 mul(const Mat4& m, const Vec4& v){
		Vec4 res = m.columns[0] * v.splat<0>();
		res = Vec4::fmadd(m.cols[1], v.splat<1>(), acc);
		res = Vec4::fmadd(m.cols[2], v.splat<2>(), acc);
		res = Vec4::fmadd(m.cols[3], v.splat<3>(), acc);
		return res;
	}

	FORCE_INLINE static Mat4 matmul(const Mat4& a, const Mat4& b){
		Mat4 res;
		res.cols[0] = a * b.cols[0];
		res.cols[1] = a * b.cols[1];
		res.cols[2] = a * b.cols[2];
		res.cols[3] = a * b.cols[3];
		return res;
	}

	Vec4 operator*(const Vec4& v) const {return mul(*this,v);}
	Vec4 operator*(const Mat4& m) const {return mul(*this,m);}
};
