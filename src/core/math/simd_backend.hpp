#pragma once

#include<cmath>
#include<utility>
#include<string>
#include<algorithm>

#if defined(FORCE_NO_SIMD)
	#define ENGINE_SIMD_NONE
#elif defined(__AVX2__)
	// use SSE anyways, __m256 not needed
	#define ENGINE_SIMD_SSE
	#define ENGINE_SIMD_AVX
	#include<immintrin.h>

	#if defined(__FMA__)
		#define ENGINE_SIMD_FMA
	#endif

#elif defined(__AVX__) || defined(__SSE4_1__) || defined(_M_AMD64) || defined(_M_X64)
	#define ENGINE_SIMD_SSE
	#include<immintrin.h>

#elif defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(__aarch64__)
	#define ENGINE_SIMD_NEON
	#include<arm_neon.h>
#else
	#define ENGINE_SIMD_NONE
	//fallback
#endif

namespace engine::math::simd{

#ifdef ENGINE_SIMD_SSE
	using Register = __m128;
#elif ENGINE_SIMD_NEON
	using Register = float32x4_t;
#else
	struct Register {float f[4]; };
#endif


#if defined(_MSC_VER)
	#define FORCE_INLINE __forceinline
#else
	#define FORCE_INLINE __attribute__((always_inline)) inline
#endif

[[nodiscard]] FORCE_INLINE std::string detected_arch(){
	#ifdef ENGINE_SIMD_AVX
		return "avx2";
	#elif ENGINE_SIMD_SSE
		return "sse";
	#elif ENGINE_SIMD_NEON
		return "neon";
	#else
		return "other";
	#endif
}

//constructors
[[nodiscard]] FORCE_INLINE Register set(float x, float y, float z, float w = 0.0f){
	#ifdef ENGINE_SIMD_SSE
		return _mm_set_ps(w,z,y,x);
	#elif ENGINE_SIMD_NEON
		float data[4] = {x,y,z,w};
		return vld1q_f32(data);
	#else
		return {x,y,z,w};
	#endif
}

[[nodiscard]] FORCE_INLINE Register load_zero_w(const float* ptr_to_3_floats){
	#ifdef ENGINE_SIMD_SSE
		__m128 v = _mm_set_ps(
			0.0f, 
			ptr_to_3_floats[2], 
			ptr_to_3_floats[1], 
			ptr_to_3_floats[0]
		);
		return v;
	#elif ENGINE_SIMD_NEON
		float data[4] = {
			ptr_to_3_floats[0],
			ptr_to_3_floats[1],
			ptr_to_3_floats[2],
			0.0f
		};
		return vld1q_f32(data);
	#else
		return {ptr_to_3_floats[0],ptr_to_3_floats[1],ptr_to_3_floats[2], 0.0f};
	#endif
}

[[nodiscard]] FORCE_INLINE Register set1(float x){
	#ifdef ENGINE_SIMD_SSE
		return _mm_set1_ps(x);
	#elif ENGINE_SIMD_NEON
		float data[4] = {x,x,x,x};
		return vdupq_n_f32(x);
	#else
		return {x,x,x,x};
	#endif
}

//getters setters
[[nodiscard]] FORCE_INLINE float x(Register a) {
	#ifdef ENGINE_SIMD_SSE
		return _mm_cvtss_f32(a);
	#elif ENGINE_SIMD_NEON
		return vgetq_lane_f32(a, 0);
	#else
		return a.f[0];
	#endif
}

[[nodiscard]] FORCE_INLINE float y(Register a) {
	#ifdef ENGINE_SIMD_SSE
		return _mm_cvtss_f32(
			_mm_shuffle_ps(a,a, _MM_SHUFFLE(1,1,1,1))
		);
	#elif ENGINE_SIMD_NEON
		return vgetq_lane_f32(a, 1);
	#else
		return a.f[1];
	#endif
}

[[nodiscard]] FORCE_INLINE float z(Register a) {
	#ifdef ENGINE_SIMD_SSE
		return _mm_cvtss_f32(
			_mm_shuffle_ps(a,a, _MM_SHUFFLE(2,2,2,2))
		);
	#elif ENGINE_SIMD_NEON
		return vgetq_lane_f32(a, 2);
	#else
		return a.f[2];
	#endif
}

[[nodiscard]] FORCE_INLINE float w(Register a) {
	#ifdef ENGINE_SIMD_SSE
		return _mm_cvtss_f32(
			_mm_shuffle_ps(a,a, _MM_SHUFFLE(3,3,3,3))
		);
	#elif ENGINE_SIMD_NEON
		return vgetq_lane_f32(a, 3);
	#else
		return a.f[3];
	#endif
}

[[nodiscard]] FORCE_INLINE Register set_x(Register a, float val) {
	#ifdef ENGINE_SIMD_SSE
		// 0x00 -> take val, put it in lane 0, leave the rest from a
		return _mm_insert_ps(a, _mm_set_ss(val), 0x00 << 4);
	#elif ENGINE_SIMD_NEON
		return vsetq_lane_f32(val, a, 0);
	#else
		a.f[0] = val;
		return a;
	#endif
}

[[nodiscard]] FORCE_INLINE Register set_y(Register a, float val) {
	#ifdef ENGINE_SIMD_SSE
		// 0x10 -> take val, put it in lane 1, leave the rest from a
		return _mm_insert_ps(a, _mm_set_ss(val), 0x01 << 4);
	#elif ENGINE_SIMD_NEON
		return vsetq_lane_f32(val, a, 1);
	#else
		a.f[1] = val;
		return a;
	#endif
}

[[nodiscard]] FORCE_INLINE Register set_z(Register a, float val) {
	#ifdef ENGINE_SIMD_SSE
		return _mm_insert_ps(a, _mm_set_ss(val), 0x02 << 4);
	#elif ENGINE_SIMD_NEON
		return vsetq_lane_f32(val, a, 2);
	#else
		a.f[2] = val;
		return a;
	#endif
}

[[nodiscard]] FORCE_INLINE Register set_w(Register a, float val) {
	#ifdef ENGINE_SIMD_SSE
		return _mm_insert_ps(a, _mm_set_ss(val), 0x03 << 4);
	#elif ENGINE_SIMD_NEON
		return vsetq_lane_f32(val, a, 3);
	#else
		a.f[3] = val;
		return a;
	#endif
}

// arithmetic
[[nodiscard]] FORCE_INLINE Register add(Register a, Register b){
	#ifdef ENGINE_SIMD_SSE
		return _mm_add_ps(a,b);
	#elif ENGINE_SIMD_NEON
		return vaddq_f32(a,b);
	#else
		return {a.f[0]+b.f[0], a.f[1]+b.f[1], a.f[2]+b.f[2], a.f[3]+b.f[3]};
	#endif
}


[[nodiscard]] FORCE_INLINE Register sub(Register a, Register b){
	#ifdef ENGINE_SIMD_SSE
		return _mm_sub_ps(a,b);
	#elif ENGINE_SIMD_NEON
		return vsubq_f32(a,b);
	#else
		return {a.f[0]-b.f[0], a.f[1]-b.f[1], a.f[2]-b.f[2], a.f[3]-b.f[3]};
	#endif
}

[[nodiscard]] FORCE_INLINE Register mul(Register a, Register b){
	#ifdef ENGINE_SIMD_SSE
		return _mm_mul_ps(a,b);
	#elif ENGINE_SIMD_NEON
		return vmulq_f32(a,b);
	#else
		return {a.f[0]*b.f[0], a.f[1]*b.f[1], a.f[2]*b.f[2], a.f[3]*b.f[3]};
	#endif
}

[[nodiscard]] FORCE_INLINE Register div(Register a, Register b){
	#ifdef ENGINE_SIMD_SSE
		return _mm_div_ps(a,b);
	#elif ENGINE_SIMD_NEON
		return vdivq_f32(a,b);
	#else
		return {a.f[0]/b.f[0], a.f[1]/b.f[1], a.f[2]/b.f[2], a.f[3]/b.f[3]};
	#endif
}

[[nodiscard]] FORCE_INLINE Register mul(Register a, float s){
	#ifdef ENGINE_SIMD_SSE
		return _mm_mul_ps(a, _mm_set1_ps(s));
	#elif ENGINE_SIMD_NEON
		return vmulq_n_f32(a,s);
	#else
		return {a.f[0] * s, a.f[1]*s, a.f[2]*s, a.f[3]*s};
	#endif
}

[[nodiscard]] FORCE_INLINE Register neg(Register a){
	#ifdef ENGINE_SIMD_SSE
		return _mm_xor_ps(a, _mm_set1_ps(-0.0f));
	#elif ENGINE_SIMD_NEON
		return vnegq_f32(a);
	#else
		return {-a.f[0], -a.f[1], -a.f[2], -a.f[3]};
	#endif

}

[[nodiscard]] FORCE_INLINE Register fmadd(Register a, Register b, Register c){
	#if defined(ENGINE_SIMD_FMA) && defined(ENGINE_SIMD_SSE)
		return _mm_fmadd_ps(a,b,c);
	#elif defined(ENGINE_SIMD_NEON) && defined(__aarch64__)
		return vfmaq_f32(c,a,b);
	#else
		return add(mul(a,b),c);
	#endif
}

[[nodiscard]] FORCE_INLINE Register min(Register a, Register b){
	#ifdef ENGINE_SIMD_SSE
		return _mm_min_ps(a,b);
	#elif ENGINE_SIMD_NEON
		return vminq_f32(a,b);
	#else
		return {
			std::min(a.f[0], b.f[0]),
			std::min(a.f[1], b.f[1]),
			std::min(a.f[2], b.f[2]),
			std::min(a.f[3], b.f[3])
		};
	#endif
}

[[nodiscard]] FORCE_INLINE Register max(Register a, Register b){
	#ifdef ENGINE_SIMD_SSE
		return _mm_max_ps(a,b);
	#elif ENGINE_SIMD_NEON
		return vmaxq_f32(a,b);
	#else
		return {
			std::max(a.f[0], b.f[0]),
			std::max(a.f[1], b.f[1]),
			std::max(a.f[2], b.f[2]),
			std::max(a.f[3], b.f[3])
		};
	#endif
}

[[nodiscard]] FORCE_INLINE Register rsqrt(Register a){
	#ifdef ENGINE_SIMD_SSE
		return _mm_rsqrt_ps(a);
	#elif ENGINE_SIMD_NEON
		return vrsqrteq_f32(a);
	#else
		return {
			1.0f / std::sqrt(a.f[0]),
			1.0f / std::sqrt(a.f[1]),
			1.0f / std::sqrt(a.f[2]),
			1.0f / std::sqrt(a.f[3])
		};
	#endif
}


[[nodiscard]] FORCE_INLINE Register rsqrt_accurate(Register a){
	const Register half_neg = set(-0.5f, -0.5f, -0.5f, -0.5f);
	const Register three_halfs = set(1.5f, 1.5f, 1.5f, 1.5f);

	Register nr = rsqrt(a);
	Register muls = mul(mul(a,nr),nr);

	return mul(nr, fmadd(muls, half_neg, three_halfs));
}

[[nodiscard]] FORCE_INLINE float dot3(Register a, Register b){
	#ifdef ENGINE_SIMD_SSE
		return _mm_cvtss_f32(_mm_dp_ps(a,b,0x71));
	#elif ENGINE_SIMD_NEON
		float32x4_t mul_res = vmulq_f32(a,b);
		mul_res = vsetq_lane_f32(0.0f,mul_res,3);
		return vaddvq_f32(mul_res);
	#else
		return a.f[0]*b.f[0] + a.f[1]*b.f[1] + a.f[2] * b.f[2];
	#endif
}

[[nodiscard]] FORCE_INLINE Register dot3_splat(Register a, Register b){
	//returns [dot, dot, dot, _]
	#ifdef ENGINE_SIMD_SSE
		return _mm_dp_ps(a,b,0x7F);
	#elif ENGINE_SIMD_NEON
		float d = dot3(a,b);
		return vdupq_n_f32(d);
	#else
		float d = a.f[0]*b.f[0] + a.f[1]*b.f[1] + a.f[2] * b.f[2];
		return {d,d,d,d};
	#endif
}

[[nodiscard]] FORCE_INLINE float dot4(Register a, Register b){
	#ifdef ENGINE_SIMD_SSE
		// _mm_dp_ps needs SSE4.1
		return _mm_cvtss_f32(_mm_dp_ps(a,b,0xFF));
	#elif ENGINE_SIMD_NEON
		float32x4_t mul_res = vmulq_f32(a,b);
		return vaddvq_f32(mul_res);
	#else
		return a.f[0]*b.f[0] + a.f[1]*b.f[1] + a.f[2] * b.f[2] + a.f[3]*b.f[3];
	#endif
}

[[nodiscard]] FORCE_INLINE Register dot4_splat(Register a, Register b){
	//returns [dot, dot, dot, dot]
	#ifdef ENGINE_SIMD_SSE
		return _mm_dp_ps(a,b,0xFF);
	#elif ENGINE_SIMD_NEON
		float d = dot4(a,b);
		return vdupq_n_f32(d);
	#else
		float d =  a.f[0]*b.f[0] + a.f[1]*b.f[1] + a.f[2] * b.f[2] + a.f[3]*b.f[3];
		return {d,d,d,d};
	#endif
}

[[nodiscard]] FORCE_INLINE Register cross3(Register a, Register b){
	#ifdef ENGINE_SIMD_SSE
		__m128 a_yzx = _mm_shuffle_ps(a,a,_MM_SHUFFLE(3,0,2,1));
		__m128 b_yzx = _mm_shuffle_ps(b,b,_MM_SHUFFLE(3,0,2,1));

		__m128 a_zxy = _mm_shuffle_ps(a,a,_MM_SHUFFLE(3,1,0,2));
		__m128 b_zxy = _mm_shuffle_ps(b,b,_MM_SHUFFLE(3,1,0,2));

		__m128 term1 = _mm_mul_ps(a_yzx, b_zxy);
		__m128 term2 = _mm_mul_ps(a_zxy, b_yzx);

		return _mm_sub_ps(term1, term2);
	
	#elif ENGINE_SIMD_NEON
		#if defined(__clang__) || defined(__GNUC__)
			float32x4_t a_yzx = __builtin_shufflevector(a,a,1,2,0,3);
			float32x4_t b_yzx = __builtin_shufflevector(b,b,1,2,0,3);

			float32x4_t a_zxy = __builtin_shufflevector(a,a,2,0,1,3);
			float32x4_t b_zxy = __builtin_shufflevector(b,b,2,0,1,3);

			return vsubq_f32(vmulq_f32(a_yzx, b_zxy), vmulq_f32(a_zxy,b_yzx));
		#else
			float res[4];
			float A[4],B[4];
			vst1q_f32(A,a);
			vst1q_f32(B,b);

			res[0] = A[1]*B[2] - A[2]*B[1];
			res[1] = A[2]*B[0] - A[0]*B[2];
			res[2] = A[0]*B[1] - A[1]*B[0];
			res[3] = 0.0f;

			return vld1q_f32(res);
		#endif

	#else
		return{
			a.f[1]*b.f[2] - a.f[2]*b.f[1],
			a.f[2]*b.f[0] - a.f[0]*b.f[2],
			a.f[0]*b.f[1] - a.f[1]*b.f[0],
			0.0f
		};
	#endif
}

[[nodiscard]] FORCE_INLINE bool equals_all(Register a, Register b){
	#ifdef ENGINE_SIMD_SSE
		__m128 cmp = _mm_cmpeq_ps(a,b);
		return _mm_movemask_ps(cmp) == 0xF;
	#elif ENGINE_SIMD_NEON
		uint32x4_t cmp = vceqq_f32(a,b);
		return vminvq_u32(cmp) == 0xFFFFFFFF;
	#else
		return a.f[0]==b.f[0] &&
			a.f[1] == b.f[1] &&
			a.f[2] == b.f[2] &&
			a.f[3] == b.f[3];
	#endif
}

[[nodiscard]] FORCE_INLINE bool equals_xyz(Register a, Register b){
	#ifdef ENGINE_SIMD_SSE
		__m128 cmp = _mm_cmpeq_ps(a,b);
		return (_mm_movemask_ps(cmp) & 0x7) == 0x7;

	#elif ENGINE_SIMD_NEON
		uint32x4_t cmp = vceqq_f32(a,b);
		uint32_t mask_data[4] = {0,0,0,0xFFFFFFFF};
		uint32x4_t w_ignore = vld1q_u32(mask_data);
		uint32x4_t cmp_xyz = vorrq_u32(cmp, w_ignore);
		return vminvq_u32(cmp_xyz) == 0xFFFFFFFF;

	#else
		return a.f[0]==b.f[0] &&
			a.f[1] == b.f[1] &&
			a.f[2] == b.f[2];
	#endif
}

[[nodiscard]] FORCE_INLINE Register abs(Register a){
	// zero the sign bit
	#ifdef ENGINE_SIMD_SSE
		const __m128 mask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF));
		return _mm_and_ps(a, mask);
	#elif ENGINE_SIMD_NEON
		return vabsq_f32(a);
	#else
		return {std::abs(a.f[0]),
			std::abs(a.f[1]),
			std::abs(a.f[2]),
			std::abs(a.f[3])
		};
	#endif
}

[[nodiscard]] FORCE_INLINE bool is_close_all(Register a, Register b, float eps){
	Register diff = abs(sub(a,b));
	#ifdef ENGINE_SIMD_SSE
		Register epsilon = set1(eps);
		__m128 cmp = _mm_cmplt_ps(diff, epsilon);
		return (_mm_movemask_ps(cmp) & 0xF) == 0xF;
	#elif ENGINE_SIMD_NEON
		Register epsilon = set1(eps);
		uint32x4_t cmp = vcltq_f32(diff, epsilon);
		return vminvq_u32(cmp) == 0xFFFFFFFF;
	#else
		return diff.f[0] < eps 
			&& diff.f[1] < eps 
			&& diff.f[2] < eps 
			&& diff.f[3] < eps;
	#endif
}

[[nodiscard]] FORCE_INLINE bool is_close_xyz(Register a, Register b, float eps){
	Register diff = abs(sub(a,b));
	#ifdef ENGINE_SIMD_SSE
		Register epsilon = set1(eps);
		__m128 cmp = _mm_cmplt_ps(diff, epsilon);
		return (_mm_movemask_ps(cmp) & 0x7) == 0x7;
	#elif ENGINE_SIMD_NEON
		Register epsilon = set1(eps);
		uint32x4_t cmp = vcltq_f32(diff, epsilon);
		cmp = vsetq_lane_u32(0xFFFFFFFF, cmp, 3);
		return vminvq_u32(cmp) == 0xFFFFFFFF;
	#else
		return diff.f[0] < eps 
			&& diff.f[1] < eps 
			&& diff.f[2] < eps;
	#endif
}

template<int Index>
[[nodiscard]] FORCE_INLINE Register splat(Register r){
	#ifdef ENGINE_SIMD_SSE
		return _mm_shuffle_ps(r,r,_MM_SHUFFLE(Index, Index, Index, Index));
	#elif ENGINE_SIMD_NEON
		return vdupq_n_f32(vgetq_lane_f32(r, Index));
	#else
		static_assert(Index >= 0 && Index < 4, "index oob");
		float val = r.f[Index];
		return {val,val,val,val};
	#endif
}

FORCE_INLINE void transpose(
		Register& c0, 
		Register& c1,
		Register& c2,
		Register& c3){
	#if defined(ENGINE_SIMD_SSE)
		__m128 tmp0 = _mm_unpacklo_ps(c0,c1);
		__m128 tmp1 = _mm_unpacklo_ps(c2,c3);
		__m128 tmp2 = _mm_unpackhi_ps(c0,c1);
		__m128 tmp3 = _mm_unpackhi_ps(c2,c3);
		c0 = _mm_movelh_ps(tmp0, tmp1);
		c1 = _mm_movehl_ps(tmp1, tmp0);
		c2 = _mm_movelh_ps(tmp2, tmp3);
		c3 = _mm_movehl_ps(tmp3, tmp2);

	#elif defined(ENGINE_SIMD_NEON)
		float32x4x2_t r01 = vzipq_f32(c0,c1);
		float32x4x2_t r23 = vzipq_f32(c2,c3);
		c0 = vcombine_f32(vget_low_f32(r01.val[0]), vget_low_f32(r23.val[0]));
		c1 = vcombine_f32(vget_low_f32(r01.val[1]), vget_low_f32(r23.val[1]));
		c2 = vcombine_f32(vget_high_f32(r01.val[0]), vget_high_f32(r23.val[0]));
		c3 = vcombine_f32(vget_high_f32(r01.val[1]), vget_high_f32(r23.val[1]));

	#else
		std::swap(c0.f[1], c1.f[0]);
		std::swap(c0.f[2], c2.f[0]);
		std::swap(c0.f[3], c3.f[0]);
		std::swap(c1.f[2], c2.f[1]);
		std::swap(c1.f[3], c3.f[1]);
		std::swap(c2.f[3], c3.f[2]);

	#endif
}


// the next 3 functions are inspired by: https://lxjk.github.io/2017/09/03/Fast-4x4-Matrix-Inverse-with-SSE-SIMD-Explained.html

FORCE_INLINE void inverse_transform_no_scale(
		Register& c0, 
		Register& c1,
		Register& c2,
		Register& c3){
	// requires input matrix to be transform matrix of scale 1

	Register c3c = c3;
	Register c2c = c2;

	#ifdef ENGINE_SIMD_SSE
		__m128 tmp0 = _mm_unpacklo_ps(c0,c1);
		__m128 tmp1 = _mm_unpackhi_ps(c0,c1);

		c0 = _mm_shuffle_ps(tmp0, c2c, _MM_SHUFFLE(3,0,1,0));
		c1 = _mm_shuffle_ps(tmp0, c2c, _MM_SHUFFLE(3,1,3,2));
		c2 = _mm_shuffle_ps(tmp1, c2c, _MM_SHUFFLE(3,2,1,0));

	#elif ENGINE_SIMD_NEON
		float32x4x2_t zip01 = vzipq_f32(c0,c1);

		c0 = vcombine_f32(
			vget_low_f32(zip01.val[0]), 
			vget_low_f32(vdup_n_f32(0.0f))
		);
		c0 = vsetq_lane_f32(vgetq_lane_f32(c2c, 0), c0, 2);

		c1 = vcombine_f32(
			vget_high_f32(zip01.val[0]), 
			vget_low_f32(vdup_n_f32(0.0f))
		);
		c1 = vsetq_lane_f32(vgetq_lane_f32(c2c, 1), c1, 2);

		c2 = vcombine_f32(
			vget_low_f32(zip01.val[1]), 
			vget_low_f32(vdup_n_f32(0.0f))
		);
		c2 = vsetq_lane_f32(vgetq_lane_f32(c2c, 2), c2, 2);

	#else
		float x0 = c0.f[0], y0 = c0.f[1], z0 = c0.f[2];
		float x1 = c1.f[0], y1 = c1.f[1], z1 = c1.f[2];
		float x2 = c2.f[0], y2 = c2.f[1], z2 = c2.f[2];
		c0 = {x0,x1,x2, 0.0f};
		c1 = {y0,y1,y2, 0.0f};
		c2 = {z0,z1,z2, 0.0f};
	#endif

	Register tx = splat<0>(c3c);
	Register ty = splat<1>(c3c);
	Register tz = splat<2>(c3c);

	Register t_part = mul(c0,tx);
	t_part = fmadd(c1,ty,t_part);
	t_part = fmadd(c2,tz,t_part);

	c3 = sub(set(0.0f,0.0f,0.0f,1.0f), t_part);
}

FORCE_INLINE void inverse_transform(
		Register& c0, 
		Register& c1,
		Register& c2,
		Register& c3){
	// requires input matrix to be transform matrix

	Register c3c = c3;
	Register one = set1(1.0f);
	Register eps = set1(1e-8f);

	Register c2c = c2;

	#ifdef ENGINE_SIMD_SSE
		__m128 tmp0 = _mm_unpacklo_ps(c0,c1);
		__m128 tmp1 = _mm_unpackhi_ps(c0,c1);

		c0 = _mm_shuffle_ps(tmp0, c2c, _MM_SHUFFLE(3,0,1,0));
		c1 = _mm_shuffle_ps(tmp0, c2c, _MM_SHUFFLE(3,1,3,2));
		c2 = _mm_shuffle_ps(tmp1, c2c, _MM_SHUFFLE(3,2,1,0));

	#elif ENGINE_SIMD_NEON
		float32x4x2_t zip01 = vzipq_f32(c0,c1);

		c0 = vcombine_f32(
			vget_low_f32(zip01.val[0]), 
			vget_low_f32(vdup_n_f32(0.0f))
		);
		c0 = vsetq_lane_f32(vgetq_lane_f32(c2c, 0), c0, 2);

		c1 = vcombine_f32(
			vget_high_f32(zip01.val[0]), 
			vget_low_f32(vdup_n_f32(0.0f))
		);
		c1 = vsetq_lane_f32(vgetq_lane_f32(c2c, 1), c1, 2);

		c2 = vcombine_f32(
			vget_low_f32(zip01.val[1]), 
			vget_low_f32(vdup_n_f32(0.0f))
		);
		c2 = vsetq_lane_f32(vgetq_lane_f32(c2c, 2), c2, 2);

	#else
		float x0 = c0.f[0], y0 = c0.f[1], z0 = c0.f[2];
		float x1 = c1.f[0], y1 = c1.f[1], z1 = c1.f[2];
		float x2 = c2.f[0], y2 = c2.f[1], z2 = c2.f[2];
		c0 = {x0,x1,x2, 0.0f};
		c1 = {y0,y1,y2, 0.0f};
		c2 = {z0,z1,z2, 0.0f};
	#endif

	Register size_sqr = mul(c0,c0);
	size_sqr = fmadd(c1,c1,size_sqr);
	size_sqr = fmadd(c2,c2,size_sqr);

	//Register safe_size_sqr = max(size_sqr, eps);
	//Register r_size_sqr = div(one, safe_size_sqr);

	#ifdef ENGINE_SIMD_SSE
		__m128 mask = _mm_cmplt_ps(size_sqr, eps);
		Register r_size_sqr = _mm_blendv_ps(_mm_div_ps(one, size_sqr), one, mask);
	#elif ENGINE_SIMD_NEON
		uint32x4_t mask = vcltq_f32(size_sqr, eps);
		Register r_size_sqr = vbslq_f32(mask, one, vdivq_f32(one, size_sqr));
	#else
		Register r_size_sqr;
		for(int i=0; i<3; ++i) 
			r_size_sqr.f[i] = (size_sqr.f[i] < 1e-8) ? 1.0f : 1.0f / size_sqr.f[i];
		r_size_sqr.f[3] = 1.0f;
	#endif

	c0 = mul(c0, r_size_sqr);
	c1 = mul(c1, r_size_sqr);
	c2 = mul(c2, r_size_sqr);

	Register tx = splat<0>(c3c);
	Register ty = splat<1>(c3c);
	Register tz = splat<2>(c3c);

	Register t_part = mul(c0, tx);
	t_part = fmadd(c1, ty, t_part);
	t_part = fmadd(c2, tz, t_part);

	c3 = sub(set(0.0f, 0.0f, 0.0f, 1.0f), t_part);
}

namespace util{

// 2x2 matmul A*B
FORCE_INLINE Register mat2_mul(Register vec1, Register vec2){
	#ifdef ENGINE_SIMD_SSE
		Register t1 = mul(vec1, _mm_shuffle_ps(vec2, vec2, _MM_SHUFFLE(3,3,0,0)));
		Register t2 = mul(
			_mm_shuffle_ps(vec1, vec1, _MM_SHUFFLE(1,0,3,2)),
			_mm_shuffle_ps(vec2, vec2, _MM_SHUFFLE(2,2,1,1))
		);
		return add(t1,t2);
	#elif ENGINE_SIMD_NEON
		Register t1 = mul(vec1, vdupq_lane_f32(vget_low_f32(vec2), 0));
		float32x4_t xxxx = vdupq_laneq_f32(vec2, 0);
		float32x4_t yyyy = vdupq_laneq_f32(vec2, 1);
		float32x4_t zzzz = vdupq_laneq_f32(vec2, 2);
		float32x4_t wwww = vdupq_laneq_f32(vec2, 3);
		float32x4_t acac = vcombine_f32(vget_low_f32(vec1), vget_low_f32(vec1));
		Register res_lo = vaddq_f32(vmulq_f32(vget_low_f32(vec1), vget_low_f32(xxxx)), vmulq_f32(vget_high_f32(vec1), vget_low_f32(yyyy)));
		Register res_hi = vaddq_f32(vmulq_f32(vget_low_f32(vec1), vget_low_f32(zzzz)), vmulq_f32(vget_high_f32(vec1), vget_low_f32(wwww)));
        return vcombine_f32(res_lo, res_hi);
	#else
		Register r;
		r.f[0] = vec1.f[0] * vec2.f[0] + vec1.f[2] * vec2.f[1];
		r.f[1] = vec1.f[1] * vec2.f[0] + vec1.f[3] * vec2.f[1];
		r.f[2] = vec1.f[0] * vec2.f[2] + vec1.f[2] * vec2.f[3];
		r.f[3] = vec1.f[1] * vec2.f[2] + vec1.f[3] * vec2.f[3];
		return r;
	#endif
}

// Adjugate(A) * B
FORCE_INLINE Register mat2_adj_mul(Register vec1, Register vec2){
	#ifdef ENGINE_SIMD_SSE
		return sub(
			mul(_mm_shuffle_ps(vec1, vec1, _MM_SHUFFLE(0,3,0,3)),vec2),
			mul(
				_mm_shuffle_ps(vec1, vec1, _MM_SHUFFLE(1,2,1,2)), 
				_mm_shuffle_ps(vec2, vec2, _MM_SHUFFLE(2,3,0,1))
		   )
		);
	#else
		Register r;
		r.f[0] = vec1.f[3] * vec2.f[0] - vec1.f[2] * vec2.f[1];
		r.f[1] = -vec1.f[1] * vec2.f[0] + vec1.f[0] * vec2.f[1];
		r.f[2] = vec1.f[3] * vec2.f[2] - vec1.f[2] * vec2.f[3];
		r.f[3] = -vec1.f[1] * vec2.f[2] + vec1.f[0] * vec2.f[3];
		return r;
	#endif
}

// A * Adjugate(B)
FORCE_INLINE Register mat2_mul_adj(Register vec1, Register vec2){
	#ifdef ENGINE_SIMD_SSE
		return sub(
			mul(vec1, _mm_shuffle_ps(vec2, vec2, _MM_SHUFFLE(0, 0, 3, 3))),
			mul(
				_mm_shuffle_ps(vec1, vec1, _MM_SHUFFLE(1, 0, 3, 2)),
				_mm_shuffle_ps(vec2, vec2, _MM_SHUFFLE(2, 2, 1, 1))
			)
		);
	#else
		Register r;
		r.f[0] =  vec1.f[0] * vec2.f[3] - vec1.f[2] * vec2.f[1];
		r.f[1] =  vec1.f[1] * vec2.f[3] - vec1.f[3] * vec2.f[1];
		r.f[2] = -vec1.f[0] * vec2.f[2] + vec1.f[2] * vec2.f[0];
		r.f[3] = -vec1.f[1] * vec2.f[2] + vec1.f[3] * vec2.f[0];
		return r;
	#endif
}

};	// namespace util


FORCE_INLINE void inverse(
		Register& c0,
		Register& c1,
		Register& c2,
		Register& c3){

	// submatrices
	#ifdef ENGINE_SIMD_SSE
		Register A = _mm_shuffle_ps(c0, c1, _MM_SHUFFLE(1, 0, 1, 0));
		Register C = _mm_shuffle_ps(c0, c1, _MM_SHUFFLE(3, 2, 3, 2));
		Register B = _mm_shuffle_ps(c2, c3, _MM_SHUFFLE(1, 0, 1, 0));
		Register D = _mm_shuffle_ps(c2, c3, _MM_SHUFFLE(3, 2, 3, 2));

	#elif ENGINE_SIMD_NEON
		Register A = vcombine_f32(vget_low_f32(c0), vget_low_f32(c1));
		Register B = vcombine_f32(vget_high_f32(c0), vget_high_f32(c1));
		Register C = vcombine_f32(vget_low_f32(c2), vget_low_f32(c3));
		Register D = vcombine_f32(vget_high_f32(c2), vget_high_f32(c3));

	#else
		Register A = {c0.f[0], c0.f[1], c1.f[0], c1.f[1]};
		Register C = {c0.f[2], c0.f[3], c1.f[2], c1.f[3]};
		Register B = {c2.f[0], c2.f[1], c3.f[0], c3.f[1]};
		Register D = {c2.f[2], c2.f[3], c3.f[2], c3.f[3]};
	#endif


	// submatrices determinants, det_sub = {|A|, |B|, |C|, |D|}
	#ifdef ENGINE_SIMD_SSE
		Register det_sub = sub(
			mul(
				_mm_shuffle_ps(c0,c2, _MM_SHUFFLE(2,0,2,0)),
				_mm_shuffle_ps(c1, c3, _MM_SHUFFLE(3,1,3,1))
			),
			mul(
				_mm_shuffle_ps(c0,c2, _MM_SHUFFLE(3,1,3,1)),
				_mm_shuffle_ps(c1,c3, _MM_SHUFFLE(2,0,2,0))
			)
		);

	#elif ENGINE_SIMD_NEON
		float32x4_t c02_lo = vsetq_lane_f32(
			vgetq_lane_f32(c2, 0), 
			vsetq_lane_f32(vgetq_lane_f32(c0, 0), vdupq_n_f32(0), 0), 
			2
		);
		c02_lo = vsetq_lane_f32(vgetq_lane_f32(c0, 2), c02_lo, 1);
		c02_lo = vsetq_lane_f32(vgetq_lane_f32(c2, 2), c02_lo, 3);

		float32x4_t c13_hi = vsetq_lane_f32(
			vgetq_lane_f32(c3, 1), 
			vsetq_lane_f32(vgetq_lane_f32(c1, 1), vdupq_n_f32(0), 0), 
			2
		);
		c13_hi = vsetq_lane_f32(vgetq_lane_f32(c1, 3), c13_hi, 1);
		c13_hi = vsetq_lane_f32(vgetq_lane_f32(c3, 3), c13_hi, 3);

		float32x4_t c02_hi = vsetq_lane_f32(
			vgetq_lane_f32(c2, 1), 
			vsetq_lane_f32(vgetq_lane_f32(c0, 1), vdupq_n_f32(0), 0), 
			2
		);
		c02_hi = vsetq_lane_f32(vgetq_lane_f32(c0, 3), c02_hi, 1);
		c02_hi = vsetq_lane_f32(vgetq_lane_f32(c2, 3), c02_hi, 3);

		float32x4_t c13_lo = vsetq_lane_f32(
			vgetq_lane_f32(c3, 0), 
			vsetq_lane_f32(vgetq_lane_f32(c1, 0), vdupq_n_f32(0), 0), 
			2
		);
		c13_lo = vsetq_lane_f32(vgetq_lane_f32(c1, 2), c13_lo, 1);
		c13_lo = vsetq_lane_f32(vgetq_lane_f32(c3, 2), c13_lo, 3);

		Register det_sub = vsubq_f32(
			vmulq_f32(c02_lo, c13_hi), 
			vmulq_f32(c02_hi, c13_lo)
		);

	#else
		Register det_sub = set(
			A.f[0]*A.f[3] - A.f[1]*A.f[2],
			C.f[0]*C.f[3] - C.f[1]*C.f[2],
			B.f[0]*B.f[3] - B.f[1]*B.f[2],
			D.f[0]*D.f[3] - D.f[1]*D.f[2]
		);
	#endif

	
	Register detA = splat<0>(det_sub);
	Register detB = splat<2>(det_sub);
	Register detC = splat<1>(det_sub);
	Register detD = splat<3>(det_sub);

	Register D_C = util::mat2_adj_mul(D,C);
	Register A_B = util::mat2_adj_mul(A,B);

	Register X_ = sub(mul(detD, A), util::mat2_mul(B, D_C));
	Register W_ = sub(mul(detA, D), util::mat2_mul(C, A_B));
	Register Y_ = sub(mul(detB, C), util::mat2_mul_adj(D, A_B));
	Register Z_ = sub(mul(detC, B), util::mat2_mul_adj(A, D_C));

	// main determinant
	Register detM = add(mul(detA, detD), mul(detB, detC));
	float tr_val;

	#ifdef ENGINE_SIMD_SSE
		Register tr = mul(A_B, _mm_shuffle_ps(D_C, D_C, _MM_SHUFFLE(3,1,2,0)));
		Register tr_sum = _mm_hadd_ps(tr, tr);
		tr_sum = _mm_hadd_ps(tr_sum, tr_sum);
		tr_val = _mm_cvtss_f32(tr_sum);

	#elif ENGINE_SIMD_NEON
		Register tr = mul(
			A_B, 
			vsetq_lane_f32(
				vgetq_lane_f32(D_C, 3), 
				vsetq_lane_f32(vgetq_lane_f32(D_C, 1), 
					vsetq_lane_f32(vgetq_lane_f32(D_C, 2), 
					vsetq_lane_f32(vgetq_lane_f32(D_C, 0), 
					vdupq_n_f32(0), 0), 1), 2
				),
				3
			)
		);
		tr_val = vaddvq_f32(tr);
	#else
		Register tr = mul(A_B, set(D_C.f[0], D_C.f[2], D_C.f[1], D_C.f[3]));
		tr_val = tr.f[0] + tr.f[1] + tr.f[2] + tr.f[3];
	#endif
	
	detM = sub(detM, set1(tr_val));

	Register adj_sign_mask = set(1.f, -1.f, -1.f, 1.f);
	Register r_detM = div(adj_sign_mask, detM);

	X_ = mul(X_, r_detM);
	Y_ = mul(Y_, r_detM);
	Z_ = mul(Z_, r_detM);
	W_ = mul(W_, r_detM);

	#ifdef ENGINE_SIMD_SSE
		c0 = _mm_shuffle_ps(X_, Z_, _MM_SHUFFLE(1,3,1,3));
		c1 = _mm_shuffle_ps(X_, Z_, _MM_SHUFFLE(0,2,0,2));
		c2 = _mm_shuffle_ps(Y_, W_, _MM_SHUFFLE(1,3,1,3));
		c3 = _mm_shuffle_ps(Y_, W_, _MM_SHUFFLE(0,2,0,2));

	#elif ENGINE_SIMD_NEON
		c0 = vsetq_lane_f32(vgetq_lane_f32(X_, 3), vsetq_lane_f32(vgetq_lane_f32(X_, 1), vsetq_lane_f32(vgetq_lane_f32(Z_, 3), vsetq_lane_f32(vgetq_lane_f32(Z_, 1), vdupq_n_f32(0), 2), 3), 0), 1);
		c1 = vsetq_lane_f32(vgetq_lane_f32(X_, 2), vsetq_lane_f32(vgetq_lane_f32(X_, 0), vsetq_lane_f32(vgetq_lane_f32(Z_, 2), vsetq_lane_f32(vgetq_lane_f32(Z_, 0), vdupq_n_f32(0), 2), 3), 0), 1);
		c2 = vsetq_lane_f32(vgetq_lane_f32(Y_, 3), vsetq_lane_f32(vgetq_lane_f32(Y_, 1), vsetq_lane_f32(vgetq_lane_f32(W_, 3), vsetq_lane_f32(vgetq_lane_f32(W_, 1), vdupq_n_f32(0), 2), 3), 0), 1);
		c3 = vsetq_lane_f32(vgetq_lane_f32(Y_, 2), vsetq_lane_f32(vgetq_lane_f32(Y_, 0), vsetq_lane_f32(vgetq_lane_f32(W_, 2), vsetq_lane_f32(vgetq_lane_f32(W_, 0), vdupq_n_f32(0), 2), 3), 0), 1);

	#else
		c0 = set(X_.f[3], X_.f[1], Z_.f[3], Z_.f[1]);
		c1 = set(X_.f[2], X_.f[0], Z_.f[2], Z_.f[0]);
		c2 = set(Y_.f[3], Y_.f[1], W_.f[3], W_.f[1]);
		c3 = set(Y_.f[2], Y_.f[0], W_.f[2], W_.f[0]);

	#endif
}

[[nodiscard]] FORCE_INLINE Register quat_conjugate(Register q){
	const Register mask = set(-0.0f, -0.0f, -0.0f, 0.0f);

	#ifdef ENGINE_SIMD_SSE
		return _mm_xor_ps(q, mask);
	#elif ENGINE_SIMD_NEON
		return vreinterpretq_f32_u32(veorq_u32(
			vreinterpretq_u32_f32(q), 
			vreinterpretq_u32_f32(mask)
		));
	#else
		return {-q.f[0], -q.f[1], -q.f[2], q.f[3]};
	#endif
}

[[nodiscard]] FORCE_INLINE Register quat_mul(Register a, Register b){
	// b * a.w
	Register res = mul(b, splat<3>(a));

	// + (a.xyz * b.w)
	res = fmadd(mul(a, splat<3>(b)), set(1.0f, 1.0f, 1.0f, 0.0f), res);

	// + cross(a,b)
	res = add(res, cross3(a,b));

	Register d3 = dot3_splat(a,b);
	Register w_part = mul(splat<3>(a), splat<3>(b));
	Register final_w = sub(w_part, d3);

	#ifdef ENGINE_SIMD_SSE
		return _mm_blend_ps(res, final_w, 0x08);
	#elif ENGINE_SIMD_NEON
		return vbslq_f32(vsetq_lane_u32(0xFFFFFFFF, vdupq_n_u32(0), 3), final_w, res);
	#else 
		res.f[3] = final_w.f[3];
		return res;
	#endif

}

[[nodiscard]] FORCE_INLINE Register quat_slerp(
		Register q1, Register q2, float t){
	Register d_splat = dot4_splat(q1,q2);

	#ifdef ENGINE_SIMD_SSE
		Register sign_bit = _mm_set1_ps(-0.0f);
		Register is_neg = _mm_cmplt_ps(d_splat, _mm_setzero_ps());
		Register flip_mask = _mm_and_ps(is_neg, sign_bit);
		Register target_q2 = _mm_xor_ps(q2, flip_mask);
	#elif ENGINE_SIMD_NEON
		uint32x4_t is_neg = vcltq_f32(d_splat, vdupq_n_f32(0.0f));
		Register sign_bit = vreinterpretq_f32_u32(
			vdupq_n_u32(0x80000000)
		);
        Register flip_mask = vreinterpretq_f32_u32(
			vandq_u32(is_neg, vreinterpretq_u32_f32(sign_bit))
		);
        Register target_q2 = vreinterpretq_f32_u32(
			veorq_u32(vreinterpretq_u32_f32(q2), vreinterpretq_u32_f32(flip_mask))
		);
	#else
		Register target_q2 = (x(d_splat) < 0.0f) ? neg(q2) : q2;
	#endif

	float abs_dot = x(abs(d_splat));

	float s1, s2;

	if(abs_dot > 0.9995f){
		s1 = 1.0f - t;
		s2 = t;
	}
	else{
		float theta_0 = std::acos(abs_dot);
		float theta = theta_0 * t;
		float inv_sin_theta_0 = 1.0f / std::sin(theta_0);

		s1 = std::sin(theta_0 - theta) * inv_sin_theta_0;
		s2 = std::sin(theta) * inv_sin_theta_0;
	}

	Register res = fmadd(set1(s1), q1, mul(set1(s2), target_q2));
	Register len_sq = dot4_splat(res,res);
	return mul(res, rsqrt_accurate(len_sq));
}

[[nodiscard]] FORCE_INLINE Register quat_fast_slerp(
		Register q1,
		Register q2,
		float t){
	//taken from:
	//	https://zeux.io/2015/07/23/approximating-slerp/
	//	https://zeux.io/2016/05/05/optimizing-slerp/

	Register dot_splat = dot4_splat(q1, q2);

	#ifdef ENGINE_SIMD_SSE
		Register abs_mask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF));
		Register d = _mm_and_ps(dot_splat, abs_mask);
	#elif ENGINE_SIMD_NEON
		Register d = vabsq_f32(dot_splat);
	#else
		Register d = {
			std::abs(dot_splat.f[0]), 
			std::abs(dot_splat.f[0]),
			std::abs(dot_splat.f[0]),
			std::abs(dot_splat.f[0])
		};
	#endif

	Register T = set1(t);
	Register T_minus_05 = sub(T, set1(0.5f));

	Register A, B;

	A = set1(-1.43519f);
	A = fmadd(d, A, set1(3.55645f));
	A = fmadd(d, A, set1(-3.2452f));
	A = fmadd(d, A, set1(1.0904f));

	B = set1(0.215638f);
	B = fmadd(d, B, set1(-1.06021f));
	B = fmadd(d, B, set1(0.848013f));

	Register k = mul(A, mul(T_minus_05, T_minus_05));
	k = add(k, B);

	Register T_minus_1 = sub(T, set1(1.0f));
	Register adj = mul(T, mul(T_minus_05, mul(T_minus_1, k)));
	Register ot = add(T, adj);

	#ifdef ENGINE_SIMD_SSE
		Register sign_bit = _mm_set1_ps(-0.0f);
		Register is_neg = _mm_cmplt_ps(dot_splat, _mm_setzero_ps());
		Register flip_mask = _mm_and_ps(is_neg, sign_bit);
		Register q2_corrected = _mm_xor_ps(q2, flip_mask);

	#elif ENGINE_SIMD_NEON
		uint32x4_t is_neg = vcltq_f32(dot_splat, vdupq_n_f32(0.0f));
		Register flip_mask = vreinterpretq_f32_u32(vandq_u32(is_neg, vdupq_n_u32(0x80000000)));
		Register q2_corrected = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(q2), vreinterpretq_u32_f32(flip_mask)));

	#else
		Register q2_corrected;
		if(dot_splat.f[0] < 0.0f){
			q2_corrected = { -q2.f[0], -q2.f[1], -q2.f[2], -q2.f[3] };
		}
		else{
			q2_corrected = q2;
		}

	#endif

	Register one_minus_ot = sub(set1(1.0f), ot);
	Register res = add(mul(q1, one_minus_ot), mul(q2_corrected, ot));
	return mul(
		res, 
		rsqrt_accurate(dot4_splat(res,res))
	);
}

[[nodiscard]] FORCE_INLINE Register quat_inverse(Register q){
	Register conj = quat_conjugate(q);
	Register dot = dot4_splat(q,q);
	return div(conj, dot);
}

[[nodiscard]] FORCE_INLINE Register quat_from_euler(
		float x,
		float y,
		float z){
	float hx = x * 0.5f;
	float hy = y * 0.5f;
	float hz = z * 0.5f;

	float sx = std::sin(hx);
	float cx = std::cos(hx);
	float sy = std::sin(hy);
	float cy = std::cos(hy);
	float sz = std::sin(hz);
	float cz = std::cos(hz);

	Register res = set(
		sx * cy * cz - cx * sy * sz,
		cx * sy * cz + sx * cy * sz,
		cx * cy * sz - sx * sy * cz,
		cx * cy * cz + sx * sy * sz
	);

	return res;
}

[[nodiscard]] FORCE_INLINE Register quat_to_euler(
		Register q){
	float x_val = x(q);
	float y_val = y(q);
	float z_val = z(q);
	float w_val = w(q);

	float sinr_cosp = 2.0f * (w_val * x_val + y_val * z_val);
	float cosr_cosp = 1.0f - 2.0f * (x_val * x_val + y_val * y_val);
	float res_x = std::atan2(sinr_cosp, cosr_cosp);

	float sinp = 2.0f * (w_val * y_val - z_val * x_val);
	float res_y;
	if (std::abs(sinp) >= 1.0f)
		res_y = std::copysign(3.1415926535f / 2.0f, sinp);
	else
		res_y = std::asin(sinp);

	float siny_cosp = 2.0f * (w_val * z_val + x_val * y_val);
	float cosy_cosp = 1.0f - 2.0f * (y_val * y_val + z_val * z_val);
	float res_z = std::atan2(siny_cosp, cosy_cosp);

	return set(res_x, res_y, res_z, 0.0f);
}

[[nodiscard]] FORCE_INLINE Register mat4_to_quat(
		Register c0,
		Register c1,
		Register c2){
    const float m00 = x(c0), m11 = y(c1), m22 = z(c2);
    const float m10 = y(c0), m20 = z(c0);
    const float m01 = x(c1), m21 = z(c1);
    const float m02 = x(c2), m12 = y(c2);

    const float t_x = 1.0f + m00 - m11 - m22;
    const float t_y = 1.0f + m11 - m00 - m22;
    const float t_z = 1.0f + m22 - m00 - m11;
    const float t_w = 1.0f + m00 + m11 + m22;

    const float s_yz_p = m21 + m12;
    const float s_yz_m = m21 - m12;
    const float s_zx_p = m02 + m20;
    const float s_zx_m = m02 - m20;
    const float s_xy_p = m10 + m01;
    const float s_xy_m = m10 - m01;

    const Register qx = set(t_x, s_xy_p, s_zx_p, s_yz_m);
    const Register qy = set(s_xy_p, t_y, s_yz_p, s_zx_m);
    const Register qz = set(s_zx_p, s_yz_p, t_z, s_xy_m);
    const Register qw = set(s_yz_m, s_zx_m, s_xy_m, t_w);

    Register res;
	#if defined(ENGINE_SIMD_SSE)
		const __m128 zero = _mm_setzero_ps();
		const __m128 m00_v = _mm_set1_ps(m00);
		const __m128 m11_v = _mm_set1_ps(m11);
		const __m128 m22_v = _mm_set1_ps(m22);

		const __m128 mask_z  = _mm_cmplt_ps(m22_v, zero);
		const __m128 mask_xy = _mm_cmpgt_ps(m00_v, m11_v);
		const __m128 mask_zw = _mm_cmplt_ps(m00_v, _mm_sub_ps(zero, m11_v));

		const __m128 res_xy = _mm_blendv_ps(qy, qx, mask_xy);
		const __m128 res_zw = _mm_blendv_ps(qw, qz, mask_zw);
		res = _mm_blendv_ps(res_zw, res_xy, mask_z);

	#elif defined(ENGINE_SIMD_NEON)
		const float32x4_t zero = vdupq_n_f32(0.0f);
		const float32x4_t m00_v = vdupq_n_f32(m00);
		const float32x4_t m11_v = vdupq_n_f32(m11);
		const float32x4_t m22_v = vdupq_n_f32(m22);

		const uint32x4_t mask_z  = vcltq_f32(m22_v, zero);
		const uint32x4_t mask_xy = vcgtq_f32(m00_v, m11_v);
		const uint32x4_t mask_zw = vcltq_f32(m00_v, vnegq_f32(m11_v));

		const float32x4_t res_xy = vbslq_f32(mask_xy, qx, qy);
		const float32x4_t res_zw = vbslq_f32(mask_zw, qz, qw);
		res = vbslq_f32(mask_z, res_xy, res_zw);

	#else
		if (m22 < 0.0f) {
			if (m00 > m11) res = qx;
			else res = qy;
		} else {
			if (m00 < -m11) res = qz;
			else res = qw;
		}

	#endif

    return mul(res, rsqrt_accurate(dot4_splat(res, res)));
}



} // namespace engine::math::simd
