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

[[nodiscard]] FORCE_INLINE Register mul(Register a, float s){
	#ifdef ENGINE_SIMD_SSE
		return _mm_mul_ps(a, _mm_set1_ps(s));
	#elif ENGINE_SIMD_NEON
		return vmulq_n_f32(a,s);
	#else
		return {a.f[0] * s, a.f[1]*s, a.f[2]*s, a.f[3]*s};
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
		return _mm_dp_ps(a,b,0xF7);
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
	Register epsilon = set1(eps);
	Register diff = abs(sub(a,b));
	#ifdef ENGINE_SIMD_SSE
		__m128 cmp = _mm_cmplt_ps(diff, epsilon);
		return (_mm_movemask_ps(cmp) & 0xF) == 0xF;
	#elif ENGINE_SIMD_NEON
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
	Register epsilon = set1(eps);
	Register diff = abs(sub(a,b));
	#ifdef ENGINE_SIMD_SSE
		__m128 cmp = _mm_cmplt_ps(diff, epsilon);
		return (_mm_movemask_ps(cmp) & 0x7) == 0x7;
	#elif ENGINE_SIMD_NEON
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
		return vdupq_lane_f32(r, Index);
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
		std::swap(c0[1], c1[0]);
		std::swap(c0[2], c2[0]);
		std::swap(c0[3], c3[0]);
		std::swap(c1[2], c2[1]);
		std::swap(c1[3], c3[1]);
		std::swap(c2[3], c3[2]);

	#endif
}

} // namespace engine::math::simd
