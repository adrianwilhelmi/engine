#pragma once

#include<cmath>
#include<utility>
#include<string>

#if defined(__AVX__) || defined(__SSE4_1__) || defined(_M_AMD64) || defined(_M_X64)
	#define ENGINE_SIMD_SSE
	#include<immintrin.h>
	std::string detected_arch(){
		return "sse";
	}
#elif defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(__aarch64__)
	#define ENGINE_SIMD_NEON
	#include<arm_neon.h>
	std::string detected_arch(){
		return "neon";
	}
#else
	#define ENGINE_SIMD_NONE
	//fallback
	std::string detected_arch(){
		return "other";
	}
#endif

namespace engine::math:simd{

#ifdef ENGINE_SIMD_SSE
	using Register = __m128;
#elif ENGINE_SIMD_NEON
	using Register = float32x4_t;
#else
	struct Register {float f[4]; };
#endif

#define FORCE_INLINE __attribute__((always_inline)) inline

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

[[nodiscard]] FORCE_INLINE Register mul(Register a, float s){
	#ifdef ENGINE_SIMD_SSE
		return _mm_mul_ps(a, _mm_set1_ps(s));
	#elif ENGINE_SIMD_NEON
		return vmulq_n_f32(a,s);
	#else
		return {a.f[0] * s, a.f[1]*s, a.f[2]*s, a.f[3]*s};
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

[[nodiscard]] FORCE_INLINE float dot3_splat(Register a, Register b){
	//returns [dot, dot, dot ,dot]
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
		float32x4_t a_yzx = vextq_f32(a,a,1);
		float32x4_t b_yzx = vextq_f32(b,b,1);

		float32x4_t a_zxy = vextq_f32(a,a,2);
		float32x4_t b_zxy = vextq_f32(b,b,2);

		return vsubq_f32(vmulq_f32(a_yzx, b_zxy),vmulq_f32(a_zxy, b_yzx));

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
	//#elif ENGINE_SIMD_NEON - no easy bit masking in neon
		
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

[[nodiscard]] FORCE_INLINE bool is_close(Register a, Register b, float eps){
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


} // namespace engine::math::simd
