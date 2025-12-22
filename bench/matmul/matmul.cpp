
#include<iostream>
#include<fstream>
#include<vector>
#include<array>
#include<string>
#include<map>

#include<core/math/mat4.hpp>

#include<benchmark/benchmark.h>
#include<glm/glm.hpp>
#include<glm/gtc/type_ptr.hpp>

using namespace engine::math;

struct AlignedGLMMat {
    alignas(32) glm::mat4 matrix;
};

struct BenchData{
	std::vector<std::array<float,16>> mats_a_naive;
	std::vector<std::array<float,16>> mats_b_naive;

	std::vector<Mat4> mats_a_custom;
	std::vector<Mat4> mats_b_custom;

	std::vector<AlignedGLMMat> mats_a_glm;
	std::vector<AlignedGLMMat> mats_b_glm;

	std::size_t count = 0;
};

BenchData g_data;

void load_data(){
	std::ifstream fa("m_a.txt"), fb("m_b.txt");
	float val;
	std::vector<float> data_a, data_b;

	while(fa >> val) data_a.push_back(val);
	while(fb >> val) data_b.push_back(val);

	g_data.count = data_a.size() / 16;

	g_data.mats_a_naive.resize(g_data.count);
	g_data.mats_b_naive.resize(g_data.count);
	g_data.mats_a_custom.resize(g_data.count);
	g_data.mats_b_custom.resize(g_data.count);
	g_data.mats_a_glm.resize(g_data.count);
	g_data.mats_b_glm.resize(g_data.count);

	for(std::size_t i = 0; i < g_data.count; ++i){
		for(std::size_t j = 0; j < 16; ++j){
			g_data.mats_a_naive[i][j] = data_a[i*16 + j];
			g_data.mats_b_naive[i][j] = data_b[i*16 + j];
		}

		g_data.mats_a_custom[i] = Mat4(
			g_data.mats_a_naive[i].data()
		).transpose();
		g_data.mats_b_custom[i] = Mat4(
			g_data.mats_b_naive[i].data()
		).transpose();

		g_data.mats_a_glm[i].matrix = glm::transpose(
			glm::make_mat4(g_data.mats_a_naive[i].data())
		);
		g_data.mats_b_glm[i].matrix = glm::transpose(
			glm::make_mat4(g_data.mats_b_naive[i].data())
		);
	}
}

void naive_matmul(
		const float* a,
		const float* b,
		float* out){
	for(int i = 0; i < 4; ++i){
		for(int j = 0; j < 4; ++j){
			float sum = 0.0f;
			for(int k = 0; k < 4; ++k){
				sum += a[i*4+k] * b[k * 4 + j];
			}
			out[i*4+j] = sum;
		}
	}
}

static void BM_naive_matmul(benchmark::State& state){
	float out[16];
	std::size_t i = 0;
	for(auto _ : state){
		std::size_t idx = i % g_data.count;
		naive_matmul(
			g_data.mats_a_naive[idx].data(),
			g_data.mats_b_naive[idx].data(),
			out
		);

		benchmark::DoNotOptimize(out);
		++i;
	}
}
BENCHMARK(BM_naive_matmul)->Repetitions(10)->DisplayAggregatesOnly(true);

static void BM_custom_matmul(benchmark::State& state){
	Mat4 out;
	std::size_t i = 0;
	for(auto _ : state){
		std::size_t idx = i % g_data.count;
		out = Mat4::matmul(
			g_data.mats_a_custom[idx],
			g_data.mats_b_custom[idx]
		);

		benchmark::DoNotOptimize(out);
		++i;
	}
}
BENCHMARK(BM_custom_matmul)->Repetitions(10)->DisplayAggregatesOnly(true);

static void BM_glm_matmul(benchmark::State& state){
	glm::mat4 out;
	std::size_t i = 0;
	for(auto _ : state){
		std::size_t idx = i % g_data.count;
		out = g_data.mats_a_glm[idx].matrix * g_data.mats_b_glm[idx].matrix;

		benchmark::DoNotOptimize(out);
		++i;
	}
}
BENCHMARK(BM_glm_matmul)->Repetitions(10)->DisplayAggregatesOnly(true);

int main(int argc, char**argv){
	#ifdef GLM_FORCE_SIMD_AVX2
		std::cout << "++ GLM_FORCE_SIMD_AVX2 is DEFINED" << std::endl;
	#else
		std::cout << "-- GLM_FORCE_SIMD_AVX2 is NOT DEFINED" << std::endl;
	#endif
	
	#if defined(__AVX2__)
		std::cout << "++ hardware AVX2 support is ENABLED in compiler" << std::endl;
	#else
		std::cout << "-- hardware AVX2 support is NOT ENABLED in compiler" << std::endl;
	#endif

		

	load_data();

	if(g_data.count == 0){
		std::cerr << "couldnt load data.." << std::endl;
		return 1;
	}

	::benchmark::Initialize(&argc, argv);

	::benchmark::RunSpecifiedBenchmarks();
	::benchmark::Shutdown();

	return 0;
}
