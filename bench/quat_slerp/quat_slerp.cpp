#include<iostream>
#include<fstream>
#include<vector>
#include<string>
#include<cmath>

#include<core/math/quat.hpp>

#include<benchmark/benchmark.h>

using namespace engine::math;

struct BenchData{
	std::vector<Quat> quats_a;
	std::vector<Quat> quats_b;
	std::vector<float> ts;

	std::size_t count = 0;
};

BenchData g_data;

void load_data(){
	std::ifstream fa("quat_a.txt"), fb("quat_b.txt"), ft("ts.txt");
	float val;
	std::vector<float> data_a, data_b, data_t;

	while(fa >> val) data_a.push_back(val);
	while(fb >> val) data_b.push_back(val);
	while(ft >> val) data_t.push_back(val);

	g_data.count = data_t.size();

	g_data.quats_a.resize(g_data.count);
	g_data.quats_b.resize(g_data.count);
	g_data.ts.resize(g_data.count);

	for(std::size_t i = 0; i < g_data.count; ++i){
		g_data.quats_a[i].set_x(
			data_a[i*4 + 0]
		);
		g_data.quats_a[i].set_y(
			data_a[i*4 + 1]
		);
		g_data.quats_a[i].set_z(
			data_a[i*4 + 2]
		);
		g_data.quats_a[i].set_w(
			data_a[i*4 + 3]
		);

		g_data.quats_b[i].set_x(
			data_b[i*4 + 0]
		);
		g_data.quats_b[i].set_y(
			data_b[i*4 + 1]
		);
		g_data.quats_b[i].set_z(
			data_b[i*4 + 2]
		);
		g_data.quats_b[i].set_w(
			data_b[i*4 + 3]
		);

		g_data.ts[i] = data_t[i];
	}
}

Quat naive_slerp(
		const Quat& q1,
		const Quat& q2,
		float t){
	float dot = q1.get_x() * q2.get_x() + q1.get_y() * q2.get_y() +
				q1.get_z() * q2.get_z() + q1.get_w() * q2.get_w();

	Quat target = q2;
	if(dot < 0.0f){
		dot = -dot;
		target = Quat(
			-q2.get_x(),
			-q2.get_y(),
			-q2.get_z(),
			-q2.get_w()
		);
	}

	if(dot > 0.9995f){
		return Quat(
			q1.get_x() + t * (target.get_x() - q1.get_x()),
			q1.get_y() + t * (target.get_y() - q1.get_y()),
			q1.get_z() + t * (target.get_z() - q1.get_z()),
			q1.get_w() + t * (target.get_w() - q1.get_w())
		).normalized();
	}

	float theta_0 = std::acos(dot);
	float theta = theta_0 * t;
	float sin_theta = std::sin(theta);
	float sin_theta_0 = std::sin(theta_0);

	float s0 = std::cos(theta) - dot * sin_theta / sin_theta_0;
	float s1 = sin_theta / sin_theta_0;

	return Quat(
		(s0 * q1.get_x()) + (s1 * target.get_x()),
		(s0 * q1.get_y()) + (s1 * target.get_y()),
		(s0 * q1.get_z()) + (s1 * target.get_z()),
		(s0 * q1.get_w()) + (s1 * target.get_w())
	);
}

static void BM_naive_slerp(benchmark::State& state){
	std::size_t i = 0;
	for(auto _ : state){
		std::size_t idx = i % g_data.count;
		Quat out = naive_slerp(
			g_data.quats_a[idx],
			g_data.quats_b[idx],
			g_data.ts[idx]
		);

		benchmark::DoNotOptimize(out);
		++i;
	}
}
BENCHMARK(BM_naive_slerp)->Repetitions(10)->DisplayAggregatesOnly(true);

static void BM_simd_slerp(benchmark::State& state){
	std::size_t i = 0;
	for(auto _ : state){
		std::size_t idx = i % g_data.count;
		Quat out = Quat::slerp(
			g_data.quats_a[idx],
			g_data.quats_b[idx],
			g_data.ts[idx]
		);

		benchmark::DoNotOptimize(out);
		++i;
	}
}
BENCHMARK(BM_simd_slerp)->Repetitions(10)->DisplayAggregatesOnly(true);

static void BM_fast_slerp(benchmark::State& state){
	std::size_t i = 0;
	for(auto _ : state){
		std::size_t idx = i % g_data.count;
		Quat out = Quat::slerp_fast(
			g_data.quats_a[idx],
			g_data.quats_b[idx],
			g_data.ts[idx]
		);

		benchmark::DoNotOptimize(out);
		++i;
	}
}
BENCHMARK(BM_fast_slerp)->Repetitions(10)->DisplayAggregatesOnly(true);

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
