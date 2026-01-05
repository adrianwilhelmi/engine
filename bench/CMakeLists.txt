set(BENCH_DEFINITIONS
	GLM_FORCE_INLINE
	GLM_FORCE_SIMD_AVX2
	GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
	NDEBUG
)

add_executable(bench_matmul matmul/matmul.cpp)
target_link_libraries(bench_matmul PRIVATE 
	EngineCore 
	benchmark::benchmark
	glm::glm
	engine_strict_flags
)

add_executable(bench_quat_slerp quat_slerp/quat_slerp.cpp)
target_link_libraries(bench_quat_slerp PRIVATE
	EngineCore
	benchmark::benchmark
	engine_strict_flags
)

if(UNIX)
	foreach(bench_target bench_matmul bench_quat_slerp)
		target_link_libraries(${bench_target} PRIVATE pthread)
		target_compile_options(${bench_target} PRIVATE -O3 -march=native)
		target_compile_definitions(${bench_target} PRIVATE ${BENCH_DEFINITIONS})
	endforeach()
endif()
