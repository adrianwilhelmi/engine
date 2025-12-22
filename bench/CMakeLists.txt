add_executable(bench_matmul matmul/matmul.cpp)

target_link_libraries(bench_matmul PRIVATE 
	EngineCore 
	benchmark::benchmark
	glm::glm
	engine_strict_flags
)

if(UNIX)
	target_link_libraries(bench_matmul PRIVATE pthread)
	target_compile_options(bench_matmul PRIVATE 
		-O3
		-march=native)

	target_compile_definitions(bench_matmul PRIVATE
		GLM_FORCE_INLINE
		GLM_FORCE_SIMD_AVX2
		GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
	)
endif()

target_compile_definitions(bench_matmul PRIVATE NDEBUG)
