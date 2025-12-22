option(ENGINE_BUILD_TESTS "Build unit tests" ON)
option(ENGINE_NO_SIMD "Force disable SIMD and use fallback" OFF)
option(ENGINE_BUILD_BENCHMARKS "Build benchmarks" OFF)

cmake_minimum_required(VERSION 3.25)
project(Engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(engine_strict_flags INTERFACE)
if(MSVC)
	target_compile_options(engine_strict_flags INTERFACE
		/W4
		/permissive-
		/Zc:__cplusplus
		/w14242
		/w14254
		/MP
	)

	if(CMAKE_BUILD_TYPE MATCHES Debug)
		target_compile_options(engine_strict_flags INTERFACE
			/fsanitize=address
		)
	endif()
endif()

if(UNIX)
	target_compile_options(engine_strict_flags INTERFACE
		-Wall
		-Wextra
		-Wpedantic
		-Wshadow
		-Wconversion
		-Wformat=2
	)

	if(CMAKE_BUILD_TYPE MATCHES Debug)
		target_compile_options(engine_strict_flags INTERFACE
			-fsanitize=address
			-g
			-fno-omit-frame-pointer
		)
		add_link_options(engine_strict_flags INTERFACE
			-fsanitize=address
		)
	else()
		set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
	endif()
endif()

find_package(SDL3 REQUIRED)
find_package(Vulkan REQUIRED)

add_subdirectory(src)

if(ENGINE_BUILD_TESTS)
	message(STATUS "Tests are ENABLED.")

	include(FetchContent)
	FetchContent_Declare(
		googletest
		URL https://github.com/google/googletest/archive/refs/heads/main.zip
	)
	set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
	FetchContent_MakeAvailable(googletest)

	enable_testing()
	add_subdirectory(test)
else()
	message(STATUS "Tests are DISABLED.")
endif()

if(ENGINE_BUILD_BENCHMARKS)
	include(FetchContent)

	FetchContent_Declare(
		googlebenchmark
		URL https://github.com/google/benchmark/archive/refs/tags/v1.8.3.zip
	)

	FetchContent_Declare(
		glm
		URL https://github.com/g-truc/glm/archive/refs/tags/1.0.1.zip
	)

	set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
	set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
	set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)

	set(GLM_BUILD_TESTS_ OFF CACHE BOOL "" FORCE)
	set(GLM_INSTALL OFF CACHE BOOL "" FORCE)

	FetchContent_MakeAvailable(googlebenchmark glm)

	add_subdirectory(bench)
endif()

add_subdirectory(app)
