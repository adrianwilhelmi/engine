option(ENGINE_BUILD_TESTS "Build unit tests" ON)


cmake_minimum_required(VERSION 3.25)
project(Engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
	add_compile_options(
		/W4
		/permissive-
		/Zc:__cplusplus
		/w14242
		/w14254
		/MP
	)

	if(CMAKE_BUILD_TYPE MATCHES Debug)
		add_compile_options(/fsanitize=address)
	endif()
endif()

if(UNIX)
	add_compile_options(
		-Wall
		-Wextra
		-Wpedantic
		-Wshadow
		-Wconversion
		-Wformat=2
	)

	if(CMAKE_BUILD_TYPE MATCHES Debug)
		add_compile_options(-fsanitize=address -g -fno-omit-frame-pointer)
		add_link_options(-fsanitize=address)
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

add_subdirectory(app)
