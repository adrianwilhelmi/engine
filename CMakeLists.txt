cmake_minimum_required(VERSION 3.30.5)
project(SDL3WindowApp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(SDL3 REQUIRED)
find_package(Vulkan REQUIRED)

add_executable(SDL3Window src/main.cpp)
target_link_libraries(SDL3Window PRIVATE SDL3::SDL3 Vulkan::Vulkan)

include(FetchContent)
FetchContent_Declare(
	googletest
	URL https://github.com/google/googletest/archive/refs/heads/main.zip
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(memory_test test/memory_test.cpp)
target_include_directories(memory_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(memory_test PRIVATE gtest_main)

add_test(NAME memory_test COMMAND memory_test)
