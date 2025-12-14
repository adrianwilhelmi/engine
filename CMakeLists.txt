cmake_minimum_required(VERSION 3.30.5)
project(SDL3WindowApp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(SDL3 REQUIRED)
find_package(Vulkan REQUIRED)

add_executable(SDL3Window src/main.cpp)
target_link_libraries(SDL3Window PRIVATE SDL3::SDL3 Vulkan::Vulkan)
