cmake_minimum_required(VERSION 3.31)
project(SDL3WindowApp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(SDL3 REQUIRED)

add_executable(SDL3Window src/main.cpp)
target_link_libraries(SDL3Window PRIVATE SDL3::SDL3)
