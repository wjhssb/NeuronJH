cmake_minimum_required(VERSION 3.8)

set(CMAKE_CXX_STANDARD 20)#设置c++版本为c++20

include_directories("include")

add_executable(test "test.cpp")