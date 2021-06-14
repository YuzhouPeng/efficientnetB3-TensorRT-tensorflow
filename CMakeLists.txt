cmake_minimum_required(VERSION 2.6)

project(Efficientnet)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
include_directories(/home/pengyuzhou/Downloads/TensorRT-7.2.1.6/include)
link_directories(/home/pengyuzhou/Downloads/TensorRT-7.2.1.6/lib)

# tensorrt dirs 
include_directories(/usr/local/cuda-11.0/targets/x86_64-linux/include)
link_directories(/usr/local/cuda-11.0/targets/x86_64-linux/lib)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")



find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})


add_executable(efficientnet ${PROJECT_SOURCE_DIR}/efficientnetb3.cpp)
target_link_libraries(efficientnet nvinfer)
target_link_libraries(efficientnet cudart)
target_link_libraries(efficientnet ${OpenCV_LIBS})

add_definitions(-O2 -pthread)

