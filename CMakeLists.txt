cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(CUDA_MOE LANGUAGES CXX CUDA)

set(CMAKE_BUILD_TYPE Release)
set(COMPILE_CC 80)  # set the cuda arch

# lib
find_package(Torch 2.3 REQUIRED)  # your torch verison
find_package(Python3 3.10 REQUIRED Interpreter Development)  # your python version
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
add_library(${PROJECT_NAME} SHARED "${CMAKE_SOURCE_DIR}/csrc/moe_kernels.cu")

# For torch 2.3, C++17 or later compatible compiler is required to use ATen.
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES ${COMPILE_CC})
set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "")

target_link_libraries(${PROJECT_NAME} ${TORCH_PYTHON_LIBRARY})

target_compile_definitions(${PROJECT_NAME} PUBLIC TORCH_EXTENSION_NAME=${PROJECT_NAME})
target_compile_definitions(${PROJECT_NAME} PUBLIC ${TORCH_CXX_FLAGS})
target_compile_definitions(${PROJECT_NAME} PUBLIC COMPILE_CC=${COMPILE_CC})

target_include_directories(${PROJECT_NAME} PUBLIC ${TORCH_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} PUBLIC ${Python3_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} PUBLIC ${CUDA_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} PUBLIC "${CMAKE_SOURCE_DIR}/third_party/cutlass/include")
target_include_directories(${PROJECT_NAME} PUBLIC "${CMAKE_SOURCE_DIR}/third_party/cutlass/tools/util/include")
target_include_directories(${PROJECT_NAME} PUBLIC "${CMAKE_SOURCE_DIR}/third_party/cutlass_extensions/include")
