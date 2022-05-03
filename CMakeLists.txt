cmake_minimum_required(VERSION 3.0.0)
project(multibody_sim_gpu LANGUAGES CUDA CXX)

find_package(benchmark REQUIRED)
find_package(nlohmann_json REQUIRED)

include_directories(include)
add_executable(${PROJECT_NAME} main.cpp src/simulation_gpu.cu)

# set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "61")

target_link_libraries(${PROJECT_NAME} benchmark::benchmark nlohmann_json::nlohmann_json)