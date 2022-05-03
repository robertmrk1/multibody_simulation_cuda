#include <chrono>
#include <iostream>
#include <nlohmann/json.hpp>
#include <fstream>

#include "simulation_gpu.cuh"
using namespace std::chrono;
using json = nlohmann::json;

using namespace GPU;

__global__ void kernelDerivative(const float *state, float *derivative,
                                 const float *masses,
                                 unsigned int n_particles) {
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (i < n_particles) {
    int pos_idx_i   = i * 6;
    int vel_idx_i_0 = pos_idx_i + 3;
    int vel_idx_i_1 = vel_idx_i_0 + 1;
    int vel_idx_i_2 = vel_idx_i_0 + 2;

    derivative[pos_idx_i]     = state[vel_idx_i_0];
    derivative[pos_idx_i + 1] = state[vel_idx_i_1];
    derivative[pos_idx_i + 2] = state[vel_idx_i_2];

    derivative[vel_idx_i_0] = 0;
    derivative[vel_idx_i_1] = 0;
    derivative[vel_idx_i_2] = 0;

    float pos_x_i = state[pos_idx_i];
    float pos_y_i = state[pos_idx_i + 1];
    float pos_z_i = state[pos_idx_i + 2];

    float mass_i = masses[i];

    for (int j = 0; j < n_particles; ++j) {
      if (i == j) continue;

      int pos_idx_j = j * 6;

      float pos_x_j = state[pos_idx_j];
      float pos_y_j = state[pos_idx_j + 1];
      float pos_z_j = state[pos_idx_j + 2];

      float mass_j = masses[j];

      float delta_x_ij = pos_x_i - pos_x_j;
      float delta_y_ij = pos_y_i - pos_y_j;
      float delta_z_ij = pos_z_i - pos_z_j;

      float inverse_distance_ij =
          rsqrtf(delta_x_ij * delta_x_ij + delta_y_ij * delta_y_ij +
                 delta_z_ij * delta_z_ij + 0.1);

      float cubed_inverse_distance_ij =
          inverse_distance_ij * inverse_distance_ij * inverse_distance_ij;

      float acceleration_factor_ij = -cubed_inverse_distance_ij * mass_j;

      derivative[vel_idx_i_0] += acceleration_factor_ij * delta_x_ij;
      derivative[vel_idx_i_1] += acceleration_factor_ij * delta_y_ij;
      derivative[vel_idx_i_2] += acceleration_factor_ij * delta_z_ij;
    }
  }
}

__global__ void kernelUpdateState(float *state, const float *derivative,
                                  float dt, unsigned int n_elements) {
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (i < n_elements) {
    state[i] += dt * derivative[i];
  }
}

Simulation::Simulation(unsigned int n_particles, float dt, float duration)
    : n_particles_(n_particles),
      n_elements_(n_particles * 6),
      state_bytes_(sizeof(float) * n_particles * 6),
      dt_(dt),
      duration_(duration),
      number_of_blocks_particles_((n_particles + threads_per_block_ - 1) /
                                  threads_per_block_),
      number_of_blocks_elements_((n_particles * 6 + threads_per_block_ - 1) /
                                 threads_per_block_) {
  cudaMalloc(&state_, state_bytes_);
  cudaMalloc(&state_derivative_, state_bytes_);
  cudaMalloc(&masses_, sizeof(float) * n_particles);
}

void Simulation::initialize(const CpuVector &initialState,
                            const CpuVector &masses) {
  cudaMemcpy(state_, initialState.data(), sizeof(float) * n_elements_,
             cudaMemcpyHostToDevice);
  cudaMemcpy(masses_, masses.data(), sizeof(float) * n_particles_,
             cudaMemcpyHostToDevice);
  vector_state_ = initialState;
}

void Simulation::calculateDerivative() {
  kernelDerivative<<<number_of_blocks_particles_, threads_per_block_>>>(
      state_, state_derivative_, masses_, n_particles_);
  cudaDeviceSynchronize();
}

void Simulation::updateState() {
  kernelUpdateState<<<number_of_blocks_elements_, threads_per_block_>>>(
      state_, state_derivative_, dt_, n_elements_);
  cudaDeviceSynchronize();
}

void Simulation::log() {
  cudaMemcpy(vector_state_.data(), state_, state_bytes_,
             cudaMemcpyDeviceToHost);
  history_.push_back(vector_state_);
}

void Simulation::free() {
  cudaFree(state_);
  cudaFree(state_derivative_);
  cudaFree(masses_);
}

void Simulation::save() {
  json j;

  j["states"] = history_;
  j["n_particles"] = n_particles_;

  std::ofstream file("../data_gpu.json");
  file << j;

  std::cout << history_.size() << "\n";
}

void Simulation::run() {
  float time = 0;
  while (time < duration_) {
    calculateDerivative();
    updateState();

    log();
    time += dt_;
    std::cout << "time: " << time << "\n";
  }
  free();
  save();
}