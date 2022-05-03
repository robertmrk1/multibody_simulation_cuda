#pragma once
#include <vector>

#include "common_gpu.cuh"
#define THREADS_PER_BLOCK 256

namespace CPU {

class Simulation {
 public:
  using CpuVector = std::vector<float>;
  using History   = std::vector<std::vector<float>>;

  Simulation(unsigned int n_particles, float dt, float duration);
  void initialize(const CpuVector& initialState, const CpuVector& masses);
  void run();

 private:
  void calculateDerivative();
  void updateState();
  void log();
  const unsigned int n_particles_;
  const unsigned int n_elements_;
  const float dt_;
  const float duration_;
  CpuVector state_;
  CpuVector state_derivative_;
  CpuVector masses_;
  CpuVector vector_state_;
  History history_;
};
};  // namespace GPU
