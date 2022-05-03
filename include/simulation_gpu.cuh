#pragma once
#include <vector>

namespace GPU {

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
  void free();
  void save();
  const unsigned int threads_per_block_ = 256;
  const unsigned int number_of_blocks_particles_;
  const unsigned int number_of_blocks_elements_;
  const unsigned int n_particles_;
  const unsigned int n_elements_;
  const unsigned int state_bytes_;
  const float dt_;
  const float duration_;
  float* state_;
  float* state_derivative_;
  float* masses_;
  CpuVector vector_state_;
  History history_;
};
};  // namespace GPU
