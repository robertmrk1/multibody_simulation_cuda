#include <benchmark/benchmark.h>

#include <iostream>
#include <vector>

#include "simulation_gpu.cuh"

#define N_PARTICLES 100
#define DT 0.016666667
#define DURATION 60

#define DISTANCE 10

using namespace std;

vector<float> initialState() {
  vector<float> initial_state;
  initial_state.reserve(6 * N_PARTICLES);
  for (size_t i = 0; i < N_PARTICLES; ++i) {
    float x = -DISTANCE + static_cast<float>(rand()) /
                              (static_cast<float>(RAND_MAX / (2 * DISTANCE)));
    float y = -DISTANCE + static_cast<float>(rand()) /
                              (static_cast<float>(RAND_MAX / (2 * DISTANCE)));
    float z = -DISTANCE + static_cast<float>(rand()) /
                              (static_cast<float>(RAND_MAX / (2 * DISTANCE)));
    initial_state.push_back(x);
    initial_state.push_back(y);
    initial_state.push_back(z);
    initial_state.push_back(0);
    initial_state.push_back(0);
    initial_state.push_back(0);
  }
  return initial_state;
}

// BENCHMARK(benchMarkGPU)->Unit(benchmark::kMillisecond);
// BENCHMARK(benchMarkCPU)->Unit(benchmark::kMillisecond);
// BENCHMARK_MAIN();

int main() {
  vector<float> initial_state = initialState();
  vector<float> masses(N_PARTICLES, 1.0);

  GPU::Simulation simulation(N_PARTICLES, DT, DURATION);
  simulation.initialize(initial_state, masses);
  simulation.run();

  // for (auto &vec : simulation.history_) {
  //   for (auto &elem : vec) {
  //     cout << elem << ", ";
  //   }
  //   cout << "\n";
  // }
  return 0;
}
