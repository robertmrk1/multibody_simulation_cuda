#pragma once
#include <chrono>
#include <vector>
#include <iostream>

using namespace std::chrono;
namespace CPU {
template <class T>
std::vector<T> addVectors(const std::vector<T> &vec1,
                          const std::vector<T> &vec2) {
  
  size_t n_elements = vec1.size();

  std::vector<T> vec_out(n_elements);
  auto start = high_resolution_clock::now();
  for (size_t i = 0; i < n_elements; ++i) {
    vec_out[i] = vec1[i] + vec2[i];
  }
  auto end = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(end - start);
  std::cout << "cpu kernel time: " << duration.count() << "\n";
  return vec_out;
}
};  // namespace CPU
