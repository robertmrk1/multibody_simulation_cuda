#pragma once
#define THREADS_PER_BLOCK 256

namespace GPU {

class Common {
 public:
  Common(size_t n_elements);
  void addVectorsH2H(float* vec1, float* vec2, float* vec3,
                     const size_t n_elemets);
  void free();
  void test(size_t n_elements);

 private:
 
  float *gpu_vec1_;
  float *gpu_vec2_;
  float *gpu_vec3_;
};
};  // namespace GPU
