// rope_vec_ops.cc
#include <aie_api/aie.hpp>
#include <stdint.h>

extern "C" {

// positions: [64], inv_timescale: [32], radians: [64][32]
void rope_make_radians_float32(float positions[64],
                               float inv_timescale[32],
                               float radians[64][32]) {
  constexpr int V = 32; // HEAD_DIM/2 lanes
  aie::vector<float, V> v_inv = aie::load_v<V>(inv_timescale);
  for (int r = 0; r < 64; ++r) {
    float* __restrict out_row = &radians[r][0];
    aie::vector<float, V> v_pos = aie::broadcast<float, V>(positions[r]);
    aie::vector<float, V> v_row = aie::mul(v_inv, v_pos);
    aie::store_v(out_row, v_row);
  }
}

// in: [64][32], out: [64][64] (duplicate 32->left and ->right)
void pack32to64_float32(float in_mat[64][32], float out_mat[64][64]) {
  constexpr int V = 32;
  for (int r = 0; r < 64; ++r) {
    float* __restrict in_row  = &in_mat[r][0];
    float* __restrict out_row = &out_mat[r][0];
    aie::vector<float, V> v = aie::load_v<V>(in_row);
    aie::store_v(out_row + 0, v);
    aie::store_v(out_row + V, v);
  }
}

// copy left 32 cols of a 64x64 into 64x32
void copy_left32_from64_float32(float in64[64][64], float out32[64][32]) {
  constexpr int V = 32;
  for (int r = 0; r < 64; ++r) {
    aie::vector<float, V> vL = aie::load_v<V>(&in64[r][0]);
    aie::store_v(&out32[r][0], vL);
  }
}

// copy right 32 cols of a 64x64 into 64x32
void copy_right32_from64_float32(float in64[64][64], float out32[64][32]) {
  constexpr int V = 32;
  for (int r = 0; r < 64; ++r) {
    aie::vector<float, V> vR = aie::load_v<V>(&in64[r][V]);
    aie::store_v(&out32[r][0], vR);
  }
}

// join two 64x32 halves into one 64x64
void join32_to_64_float32(float left32[32][32],
                          float right32[32][32],
                          float out64[32][64]) {
  constexpr int V = 32;
  for (int r = 0; r < 32; ++r) {
    aie::vector<float, V> vL = aie::load_v<V>(&left32[r][0]);
    aie::vector<float, V> vR = aie::load_v<V>(&right32[r][0]);
    aie::store_v(&out64[r][0], vL);
    aie::store_v(&out64[r][V], vR);
  }
}

// elementwise: C = A * B   (all 64x32)
void mul32_float32(float A[64][32], float B[64][32], float C[64][32]) {
  constexpr int V = 32;
  for (int r = 0; r < 64; ++r) {
    aie::vector<float, V> vA = aie::load_v<V>(&A[r][0]);
    aie::vector<float, V> vB = aie::load_v<V>(&B[r][0]);
    aie::vector<float, V> vC = aie::mul(vA, vB);
    aie::store_v(&C[r][0], vC);
  }
}

// elementwise: C = A + B
void add32_float32(float A[64][32], float B[64][32], float C[64][32]) {
  constexpr int V = 32;
  for (int r = 0; r < 64; ++r) {
    aie::vector<float, V> vA = aie::load_v<V>(&A[r][0]);
    aie::vector<float, V> vB = aie::load_v<V>(&B[r][0]);
    aie::vector<float, V> vC = aie::add(vA, vB);
    aie::store_v(&C[r][0], vC);
  }
}

// elementwise: C = A - B
void sub32_float32(float A[64][32], float B[64][32], float C[64][32]) {
  constexpr int V = 32;
  for (int r = 0; r < 64; ++r) {
    aie::vector<float, V> vA = aie::load_v<V>(&A[r][0]);
    aie::vector<float, V> vB = aie::load_v<V>(&B[r][0]);
    aie::vector<float, V> vC = aie::sub(vA, vB);
    aie::store_v(&C[r][0], vC);
  }
}

} // extern "C"
