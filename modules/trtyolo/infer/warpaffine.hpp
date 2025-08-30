#pragma once

#include <cuda_runtime_api.h>

#include "utils/common.hpp"

namespace trtyolo {

struct AffineTransform {
    float3 matrix[2];         
    int    dst_offset_x;      
    int    dst_offset_y;      
    int    last_src_width_;   
    int    last_src_height_;  

    void updateMatrix(int src_width, int src_height, int dst_width, int dst_height);

    void applyTransform(float x, float y, float* transformed_x, float* transformed_y) const;
};

void cudaWarpAffine(const void* src, const int src_cols, const int src_rows, const size_t src_pitch,
                    void* dst, const int dst_cols, const int dst_rows,
                    const float3 matrix[2], const ProcessConfig config, cudaStream_t stream);

void cudaMutliWarpAffine(const void* src, const int src_cols, const int src_rows, const size_t src_pitch,
                         void* dst, const int dst_cols, const int dst_rows,
                         const float3 matrix[2], const ProcessConfig config, int num_images, cudaStream_t stream);

}  // namespace trtyolo