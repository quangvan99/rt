#pragma once

#include <NvInferRuntime.h>

#include <memory>
#include <string>
#include <vector>

#include "core/buffer.hpp"
#include "core/core.hpp"
#include "trtyolo.hpp"
#include "utils/common.hpp"
#include "warpaffine.hpp"

namespace trtyolo {

class TrtBackend {
public:

    TrtBackend(const std::string& trt_engine_file, const InferConfig& infer_config);

    TrtBackend() = default;

    ~TrtBackend();

    std::unique_ptr<TrtBackend> clone();

    void infer(const std::vector<Image>& inputs);

    cudaStream_t                 stream;            
    InferConfig                  infer_config;      
    std::vector<TensorInfo>      tensor_infos;      
    std::vector<AffineTransform> affine_transforms; 
    int4                         min_shape;         
    int4                         max_shape;         
    bool                         dynamic;           

private:
    void getTensorInfo();
    void initialize();
    void captureCudaGraph();
    void dynamicInfer(const std::vector<Image>& inputs);
    void staticInfer(const std::vector<Image>& inputs);

    std::unique_ptr<TRTManager> manager_;        
    CudaGraph                   cuda_graph_;     
    std::unique_ptr<BaseBuffer> inputs_buffer_;  
    BufferType                  buffer_type_;    

    bool zero_copy_;                             

    int input_size_;                             
    int infer_size_;                             
};

}  // namespace trtyolo