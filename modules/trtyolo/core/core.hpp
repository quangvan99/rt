#pragma once

#include <NvInferPlugin.h>

#include <map>
#include <memory>
#include <string>

namespace trtyolo {

class TRTLogger : public nvinfer1::ILogger {
public:
    explicit TRTLogger(nvinfer1::ILogger::Severity severity = nvinfer1::ILogger::Severity::kINFO);

    TRTLogger(const TRTLogger&)            = delete;  
    TRTLogger& operator=(const TRTLogger&) = delete;  
    TRTLogger(TRTLogger&&)                 = delete;  
    TRTLogger& operator=(TRTLogger&&)      = delete;  


    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override;

private:
    nvinfer1::ILogger::Severity severity_;  
    static const std::map<nvinfer1::ILogger::Severity, std::string> severity_map_;
};

class TRTManager {
public:
    TRTManager();

    ~TRTManager();

    
    TRTManager(const TRTManager&)            = delete;  
    TRTManager& operator=(const TRTManager&) = delete;  
    TRTManager(TRTManager&&)                 = delete;  
    TRTManager& operator=(TRTManager&&)      = delete;  

    void initialize(void const* blob, std::size_t size);

    std::unique_ptr<TRTManager> clone() const;

    bool setTensorAddress(char const* tensorName, void* data);

    bool setInputShape(char const* tensorName, nvinfer1::Dims const& dims);

    bool enqueueV3(cudaStream_t stream);

    nvinfer1::Dims getTensorShape(char const* tensorName) const noexcept;

    nvinfer1::DataType getTensorDataType(char const* tensorName) const noexcept;

    nvinfer1::TensorIOMode getTensorIOMode(char const* tensorName) const noexcept;

    nvinfer1::Dims getProfileShape(char const* tensorName, int32_t profileIndex, nvinfer1::OptProfileSelector select) const noexcept;

    int32_t getNbIOTensors() const noexcept;

    char const* getIOTensorName(int32_t index) const noexcept;

private:
    std::unique_ptr<nvinfer1::IExecutionContext> context_;  
    std::shared_ptr<nvinfer1::ICudaEngine>       engine_;   
    std::unique_ptr<nvinfer1::IRuntime>          runtime_;  
    std::unique_ptr<TRTLogger>                   logger_;   
};

class CudaGraph {
public:
    explicit CudaGraph() = default;
    ~CudaGraph() { destroy(); }
    CudaGraph(const CudaGraph&)            = delete;
    CudaGraph& operator=(const CudaGraph&) = delete;
    CudaGraph(CudaGraph&&)                 = delete;
    CudaGraph& operator=(CudaGraph&&)      = delete;

    void destroy();

    void beginCapture(cudaStream_t stream);

    void endCapture(cudaStream_t stream);

    void launch(cudaStream_t stream);

    void initializeNodes(size_t num = 0);

    void updateKernelNodeParams(size_t index, void** kernelParams);

    void updateMemcpyNodeParams(size_t index, void* src, void* dst, size_t size);

private:
    cudaGraphNodeType getNodeType(size_t index);

private:
    std::unique_ptr<cudaGraphNode_t[]> nodes_;                
    cudaGraph_t                        graph_     = nullptr;  
    cudaGraphExec_t                    graphExec_ = nullptr;  
};

}  // namespace trtyolo
