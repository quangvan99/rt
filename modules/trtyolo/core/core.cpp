#include "core.hpp"
#include "utils/common.hpp"

namespace trtyolo {

const std::map<nvinfer1::ILogger::Severity, std::string> TRTLogger::severity_map_ = {
    {nvinfer1::ILogger::Severity::kINTERNAL_ERROR, "INTERNAL_ERROR: "},
    {         nvinfer1::ILogger::Severity::kERROR,          "ERROR: "},
    {       nvinfer1::ILogger::Severity::kWARNING,        "WARNING: "},
    {          nvinfer1::ILogger::Severity::kINFO,           "INFO: "},
    {       nvinfer1::ILogger::Severity::kVERBOSE,        "VERBOSE: "}
};

TRTLogger::TRTLogger(nvinfer1::ILogger::Severity severity) : severity_(severity) {}

void TRTLogger::log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept {
    if (severity > severity_) return;
    std::ostream& stream = severity >= nvinfer1::ILogger::Severity::kINFO ? std::cout : std::cerr;
    auto it = severity_map_.find(severity);
    if (it != severity_map_.end()) {
        stream << it->second << msg << '\n'; 
    }
}

TRTManager::TRTManager() : context_(nullptr), engine_(nullptr), runtime_(nullptr), logger_(nullptr) {}

void TRTManager::initialize(void const* blob, std::size_t size) {
    logger_ = std::make_unique<TRTLogger>(nvinfer1::ILogger::Severity::kWARNING);

    initLibNvInferPlugins(logger_.get(), "");

    runtime_ = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(*logger_));
    if (!runtime_) {
        throw std::runtime_error("Failed to create TensorRT runtime.");
    }

    engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(runtime_->deserializeCudaEngine(blob, size));
    if (!engine_) {
        throw std::runtime_error("Failed to deserialize CUDA engine.");
    }

    context_ = std::unique_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
    if (!context_) {
        throw std::runtime_error("Failed to create execution context.");
    }
}

std::unique_ptr<TRTManager> TRTManager::clone() const {
    if (!engine_ || !runtime_) {
        throw std::runtime_error("Invalid engine or runtime in TRTManager.");
    }

    auto newManager = std::make_unique<TRTManager>();

    newManager->engine_ = engine_;

    newManager->context_ = std::unique_ptr<nvinfer1::IExecutionContext>(newManager->engine_->createExecutionContext());
    if (!newManager->context_) {
        throw std::runtime_error("Failed to create new execution context during clone.");
    }

    return newManager;
}


TRTManager::~TRTManager() {
    context_.reset();
    engine_.reset();
    runtime_.reset();
    logger_.reset();
}

bool TRTManager::setTensorAddress(char const* tensorName, void* data) {
    return context_->setTensorAddress(tensorName, data);
}

bool TRTManager::setInputShape(char const* tensorName, nvinfer1::Dims const& dims) {
    return context_->setInputShape(tensorName, dims);
}

bool TRTManager::enqueueV3(cudaStream_t stream) {
    return context_->enqueueV3(stream);
}


nvinfer1::Dims TRTManager::getTensorShape(char const* tensorName) const noexcept {
    return engine_->getTensorShape(tensorName);
}

nvinfer1::DataType TRTManager::getTensorDataType(char const* tensorName) const noexcept {
    return engine_->getTensorDataType(tensorName);
}

nvinfer1::TensorIOMode TRTManager::getTensorIOMode(char const* tensorName) const noexcept {
    return engine_->getTensorIOMode(tensorName);
}

nvinfer1::Dims TRTManager::getProfileShape(char const* tensorName, int32_t profileIndex, nvinfer1::OptProfileSelector select) const noexcept {
    return engine_->getProfileShape(tensorName, profileIndex, select);
}

int32_t TRTManager::getNbIOTensors() const noexcept {
    return engine_->getNbIOTensors();
}

char const* TRTManager::getIOTensorName(int32_t index) const noexcept {
    return engine_->getIOTensorName(index);
}

void CudaGraph::destroy() {
    
    if (graphExec_) {
        cudaGraphExecDestroy(graphExec_);
        graphExec_ = nullptr;  
    }
    
    if (graph_) {
        cudaGraphDestroy(graph_);
        graph_ = nullptr;  
    }
}

void CudaGraph::beginCapture(cudaStream_t stream) {
    CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
}

void CudaGraph::endCapture(cudaStream_t stream) {
    CHECK(cudaStreamEndCapture(stream, &graph_));
    CHECK(cudaGraphInstantiate(&graphExec_, graph_, nullptr, nullptr, 0));
}

void CudaGraph::launch(cudaStream_t stream) {
    CHECK(cudaGraphLaunch(graphExec_, stream));
    CHECK(cudaStreamSynchronize(stream));
}

void CudaGraph::initializeNodes(size_t num) {
    if (num == 0) {
        CHECK(cudaGraphGetNodes(graph_, nullptr, &num));
    }

    if (num > 0) {
        nodes_ = std::make_unique<cudaGraphNode_t[]>(num);
        CHECK(cudaGraphGetNodes(graph_, nodes_.get(), &num));
    } else {
        throw std::runtime_error("Failed to initialize nodes: graph has no nodes.");
    }
}

void CudaGraph::updateKernelNodeParams(size_t index, void** kernelParams) {
    if (getNodeType(index) != cudaGraphNodeTypeKernel) {
        throw std::runtime_error("Node at index " + std::to_string(index) + " is not a kernel node.");
    }

    cudaKernelNodeParams kernelNodeParams;
    CHECK(cudaGraphKernelNodeGetParams(nodes_[index], &kernelNodeParams));

    kernelNodeParams.kernelParams = kernelParams;
    CHECK(cudaGraphExecKernelNodeSetParams(graphExec_, nodes_[index], &kernelNodeParams));
}

void CudaGraph::updateMemcpyNodeParams(size_t index, void* src, void* dst, size_t size) {
    if (getNodeType(index) != cudaGraphNodeTypeMemcpy) {
        throw std::runtime_error("Node at index " + std::to_string(index) + " is not a memcpy node.");
    }

    cudaMemcpy3DParms memcpyNodeParams;
    CHECK(cudaGraphMemcpyNodeGetParams(nodes_[index], &memcpyNodeParams));

    memcpyNodeParams.srcPtr = make_cudaPitchedPtr(src, size, size, 1);
    memcpyNodeParams.dstPtr = make_cudaPitchedPtr(dst, size, size, 1);
    memcpyNodeParams.extent = make_cudaExtent(size, 1, 1);

    CHECK(cudaGraphExecMemcpyNodeSetParams(graphExec_, nodes_[index], &memcpyNodeParams));
}

cudaGraphNodeType CudaGraph::getNodeType(size_t index) {
    cudaGraphNodeType nodeType;
    CHECK(cudaGraphNodeGetType(nodes_[index], &nodeType));
    return nodeType;
}

}  // namespace trtyolo
