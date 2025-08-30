#include <cassert>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector_functions.hpp>
#include "backend.hpp"
#include "utils/common.hpp"

namespace trtyolo {

Image::Image(void* data, int width, int height) : ptr(data), width(width), height(height), pitch(width * sizeof(uint8_t) * 3) {
    if (width <= 0 || height <= 0) {
        throw std::invalid_argument(MAKE_ERROR_MESSAGE("Image: width and height must be positive"));
    }
}

Image::Image(void* data, int width, int height, size_t pitch)
    : ptr(data), width(width), height(height), pitch(pitch) {
    if (width <= 0 || height <= 0) {
        throw std::invalid_argument(MAKE_ERROR_MESSAGE("Image: width and height must be positive"));
    }
    if (pitch < static_cast<size_t>(width * sizeof(uint8_t) * 3)) {
        throw std::invalid_argument(MAKE_ERROR_MESSAGE("Image: pitch must >= width * 3"));
    }
}

std::ostream& operator<<(std::ostream& os, const Box& box) {
    os << "Box(left=" << box.left << ", top=" << box.top << ", right=" << box.right << ", bottom=" << box.bottom << ")";
    return os;
}

std::ostream& operator<<(std::ostream& os, const DetectRes& res) {
    os << "DetectRes(\n    num=" << res.num << ",\n    classes=[";
    for (const auto& c : res.classes) os << c << ", ";
    os << "],\n    scores=[";
    for (const auto& s : res.scores) os << s << ", ";
    os << "],\n    boxes=[\n";
    for (const auto& box : res.boxes) os << "        " << box << ",\n";
    os << "    ]\n)";
    return os;
}

class InferOption::Impl {
public:
    InferConfig   getInferConfig() const { return infer_config; }
    ProcessConfig getProcessConfig() const { return process_config; }
    void          setDeviceId(int id) { infer_config.device_id = id; }
    void          enableCudaMem() { infer_config.cuda_mem = true; }
    void          enableManagedMemory() { infer_config.enable_managed_memory = true; }
    void          enablePerformanceReport() { infer_config.enable_performance_report = true; }
    void          setInputDimensions(int width, int height) { infer_config.input_shape = make_int2(height, width); }
    void          enableSwapRB() { process_config.swap_rb = true; }
    void          setBorderValue(float value) { process_config.border_value = value; }
    void          setNormalizeParams(const std::vector<float>& mean, const std::vector<float>& std) {
        assert(mean.size() == 3 && std.size() == 3 && "ProcessConfig: requires the size of mean and std to be 3.");

        process_config.alpha.x = 1.0 / 255.0f / std[0];
        process_config.alpha.y = 1.0 / 255.0f / std[1];
        process_config.alpha.z = 1.0 / 255.0f / std[2];
        process_config.beta.x  = -mean[0] / std[0];
        process_config.beta.y  = -mean[1] / std[1];
        process_config.beta.z  = -mean[2] / std[2];
    }

private:
    InferConfig   infer_config;   
    ProcessConfig process_config;  
};

InferOption::InferOption() : impl_(std::make_unique<InferOption::Impl>()) {}
InferOption::~InferOption() = default;

void InferOption::setDeviceId(int id) { impl_->setDeviceId(id); }
void InferOption::enableCudaMem() { impl_->enableCudaMem(); }
void InferOption::enableManagedMemory() { impl_->enableManagedMemory(); }
void InferOption::enablePerformanceReport() { impl_->enablePerformanceReport(); }
void InferOption::enableSwapRB() { impl_->enableSwapRB(); }
void InferOption::setBorderValue(float border_value) { impl_->setBorderValue(border_value); }
void InferOption::setNormalizeParams(const std::vector<float>& mean, const std::vector<float>& std) { impl_->setNormalizeParams(mean, std); }
void InferOption::setInputDimensions(int width, int height) { impl_->setInputDimensions(height, width); }

class BaseModel::Impl {
public:
    Impl()  = default;
    ~Impl() = default;

    Impl(const std::string& trt_engine_file, const InferOption& infer_option)
        : backend_(std::make_unique<TrtBackend>(trt_engine_file, infer_option.impl_->getInferConfig())) {
        if (backend_->infer_config.enable_performance_report) {
            infer_gpu_trace_ = std::make_unique<GpuTimer>(backend_->stream);
            infer_cpu_trace_ = std::make_unique<CpuTimer>();
        }
    }

    std::unique_ptr<Impl> clone() const {
        auto clone_impl              = std::make_unique<Impl>();
        clone_impl->backend_         = backend_->clone();
        clone_impl->infer_gpu_trace_ = std::make_unique<GpuTimer>(clone_impl->backend_->stream);
        clone_impl->infer_cpu_trace_ = std::make_unique<CpuTimer>();
        return clone_impl;
    }

    size_t batch() const {
        return backend_->max_shape.x;
    }

    DetectRes postProcessDetect(int idx) {
        auto& out   = backend_->tensor_infos[1];
        float* o   = static_cast<float*>(out.buffer->host()) + idx * out.shape.d[1] * out.shape.d[2];

        DetectRes result;
        result.out.resize(out.shape.d[1] * out.shape.d[2]);
        std::memcpy(result.out.data(), o, out.shape.d[1] * out.shape.d[2] * sizeof(float));
        return result;
    }

    std::unique_ptr<TrtBackend> backend_;        

private:
    unsigned long long          total_request_{0};  
    std::unique_ptr<GpuTimer>   infer_gpu_trace_;   
    std::unique_ptr<CpuTimer>   infer_cpu_trace_;  
};

BaseModel::BaseModel()  = default;
BaseModel::~BaseModel() = default;

BaseModel::BaseModel(const std::string& trt_engine_file, const InferOption& infer_option)
    : impl_(std::make_unique<Impl>(trt_engine_file, infer_option)) {}

int BaseModel::batch() const {
    return impl_->batch();
}


DetectModel::DetectModel()  = default;
DetectModel::~DetectModel() = default;

DetectModel::DetectModel(const std::string& trt_engine_file, const InferOption& infer_option)
    : BaseModel(trt_engine_file, infer_option) {}

std::unique_ptr<DetectModel> DetectModel::clone() const {
    auto clone_model   = std::make_unique<DetectModel>();
    clone_model->impl_ = impl_->clone();
    return clone_model;
}

std::vector<DetectRes> DetectModel::predict(const std::vector<Image>& images) {
    impl_->backend_->infer(images);

    std::vector<DetectRes> results(images.size());
    for (size_t idx = 0; idx < images.size(); ++idx) {
        results[idx] = impl_->postProcessDetect(idx);
    }

    return results;
}

DetectRes DetectModel::predict(const Image& image) {
    return predict(std::vector<Image>{image}).front();
}

}  // namespace trtyolo