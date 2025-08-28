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
    InferConfig   infer_config;    // < 推理选项配置
    ProcessConfig process_config;  // < 图像预处理配置
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
    // 私有的无参构造函数，仅在 clone 方法中使用
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

    std::tuple<std::string, std::string, std::string> performanceReport() {
        if (backend_->infer_config.enable_performance_report) {
            float             throughput = total_request_ / infer_cpu_trace_->totalMilliseconds() * 1000;
            std::stringstream ss;

            // 构建吞吐量字符串
            ss << "Throughput: " << throughput << " qps";
            std::string throughputStr = ss.str();
            ss.str("");  // 清空 stringstream

            auto percentiles = std::vector<float>{90, 95, 99};

            auto getLatencyStr = [&](const auto& trace, const std::string& device) {
                auto result = getPerformanceResult(trace->milliseconds(), {0.90, 0.95, 0.99});
                ss << device << " Latency: min = " << result.min << " ms, max = " << result.max
                   << " ms, mean = " << result.mean << " ms, median = " << result.median << " ms";
                for (int32_t i = 0, n = percentiles.size(); i < n; ++i) {
                    ss << ", percentile(" << percentiles[i] << "%) = " << result.percentiles[i] << " ms";
                }
                std::string output = ss.str();
                ss.str("");  // 清空 stringstream
                return output;
            };

            std::string cpuLatencyStr = getLatencyStr(infer_cpu_trace_, "CPU");
            std::string gpuLatencyStr = getLatencyStr(infer_gpu_trace_, "GPU");

            total_request_ = 0;
            infer_cpu_trace_->reset();
            infer_gpu_trace_->reset();

            return std::make_tuple(throughputStr, cpuLatencyStr, gpuLatencyStr);
        }
        // 性能报告未启用时返回空字符串
        return std::make_tuple("", "", "");
    }

    size_t batch() const {
        return backend_->max_shape.x;
    }

    // 装饰器函数
    template <typename Func, typename ReturnType>
    ReturnType withPerformanceReport(const std::vector<Image>& images, Func func) {
        if (backend_->infer_config.enable_performance_report) {
            total_request_ += (backend_->dynamic ? images.size() : backend_->max_shape.x);
            infer_cpu_trace_->start();
            infer_gpu_trace_->start();
        }

        backend_->infer(images);  // 调用推理方法
        ReturnType result = func(images.size());

        if (backend_->infer_config.enable_performance_report) {
            infer_gpu_trace_->stop();
            infer_cpu_trace_->stop();
        }

        return result;
    }

    // DetectModel 的后处理方法实现
    DetectRes postProcessDetect(int idx) {
        auto& num_tensor   = backend_->tensor_infos[1];
        auto& box_tensor   = backend_->tensor_infos[2];
        auto& score_tensor = backend_->tensor_infos[3];
        auto& class_tensor = backend_->tensor_infos[4];

        int    num     = static_cast<int*>(num_tensor.buffer->host())[idx];
        float* boxes   = static_cast<float*>(box_tensor.buffer->host()) + idx * box_tensor.shape.d[1] * box_tensor.shape.d[2];
        float* scores  = static_cast<float*>(score_tensor.buffer->host()) + idx * score_tensor.shape.d[1];
        int*   classes = static_cast<int*>(class_tensor.buffer->host()) + idx * class_tensor.shape.d[1];

        DetectRes result;
        result.num   = num;
        int box_size = box_tensor.shape.d[2];

        auto& affine_transform = backend_->infer_config.input_shape.has_value()
                                     ? backend_->affine_transforms.front()
                                     : backend_->affine_transforms[idx];

        result.boxes.reserve(num);
        result.scores.reserve(num);
        result.classes.reserve(num);

        for (int i = 0; i < num; ++i) {
            int   base_index = i * box_size;
            float left = boxes[base_index], top = boxes[base_index + 1];
            float right = boxes[base_index + 2], bottom = boxes[base_index + 3];

            affine_transform.applyTransform(left, top, &left, &top);
            affine_transform.applyTransform(right, bottom, &right, &bottom);

            result.boxes.emplace_back(Box{left, top, right, bottom});
            result.scores.push_back(scores[i]);
            result.classes.push_back(classes[i]);
        }

        return result;
    }

private:
    std::unique_ptr<TrtBackend> backend_;           // < TensorRT 后端
    unsigned long long          total_request_{0};  // < 总请求数
    std::unique_ptr<GpuTimer>   infer_gpu_trace_;   // < GPU推理计时器
    std::unique_ptr<CpuTimer>   infer_cpu_trace_;   // < CPU推理计时器
};

BaseModel::BaseModel()  = default;
BaseModel::~BaseModel() = default;

BaseModel::BaseModel(const std::string& trt_engine_file, const InferOption& infer_option)
    : impl_(std::make_unique<Impl>(trt_engine_file, infer_option)) {}

int BaseModel::batch() const {
    return impl_->batch();
}

std::tuple<std::string, std::string, std::string> BaseModel::performanceReport() {
    return impl_->performanceReport();
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
    auto processImages = [this](size_t num) -> std::vector<DetectRes> {
        std::vector<DetectRes> results(num);
        for (size_t idx = 0; idx < num; ++idx) {
            results[idx] = this->impl_->postProcessDetect(idx);
        }
        return results;
    };

    // withPerformanceReport
    return impl_->withPerformanceReport<decltype(processImages), std::vector<DetectRes>>(images, processImages);
}

DetectRes DetectModel::predict(const Image& image) {
    return predict(std::vector<Image>{image}).front();
}

}  // namespace trtyolo