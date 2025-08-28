#pragma once

#include <NvInferRuntime.h>

#include <memory>
#include <numeric>
#include <string>

namespace trtyolo {

class BaseBuffer {
public:
    virtual ~BaseBuffer() = default;

    virtual void allocate(size_t size) = 0;

    virtual void free() = 0;

    virtual void* device() = 0;

    virtual void* host() = 0;

    virtual size_t size() const = 0;

    virtual void hostToDevice(cudaStream_t stream = nullptr) = 0;

    virtual void deviceToHost(cudaStream_t stream = nullptr) = 0;
};


class DeviceBuffer : public BaseBuffer {
public:
    DeviceBuffer() : size_(0), device_(nullptr) {}
    DeviceBuffer(const DeviceBuffer&)            = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept;
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;
    ~DeviceBuffer() { free(); }

    void   allocate(size_t size) override;
    void   free() override;
    void*  device() override;
    void*  host() override;
    size_t size() const override;
    void   hostToDevice(cudaStream_t stream = nullptr) override;
    void   deviceToHost(cudaStream_t stream = nullptr) override;

private:
    void*  device_;  
    size_t size_;    
};

class DiscreteBuffer : public BaseBuffer {
public:
    DiscreteBuffer() : size_(0), host_(nullptr), device_(nullptr) {}
    DiscreteBuffer(const DiscreteBuffer&)            = delete;
    DiscreteBuffer& operator=(const DiscreteBuffer&) = delete;
    DiscreteBuffer(DiscreteBuffer&& other) noexcept;
    DiscreteBuffer& operator=(DiscreteBuffer&& other) noexcept;
    ~DiscreteBuffer() { free(); }

    void   allocate(size_t size) override;
    void   free() override;
    void*  device() override;
    void*  host() override;
    size_t size() const override;
    void   hostToDevice(cudaStream_t stream = nullptr) override;
    void   deviceToHost(cudaStream_t stream = nullptr) override;

private:
    void*  host_;    
    void*  device_; 
    size_t size_;   
};

class UnifiedBuffer : public BaseBuffer {
public:
    UnifiedBuffer() : size_(0), host_(nullptr), device_(nullptr) {}
    UnifiedBuffer(const UnifiedBuffer&)            = delete;
    UnifiedBuffer& operator=(const UnifiedBuffer&) = delete;
    UnifiedBuffer(UnifiedBuffer&& other) noexcept;
    UnifiedBuffer& operator=(UnifiedBuffer&& other) noexcept;
    ~UnifiedBuffer() { free(); }

    void   allocate(size_t size) override;
    void   free() override;
    void*  device() override;
    void*  host() override;
    size_t size() const override;
    void   hostToDevice(cudaStream_t stream = nullptr) override;
    void   deviceToHost(cudaStream_t stream = nullptr) override;

private:
    void*  host_;   
    void*  device_;  
    size_t size_;   
};


class MappedBuffer : public BaseBuffer {
public:
    MappedBuffer() : size_(0), host_(nullptr), device_(nullptr) {}
    MappedBuffer(const MappedBuffer&)            = delete;
    MappedBuffer& operator=(const MappedBuffer&) = delete;
    MappedBuffer(MappedBuffer&& other) noexcept;
    MappedBuffer& operator=(MappedBuffer&& other) noexcept;
    ~MappedBuffer() { free(); }

    void   allocate(size_t size) override;
    void   free() override;
    void*  device() override;
    void*  host() override;
    size_t size() const override;
    void   hostToDevice(cudaStream_t stream = nullptr) override;
    void   deviceToHost(cudaStream_t stream = nullptr) override;

private:
    void*  host_;    
    void*  device_; 
    size_t size_;    
};


enum class BufferType {
    Device,    
    Discrete, 
    Unified,  
    Mapped    
};


class BufferFactory {
public:
    static std::unique_ptr<BaseBuffer> createBuffer(BufferType type);
};


struct TensorInfo {
private:
    nvinfer1::DataType dtype_;           
    size_t             bytes_;           

public:
    std::string                 name;    
    nvinfer1::Dims              shape;   
    bool                        input;   
    std::unique_ptr<BaseBuffer> buffer;  

    TensorInfo(const std::string name, const nvinfer1::Dims& shape, const nvinfer1::DataType dtype, const bool input, BufferType buffer_type)
        : name(name), shape(shape), dtype_(dtype), input(input) {
        buffer = BufferFactory::createBuffer(buffer_type);
        update();
    }

    void update() {
        bytes_ = std::accumulate(shape.d, shape.d + shape.nbDims, 1, std::multiplies<int>()) * dtype_to_bytes(dtype_);
        buffer->allocate(bytes_);
    }

private:

    size_t dtype_to_bytes(nvinfer1::DataType dtype) {
        switch (dtype) {
            case nvinfer1::DataType::kINT32:
            case nvinfer1::DataType::kFLOAT:
                return 4U;
            case nvinfer1::DataType::kHALF:
                return 2U;
            case nvinfer1::DataType::kBOOL:
            case nvinfer1::DataType::kUINT8:
            case nvinfer1::DataType::kINT8:
            case nvinfer1::DataType::kFP8:
                return 1U;
            default:
                break;
        }
        return 0;
    }
};

}  // namespace trtyolo
