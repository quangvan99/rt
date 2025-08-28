#include "buffer.hpp"
#include "utils/common.hpp"

namespace trtyolo {

DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept
    : size_(other.size_), device_(other.device_) {
    other.device_ = nullptr;
    other.size_   = 0;
}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
        free();
        size_         = other.size_;
        device_       = other.device_;
        other.device_ = nullptr;
        other.size_   = 0;
    }
    return *this;
}

void DeviceBuffer::allocate(size_t size) {
    if (size > size_) {
        free();
        CHECK(cudaMalloc(&device_, size)); 
        size_ = size;
    }
}

void DeviceBuffer::free() {
    if (device_) CHECK(cudaFree(device_));  
    device_ = nullptr;
    size_   = 0;
}

void* DeviceBuffer::device() {
    return device_;
}

void* DeviceBuffer::host() {
    return nullptr;
}

size_t DeviceBuffer::size() const {
    return size_;
}

void DeviceBuffer::hostToDevice(cudaStream_t stream) {}

void DeviceBuffer::deviceToHost(cudaStream_t stream) {}

DiscreteBuffer::DiscreteBuffer(DiscreteBuffer&& other) noexcept
    : size_(other.size_), host_(other.host_), device_(other.device_) {
    other.host_   = nullptr;
    other.device_ = nullptr;
    other.size_   = 0;
}

DiscreteBuffer& DiscreteBuffer::operator=(DiscreteBuffer&& other) noexcept {
    if (this != &other) {
        free();
        size_         = other.size_;
        host_         = other.host_;
        device_       = other.device_;
        other.host_   = nullptr;
        other.device_ = nullptr;
        other.size_   = 0;
    }
    return *this;
}

void DiscreteBuffer::allocate(size_t size) {
    if (size > size_) {
        free();
        CHECK(cudaMallocHost(&host_, size)); 
        CHECK(cudaMalloc(&device_, size));   
        size_ = size;
    }
}

void DiscreteBuffer::free() {
    if (host_) {
        CHECK(cudaFreeHost(host_)); 
        host_ = nullptr;           
    }
    if (device_) {
        CHECK(cudaFree(device_));  
        device_ = nullptr;         
    }
    size_ = 0;
}

void* DiscreteBuffer::device() {
    return device_;
}

void* DiscreteBuffer::host() {
    return host_;
}

size_t DiscreteBuffer::size() const {
    return size_;
}

void DiscreteBuffer::hostToDevice(cudaStream_t stream) {
    if (stream) {
        CHECK(cudaMemcpyAsync(device_, host_, size_, cudaMemcpyHostToDevice, stream));  
    } else {
        CHECK(cudaMemcpy(device_, host_, size_, cudaMemcpyHostToDevice));              
    }
}

void DiscreteBuffer::deviceToHost(cudaStream_t stream) {
    if (stream) {
        CHECK(cudaMemcpyAsync(host_, device_, size_, cudaMemcpyDeviceToHost, stream));
    } else {
        CHECK(cudaMemcpy(host_, device_, size_, cudaMemcpyDeviceToHost));             
    }
}

UnifiedBuffer::UnifiedBuffer(UnifiedBuffer&& other) noexcept
    : size_(other.size_), host_(other.host_), device_(other.device_) {
    other.host_   = nullptr;
    other.device_ = nullptr;
    other.size_   = 0;
}

UnifiedBuffer& UnifiedBuffer::operator=(UnifiedBuffer&& other) noexcept {
    if (this != &other) {
        free();
        size_         = other.size_;
        host_         = other.host_;
        device_       = other.device_;
        other.host_   = nullptr;
        other.device_ = nullptr;
        other.size_   = 0;
    }
    return *this;
}

void UnifiedBuffer::allocate(size_t size) {
    if (size > size_) {
        free();
        CHECK(cudaMallocManaged(&host_, size));  
        device_ = host_;                      
        size_   = size;
    }
}

void UnifiedBuffer::free() {
    if (host_) CHECK(cudaFree(host_));  
    host_   = nullptr;
    device_ = nullptr;
    size_   = 0;
}

void* UnifiedBuffer::device() {
    return device_;
}

void* UnifiedBuffer::host() {
    return host_;
}

size_t UnifiedBuffer::size() const {
    return size_;
}

void UnifiedBuffer::hostToDevice(cudaStream_t stream) {}

void UnifiedBuffer::deviceToHost(cudaStream_t stream) {}

MappedBuffer::MappedBuffer(MappedBuffer&& other) noexcept
    : size_(other.size_), host_(other.host_), device_(other.device_) {
    other.host_   = nullptr;
    other.device_ = nullptr;
    other.size_   = 0;
}

MappedBuffer& MappedBuffer::operator=(MappedBuffer&& other) noexcept {
    if (this != &other) {
        free();
        size_         = other.size_;
        host_         = other.host_;
        device_       = other.device_;
        other.host_   = nullptr;
        other.device_ = nullptr;
        other.size_   = 0;
    }
    return *this;
}

void MappedBuffer::allocate(size_t size) {
    if (size > size_) {
        free();
        CHECK(cudaHostAlloc(&host_, size, cudaHostAllocMapped));
        CHECK(cudaHostGetDevicePointer(&device_, host_, 0));    
        size_ = size;
    }
}

void MappedBuffer::free() {
    if (host_) CHECK(cudaFreeHost(host_)); 
    host_   = nullptr;
    device_ = nullptr;
    size_   = 0;
}

void* MappedBuffer::device() {
    return device_;
}

void* MappedBuffer::host() {
    return host_;
}

size_t MappedBuffer::size() const {
    return size_;
}

void MappedBuffer::hostToDevice(cudaStream_t stream) {}

void MappedBuffer::deviceToHost(cudaStream_t stream) {}

std::unique_ptr<BaseBuffer> BufferFactory::createBuffer(BufferType type) {
    switch (type) {
        case BufferType::Device:
            return std::make_unique<DeviceBuffer>();           
        case BufferType::Discrete:
            return std::make_unique<DiscreteBuffer>();       
        case BufferType::Unified:
            return std::make_unique<UnifiedBuffer>();         
        case BufferType::Mapped:
            return std::make_unique<MappedBuffer>();           
        default:
            throw std::invalid_argument("Unknown buffer type"); 
    }
}

}  // namespace trtyolo
