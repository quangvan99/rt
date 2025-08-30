#pragma once

#include <cuda_runtime.h>

#include <cassert>
#include <chrono>
#include <iostream>
#include <numeric>
#include <optional>
#include <vector>

namespace trtyolo {

inline void checkCudaError(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA Failure at " << file << ":" << line << ": "
                  << cudaGetErrorString(code) << std::endl;
        exit(EXIT_FAILURE);
    }
}

#define CHECK(code) checkCudaError((code), __FILE__, __LINE__)
#define MAKE_ERROR_MESSAGE(msg) (std::string("Error in ") + __FILE__ + ":" + std::to_string(__LINE__) + " (" + __FUNCTION__ + "): " + msg)

struct ProcessConfig {
    bool   swap_rb      = false;                                                  
    float  border_value = 114.0f;                                                 
    float3 alpha        = make_float3(1.0 / 255.0f, 1.0 / 255.0f, 1.0 / 255.0f);  
    float3 beta         = make_float3(0.0f, 0.0f, 0.0f);                          
};

struct InferConfig {
    int                 device_id                 = 0;      
    bool                cuda_mem                  = false;  
    bool                enable_managed_memory     = false;  
    bool                enable_performance_report = false;  
    std::optional<int2> input_shape;                        
    ProcessConfig       config;                             
};


void ReadBinaryFromFile(const std::string& file, std::string* contents);
bool SupportsIntegratedZeroCopy(const int gpu_id);
float findPercentile(float percentile, std::vector<float> const& timings);
float findMedian(std::vector<float> const& timings);
struct PerformanceResult {
    float              min{0.f};    
    float              max{0.f};    
    float              mean{0.f};   
    float              median{0.f}; 
    std::vector<float> percentiles; 
};


PerformanceResult getPerformanceResult(std::vector<float> const& timings, std::vector<float> const& percentiles);


class TimerBase {
public:
    virtual void       start() {}  
    virtual void       stop() {}   
    std::vector<float> milliseconds() const noexcept {
        return mMs;
    }  
    void reset() noexcept {
        mMs.clear();
    }  
    float totalMilliseconds() const noexcept {
        return std::accumulate(mMs.begin(), mMs.end(), 0.0F);
    }  

protected:
    std::vector<float> mMs;  
};

class CpuTimer : public TimerBase {
public:
    void start() override;                                                      
    void stop() override;                                                       

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> mStart, mStop;  
}; 

class GpuTimer : public TimerBase {
public:
    explicit GpuTimer(cudaStream_t stream); 
    ~GpuTimer();                            

    void start() override;                  
    void stop() override;                   

private:
    cudaEvent_t  mStart, mStop;             
    cudaStream_t mStream;                   
};

}  // namespace trtyolo