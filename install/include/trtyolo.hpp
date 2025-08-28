#pragma once

#ifdef _MSC_VER
#define TRTYOLOAPI __declspec(dllexport)
#else
#define TRTYOLOAPI __attribute__((visibility("default")))
#endif

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace trtyolo {

class BaseModel; 

struct TRTYOLOAPI Image {
    void*  ptr;         
    int    width  = 0;  
    int    height = 0;  
    size_t pitch  = 0;  

    Image(void* data, int width, int height);

    Image(void* data, int width, int height, size_t pitch);

};

struct TRTYOLOAPI Box {
    float left;    
    float top;     
    float right;   
    float bottom;  

    Box(float left, float top, float right, float bottom)
        : left(left), top(top), right(right), bottom(bottom) {}
};

struct TRTYOLOAPI BaseRes {
    int                num = 0;  
    std::vector<int>   classes;  
    std::vector<float> scores;   

    BaseRes() = default;
    BaseRes(int num, const std::vector<int>& classes, const std::vector<float>& scores)
        : num(num), classes(classes), scores(scores) {}
};

struct TRTYOLOAPI DetectRes : public BaseRes {
    std::vector<Box> boxes;  

    DetectRes() = default;

    DetectRes(int num, const std::vector<int>& classes, const std::vector<float>& scores, const std::vector<Box>& boxes)
        : BaseRes(num, classes, scores), boxes(boxes) {}

};

class TRTYOLOAPI InferOption {
public:
    InferOption();
    ~InferOption();

    void setDeviceId(int id);
    void enableCudaMem();
    void enableManagedMemory();
    void enablePerformanceReport();
    void enableSwapRB();
    void setBorderValue(float border_value);

    void setNormalizeParams(const std::vector<float>& mean, const std::vector<float>& std);

    void setInputDimensions(int width, int height);

private:
    class Impl;                   
    std::unique_ptr<Impl> impl_;  
    friend class trtyolo::BaseModel;
};

class TRTYOLOAPI BaseModel {
public:
    
    BaseModel();
    ~BaseModel();

    explicit BaseModel(const std::string& trt_engine_file, const InferOption& infer_option);
    int batch() const;

    std::tuple<std::string, std::string, std::string> performanceReport();

protected:
    class Impl;                   
    std::unique_ptr<Impl> impl_;  
};

class TRTYOLOAPI DetectModel : public BaseModel {
public:
    
    DetectModel();
    ~DetectModel();

    explicit DetectModel(const std::string& trt_engine_file, const InferOption& infer_option);

    std::unique_ptr<DetectModel> clone() const;

    DetectRes predict(const Image& image);
    std::vector<DetectRes> predict(const std::vector<Image>& images);
};


}  