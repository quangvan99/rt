#include <memory>
#include <opencv2/opencv.hpp>
#include "trtyolo.hpp"

void printDetRes(const trtyolo::DetectRes& res) {
    std::cout << "DetectRes(\n    num=" << res.num << ",\n    classes=[";
    for (const auto& c : res.classes) std::cout << c << ", ";
    std::cout << "],\n    scores=[";
    for (const auto& s : res.scores) std::cout << s << ", ";
    std::cout << "],\n    boxes=[\n";
    for (const auto& box : res.boxes) std::cout << "Box(left=" << box.left << ", top=" << box.top << ", right=" << box.right << ", bottom=" << box.bottom << ") ";
    std::cout << "    ]\n)";
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] 
                  << " <engine_file> <image_file>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string engine_file = argv[1];
    std::string image_file = argv[2];
    
    trtyolo::InferOption option;
    option.enableSwapRB();  

    auto detector = std::make_unique<trtyolo::DetectModel>(
        engine_file,       
        option                         
    );
    
    cv::Mat cv_image = cv::imread(image_file);
    if (cv_image.empty()) {
        throw std::runtime_error("Failed to load test image.");
    }
    
    trtyolo::Image input_image(
        cv_image.data,     
        cv_image.cols,     
        cv_image.rows     
    );
    
    trtyolo::DetectRes res = detector->predict(input_image);
    printDetRes(res);
    for (const auto& box : res.boxes){
        cv::rectangle(cv_image, cv::Point(box.left, box.top), cv::Point(box.right, box.bottom), cv::Scalar(0, 255, 0), 2);
    }

    cv::imwrite("result.jpg", cv_image);
}