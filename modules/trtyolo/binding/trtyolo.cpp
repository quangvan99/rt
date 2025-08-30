#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>
#include "infer/trtyolo.hpp"

namespace py = pybind11;


void binding_result_module(py::module& m) {
    m.doc() = "Result module of TensorRT-YOLO, including Mask, KeyPoint, Box, and other result-related classes.";

    py::class_<trtyolo::Box>(m, "Box", "A class representing a bounding box with (left, top, right, bottom) coordinates.")
        .def_readwrite("left", &trtyolo::Box::left, "The left coordinate of the bounding box.")
        .def_readwrite("top", &trtyolo::Box::top, "The top coordinate of the bounding box.")
        .def_readwrite("right", &trtyolo::Box::right, "The right coordinate of the bounding box.")
        .def_readwrite("bottom", &trtyolo::Box::bottom, "The bottom coordinate of the bounding box.")
        .def("__str__", [](const trtyolo::Box& box) {
            std::ostringstream oss;
            oss << box;
            return oss.str();
        });


    py::class_<trtyolo::DetectRes>(m, "DetectRes", "A class representing detection results, including the number of detections, classes, scores, and bounding boxes.")
        .def_readwrite("num", &trtyolo::DetectRes::num, "The number of detection results.")
        .def_readwrite("classes", &trtyolo::DetectRes::classes, "List of class IDs for detected objects.")
        .def_readwrite("scores", &trtyolo::DetectRes::scores, "List of confidence scores for each detection.")
        .def_readwrite("boxes", &trtyolo::DetectRes::boxes, "List of bounding boxes for detected objects.")
        .def_readwrite("out", &trtyolo::DetectRes::out, "List of output.")
        .def("to_numpy", [](const trtyolo::DetectRes& res) {
            py::array_t<float> np_array({res.out.size()});
            auto                 ptr = np_array.mutable_data();
            std::copy(res.out.begin(), res.out.end(), ptr);
            return np_array;
        })
        .def("__str__", [](const trtyolo::DetectRes& res) {
            std::ostringstream oss;
            oss << res;
            return oss.str();
        });
}

void binding_option_module(py::module& m) {
    m.doc() = "Option module of TensorRT-YOLO, providing options to configure inference settings.";

    py::class_<trtyolo::InferOption>(m, "InferOption", "A class to configure inference options, including device settings, memory options, and image preprocessing.")
        .def(py::init<>())
        .def("set_device_id", &trtyolo::InferOption::setDeviceId, "Set the device ID (GPU) for inference.")
        .def("enable_cuda_memory", &trtyolo::InferOption::enableCudaMem, "Inference data already in CUDA memory.")
        .def("enable_managed_memory", &trtyolo::InferOption::enableManagedMemory, "Enable managed memory for inference.")
        .def("enable_performance_report", &trtyolo::InferOption::enablePerformanceReport, "Enable performance report for inference.")
        .def("enable_swap_rb", &trtyolo::InferOption::enableSwapRB, "Enable RGB-to-BGR swap for image input.")
        .def("set_border_value", &trtyolo::InferOption::setBorderValue, "Set border value for image resizing (used for padding).")
        .def("set_normalize_params", &trtyolo::InferOption::setNormalizeParams, "Set normalization parameters for image preprocessing.")
        .def("set_input_dimensions", &trtyolo::InferOption::setInputDimensions, "Set the input dimensions (height, width) for the model.");
}

trtyolo::Image PyArray2Image(py::array& pyarray) {
    if (pyarray.ndim() != 3) {
        throw std::invalid_argument("Require rank of array to be 3 with HWC format while converting it to trtyolo::Image.");
    }
    py::buffer_info buf_info = pyarray.request();
    int             height   = buf_info.shape[0];
    int             width    = buf_info.shape[1];
    void*           data     = buf_info.ptr;
    if (buf_info.format != py::format_descriptor<uint8_t>::format()) {
        throw std::invalid_argument("Expected array data type is uint8.");
    }
    return trtyolo::Image(data, width, height);
}

template <typename ModelType>
void bind_model(py::module& m, const std::string& model_name) {
    py::class_<ModelType, std::unique_ptr<ModelType>>(m, model_name.c_str(), "A model class for performing inference tasks.")
        .def(py::init<const std::string&, const trtyolo::InferOption&>(), "Initialize the model with a model file and inference options.")
        .def("clone", &ModelType::clone, "Clone the model instance.")
        .def(
            "predict", [](ModelType& self, py::array& input) {
                return self.predict(PyArray2Image(input));
            },
            "Predict the result from a single image (HWC format numpy array).")
        .def("predict", [](ModelType& self, std::vector<py::array>& inputs) {
                std::vector<trtyolo::Image> images;
                std::transform(inputs.begin(), inputs.end(), std::back_inserter(images), [](py::array& input) {
                    return PyArray2Image(input);
                });
                return self.predict(images); }, "Predict the results from a list of images (list of HWC format numpy arrays).")
        .def("batch", &ModelType::batch, "Get the batch size of the model.");
}

void binding_model_module(py::module& m) {
    m.doc() = "Python bindings for model.hpp, including classes like ClassifyModel, DetectModel, OBBModel, etc.";

    bind_model<trtyolo::DetectModel>(m, "DetectModel");
}

PYBIND11_MODULE(py_trtyolo, m) {
    m.doc() = "TensorRT-YOLO Python bindings using Pybind11";

    py::module result_module = m.def_submodule("result", "Result module of TensorRT-YOLO.");
    binding_result_module(result_module);

    py::module option_module = m.def_submodule("option", "Option module of TensorRT-YOLO.");
    binding_option_module(option_module);

    py::module model_module = m.def_submodule("model", "Model module of TensorRT-YOLO.");
    binding_model_module(model_module);
}
