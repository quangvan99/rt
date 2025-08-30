#include <algorithm>
#include <cstring>

#include "backend.hpp"

namespace trtyolo {

TrtBackend::TrtBackend(const std::string& trt_engine_file, const InferConfig& infer_config) : infer_config(infer_config) {
    cudaSetDevice(infer_config.device_id); 
    CHECK(cudaStreamCreate(&stream));
    zero_copy_ = SupportsIntegratedZeroCopy(infer_config.device_id);
    manager_ = std::make_unique<TRTManager>();
    std::string engine_buffer;
    ReadBinaryFromFile(trt_engine_file, &engine_buffer);
    manager_->initialize(engine_buffer.data(), engine_buffer.size());
    getTensorInfo();
    initialize();
    if (!dynamic) captureCudaGraph();
}

std::unique_ptr<TrtBackend> TrtBackend::clone() {
    auto clone_backend          = std::make_unique<TrtBackend>();
    clone_backend->infer_config = infer_config;

    cudaSetDevice(infer_config.device_id);            
    CHECK(cudaStreamCreate(&clone_backend->stream));  

    clone_backend->zero_copy_ = zero_copy_;
    clone_backend->manager_ = manager_->clone();
    clone_backend->getTensorInfo();
    clone_backend->initialize();
    if (!clone_backend->dynamic) clone_backend->captureCudaGraph();

    return clone_backend;
}

TrtBackend::~TrtBackend() {
    std::vector<TensorInfo>().swap(tensor_infos);
    std::vector<AffineTransform>().swap(affine_transforms);
    if (!dynamic) cuda_graph_.destroy();
    CHECK(cudaStreamDestroy(stream));
}

void TrtBackend::getTensorInfo() {
    std::vector<TensorInfo>().swap(tensor_infos);
    buffer_type_     = infer_config.enable_managed_memory ? BufferType::Unified : (zero_copy_ ? BufferType::Mapped : BufferType::Discrete);
    auto num_tensors = manager_->getNbIOTensors();
    for (auto i = 0; i < num_tensors; ++i) {
        std::string name  = std::string(manager_->getIOTensorName(i));
        auto        shape = manager_->getTensorShape(name.c_str());
        auto        dtype = manager_->getTensorDataType(name.c_str());
        bool        input = (manager_->getTensorIOMode(name.c_str()) == nvinfer1::TensorIOMode::kINPUT);

        if (input) {
            dynamic = std::any_of(shape.d, shape.d + shape.nbDims, [](int val) { return val == -1; });
            if (dynamic) {
                shape     = manager_->getProfileShape(name.c_str(), 0, nvinfer1::OptProfileSelector::kMIN);
                min_shape = make_int4(shape.d[0], shape.d[1], shape.d[2], shape.d[3]);
                shape     = manager_->getProfileShape(name.c_str(), 0, nvinfer1::OptProfileSelector::kMAX);
            }
            max_shape = make_int4(shape.d[0], shape.d[1], shape.d[2], shape.d[3]);
        } else if (!input && dynamic) {
            shape.d[0] = max_shape.x;
        }
        tensor_infos.emplace_back(name, shape, dtype, input, input ? BufferType::Device : buffer_type_);
    }

    for (size_t i = 0; i < tensor_infos.size(); ++i) {
        auto& t = tensor_infos[i];

        std::cout << "Tensor[" << i << "] "
                << " | name=" << t.name
                << " | shape=" << t.shape.nbDims;

        // In chi tiết từng dimension
        std::cout << " [";
        for (int j = 0; j < t.shape.nbDims; ++j) {
            if (j > 0) std::cout << ",";
            std::cout << t.shape.d[j];
        }
        std::cout << "]" << std::endl;
    }
}

void TrtBackend::initialize() {
    std::vector<AffineTransform>().swap(affine_transforms);
    inputs_buffer_ = BufferFactory::createBuffer(buffer_type_);

    infer_size_ = max_shape.y * max_shape.w * max_shape.z;

    if (infer_config.input_shape.has_value()) {
        input_size_ = max_shape.y * infer_config.input_shape->y * infer_config.input_shape->x;
        affine_transforms.emplace_back(AffineTransform());
        affine_transforms.front().updateMatrix(
            infer_config.input_shape->y,
            infer_config.input_shape->x,
            max_shape.w,
            max_shape.z);
        inputs_buffer_->allocate(max_shape.x * input_size_); 
    } else {
        affine_transforms.resize(max_shape.x, AffineTransform());
        if (!dynamic) inputs_buffer_->allocate(max_shape.x * infer_size_);
    }
}

void TrtBackend::captureCudaGraph() {
    // Step 1: Pre-inference execution before graph capture
    {
        for (auto& tensor_info : tensor_infos) {
            manager_->setTensorAddress(tensor_info.name.c_str(), tensor_info.buffer->device());
        }

        if (!manager_->enqueueV3(stream)) {
            throw std::runtime_error("captureCudaGraph: EnqueueV3 failed before graph creation.");
        }
        CHECK(cudaStreamSynchronize(stream));
    }

    // Lambda: Calculate input size and device pointers
    auto calculate_input_size_and_device = [&](int idx, int input_width, int input_height) {
        size_t input_size   = input_height * input_width * max_shape.y;
        void*  input_device = static_cast<uint8_t*>(inputs_buffer_->device()) + idx * input_size;
        void*  infer_device = static_cast<float*>(tensor_infos.front().buffer->device()) + idx * infer_size_;
        return std::make_pair(input_device, infer_device);
    };

    // Lambda: Perform WarpAffine operation
    auto warp_affine = [&](bool multi, int input_width, int input_height) {
        if (multi) {
            // Multi-instance WarpAffine
            cudaMutliWarpAffine(
                inputs_buffer_->device(),
                input_width,
                input_height,
                input_width * max_shape.y,
                tensor_infos.front().buffer->device(),
                max_shape.w,
                max_shape.z,
                affine_transforms.front().matrix,
                infer_config.config,
                max_shape.x,
                stream);
        } else {
            // Single-instance WarpAffine
            for (int idx = 0; idx < max_shape.x; ++idx) {
                auto [input_device, infer_device] = calculate_input_size_and_device(idx, input_width, input_height);
                cudaWarpAffine(
                    input_device,
                    input_width,
                    input_height,
                    input_width * max_shape.y,
                    infer_device,
                    max_shape.w,
                    max_shape.z,
                    affine_transforms[idx].matrix,
                    infer_config.config,
                    stream);
            }
        }
    };

    // Step 2: Begin CUDA Graph Capture
    cuda_graph_.beginCapture(stream);

    // Step 3: Perform memory transfer and WarpAffine based on configuration
    if (infer_config.cuda_mem) {
        int input_width  = infer_config.input_shape ? infer_config.input_shape->y : max_shape.w;
        int input_height = infer_config.input_shape ? infer_config.input_shape->x : max_shape.z;
        warp_affine(false, input_width, input_height);
    } else {
        inputs_buffer_->hostToDevice(stream);
        // std::cout << "============>" << infer_config.input_shape.has_value() << std::endl;
        int input_width  = infer_config.input_shape ? infer_config.input_shape->y : max_shape.w;
        int input_height = infer_config.input_shape ? infer_config.input_shape->x : max_shape.z;
        warp_affine(infer_config.input_shape.has_value(), input_width, input_height);
    }

    // Step 4: Enqueue inference
    if (!manager_->enqueueV3(stream)) {
        throw std::runtime_error("captureCudaGraph: EnqueueV3 failed when graph creation.");
    }

    // Step 5: Copy output tensors to host if required
    for (auto& tensor_info : tensor_infos) {
        if (!tensor_info.input) {
            tensor_info.buffer->deviceToHost(stream);
        }
    }

    // Step 6: End CUDA Graph Capture
    cuda_graph_.endCapture(stream);

    // Step 7: Initialize CUDA Graph Nodes
    if (!(infer_config.input_shape.has_value() && !infer_config.cuda_mem)) {
        int num_nodes = max_shape.x + (infer_config.cuda_mem ? 0 : 1); 
        cuda_graph_.initializeNodes(num_nodes);
    }
}

void TrtBackend::staticInfer(const std::vector<Image>& inputs) {
    auto num = inputs.size();


    if (num < 1 || num > max_shape.x) {
        throw std::invalid_argument("Number of inputs out of range");
    }

    if (infer_config.input_shape.has_value()) {
        if (infer_config.cuda_mem) {
            for (int idx = 0; idx < num; ++idx) {
                auto infer_device_ptr = static_cast<float*>(tensor_infos.front().buffer->device()) + idx * infer_size_;

                void* kernelParams[] = {
                    (void*)&inputs[idx].ptr,
                    (void*)&inputs[idx].width,
                    (void*)&inputs[idx].height,
                    (void*)&inputs[idx].pitch,
                    (void*)&infer_device_ptr,
                    (void*)&max_shape.w,
                    (void*)&max_shape.z,
                    (void*)&affine_transforms.front().matrix[0],
                    (void*)&affine_transforms.front().matrix[1],
                    (void*)&infer_config.config};

                cuda_graph_.updateKernelNodeParams(idx, kernelParams);
            }
        } else {
            for (auto idx = 0; idx < num; ++idx) {
                std::memcpy(static_cast<uint8_t*>(inputs_buffer_->host()) + idx * input_size_, inputs[idx].ptr, input_size_);
            }
        }

    } else {
        if (!infer_config.cuda_mem) {
            int              total_size = 0;
            std::vector<int> input_sizes(num);


            for (int idx = 0; idx < num; ++idx) {
                input_sizes[idx]  = inputs[idx].height * inputs[idx].pitch;
                total_size       += input_sizes[idx];
            }


            inputs_buffer_->allocate(total_size);
            uint8_t* input_ptr = static_cast<uint8_t*>(inputs_buffer_->host());

            for (int idx = 0; idx < num; ++idx) {
                std::memcpy(input_ptr, inputs[idx].ptr, input_sizes[idx]);
                input_ptr += input_sizes[idx];
            }


            if (buffer_type_ == BufferType::Discrete) {
                cuda_graph_.updateMemcpyNodeParams(0, inputs_buffer_->host(), inputs_buffer_->device(), total_size);
            }
        }

        uint8_t* input_ptr = !infer_config.cuda_mem ? static_cast<uint8_t*>(inputs_buffer_->device()) : nullptr;
        for (int idx = 0; idx < num; ++idx) {
            affine_transforms[idx].updateMatrix(inputs[idx].width, inputs[idx].height, max_shape.w, max_shape.z);
            
            auto infer_device_ptr = static_cast<float*>(tensor_infos.front().buffer->device()) + idx * infer_size_;

            void* kernelParams[] = {
                infer_config.cuda_mem ? (void*)&inputs[idx].ptr : (void*)&input_ptr,
                (void*)&inputs[idx].width,
                (void*)&inputs[idx].height,
                (void*)&inputs[idx].pitch,
                (void*)&infer_device_ptr,
                (void*)&max_shape.w,
                (void*)&max_shape.z,
                (void*)&affine_transforms[idx].matrix[0],
                (void*)&affine_transforms[idx].matrix[1],
                (void*)&infer_config.config};
   
            int node_idx = (infer_config.cuda_mem || buffer_type_ != BufferType::Discrete) ? idx : idx + 1;
            cuda_graph_.updateKernelNodeParams(node_idx, kernelParams);

            if (!infer_config.cuda_mem) {
                input_ptr += inputs[idx].height * inputs[idx].pitch;
            }
        }
    }

    // Launch the CUDA graph
    cuda_graph_.launch(stream);
}

void TrtBackend::dynamicInfer(const std::vector<Image>& inputs) {
    auto num = inputs.size();

    if (num < min_shape.x || num > max_shape.x) {
        throw std::invalid_argument("Number of inputs out of range");
    }

    for (auto& tensor_info : tensor_infos) {
        tensor_info.shape.d[0] = num;
        tensor_info.update();
        manager_->setTensorAddress(tensor_info.name.c_str(), tensor_info.buffer->device());
        if (tensor_info.input) {
            manager_->setInputShape(tensor_info.name.c_str(), tensor_info.shape);
        }
    }

    if (infer_config.input_shape.has_value()) {

        if (!infer_config.cuda_mem) {
            for (int idx = 0; idx < num; ++idx) {
                std::memcpy(static_cast<uint8_t*>(inputs_buffer_->host()) + idx * input_size_, inputs[idx].ptr, input_size_);
            }
            inputs_buffer_->hostToDevice(stream);
        }

        for (int idx = 0; idx < num; ++idx) {
            cudaWarpAffine(
                infer_config.cuda_mem ? inputs[idx].ptr : static_cast<uint8_t*>(inputs_buffer_->device()) + idx * input_size_,
                inputs[idx].width,
                inputs[idx].height,
                inputs[idx].pitch,
                static_cast<float*>(tensor_infos.front().buffer->device()) + idx * infer_size_,
                max_shape.w,
                max_shape.z,
                affine_transforms.front().matrix,
                infer_config.config,
                stream);
        }
    } else {
        int              total_size = 0;
        std::vector<int> input_sizes(num);

        
        for (int idx = 0; idx < num; ++idx) {
            input_sizes[idx]  = inputs[idx].height * inputs[idx].pitch;
            total_size       += input_sizes[idx];
            affine_transforms[idx].updateMatrix(inputs[idx].width, inputs[idx].height, max_shape.w, max_shape.z);
        }

        
        if (!infer_config.cuda_mem) {
            
            inputs_buffer_->allocate(total_size);
            uint8_t* input_host = static_cast<uint8_t*>(inputs_buffer_->host());

            
            for (int idx = 0; idx < num; ++idx) {
                std::memcpy(input_host, inputs[idx].ptr, input_sizes[idx]);
                input_host += input_sizes[idx];
            }

            
            inputs_buffer_->hostToDevice(stream);

            
            uint8_t* input_device = static_cast<uint8_t*>(inputs_buffer_->device());
            for (int idx = 0; idx < num; ++idx) {
                cudaWarpAffine(
                    input_device,
                    inputs[idx].width,
                    inputs[idx].height,
                    inputs[idx].pitch,
                    static_cast<float*>(tensor_infos.front().buffer->device()) + idx * infer_size_,
                    max_shape.w,
                    max_shape.z,
                    affine_transforms[idx].matrix,
                    infer_config.config,
                    stream);
                input_device += input_sizes[idx];
            }
        } else {
            
            for (int idx = 0; idx < num; ++idx) {
                cudaWarpAffine(
                    inputs[idx].ptr,
                    inputs[idx].width,
                    inputs[idx].height,
                    inputs[idx].pitch,
                    static_cast<float*>(tensor_infos.front().buffer->device()) + idx * infer_size_,
                    max_shape.w,
                    max_shape.z,
                    affine_transforms[idx].matrix,
                    infer_config.config,
                    stream);
            }
        }
    }

    
    if (!manager_->enqueueV3(stream)) {
        throw std::runtime_error("Infer Error.");
    }

    
    for (auto& tensor_info : tensor_infos) {
        if (!tensor_info.input) {
            tensor_info.buffer->deviceToHost(stream);
        }
    }

    
    CHECK(cudaStreamSynchronize(stream));
}

void TrtBackend::infer(const std::vector<Image>& inputs) {
    if (dynamic) {
        dynamicInfer(inputs);
    } else {
        staticInfer(inputs);
    }
}

}  // namespace trtyolo