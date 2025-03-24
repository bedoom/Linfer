
#include "fastreid.hpp"

#include <memory>
#include "trt_common/trt_infer.hpp"
#include "trt_common/ilogger.hpp"
#include "trt_common/preprocess_kernel.cuh"
#include "trt_common/cuda_tools.hpp"


namespace FastReID{

    using namespace std;

    struct AffineMatrix{
        float i2d[6];       // image to dst(network), 2x3 matrix
        float d2i[6];       // dst to image, 2x3 matrix

        void compute(const cv::Size& from, const cv::Size& to){
            float scale_x = to.width / (float)from.width;
            float scale_y = to.height / (float)from.height;
            float scale = std::min(scale_x, scale_y);
            
            i2d[0] = scale;  i2d[1] = 0;  i2d[2] = (-scale * from.width + to.width + scale - 1) * 0.5f;
            i2d[3] = 0;  i2d[4] = scale;  i2d[5] = (-scale * from.height + to.height + scale - 1) * 0.5f;
            cv::Mat m2x3_i2d(2, 3, CV_32F, i2d);
            cv::Mat m2x3_d2i(2, 3, CV_32F, d2i);
            cv::invertAffineTransform(m2x3_i2d, m2x3_d2i);
        }
    };

    class ReIDImpl : public ReID{
    public:
        ~ReIDImpl() = default;

        bool startup(const string& file, int gpuid){
            // ImageNet 归一化方式，提高 ReID 泛化能力
            float mean[3] = {0.485, 0.456, 0.406};
            float std[3]  = {0.229, 0.224, 0.225};
            normalize_ = CUDAKernel::Norm::mean_std(mean, std, 1/255.0f, CUDAKernel::ChannelType::Invert);

            // 加载引擎
            TRT::set_device(gpuid);
            model_ = TRT::load_infer(file);
            if(model_ == nullptr){
                INFOE("Load model failed: %s", file.c_str());
                return false;
            }

            model_->print();

            // 绑定输入输出
            input_ = model_->input();  // batch_sizex3x256x128
            output_ = model_->output();

            input_height_ = input_->shape(2);  // 256
            input_width_ = input_->shape(3);   // 128
            stream_ = model_->get_stream();
            gpu_ = gpuid;

            // 分配 input Tensor GPU 内存, 单帧处理 1x3x256x128
            input_->resize(1, 3, input_height_, input_width_).to_gpu();

            // 这里8个值的目的是保证 8 * sizeof(float) % 32 == 0
            // 先调整 shape 再 分配 affine_matrix Tensor GPU 内存
            affine_matrix_device_ = make_shared<TRT::Tensor>();
            affine_matrix_device_->set_stream(stream_);
            affine_matrix_device_->resize(1, 8).to_gpu();

            return true;
        }

        bool reid(const cv::Mat& image, vector<float32_t>& out) override{

            // 预处理
            AffineMatrix affineMatrix{};
            preprocess(image, affineMatrix);

            // 推理
            model_->forward(false);
            model_->synchronize();

            output_->to_cpu();
            auto* parray = output_->cpu<float32_t>();
            out.assign(parray, parray + output_->size(1));
            return true;
        }

        bool preprocess(const cv::Mat& image, AffineMatrix& affineMatrix){
            if(image.empty()){
                INFOE("Image is empty.");
                return false;
            }
            auto tensor = input_;
            if(tensor == nullptr){
                INFOE("Input Tensor is empty.");
                return false;
            }
            TRT::CUStream preprocess_stream = tensor->get_stream();

            cv::Size input_size(input_width_, input_height_);
            affineMatrix.compute(image.size(), input_size);
            tensor->resize(1, 3, input_height_, input_width_);
            size_t size_image = image.cols * image.rows * 3;
            // 对齐32字节
            size_t size_matrix = iLogger::upbound(sizeof(affineMatrix.d2i), 32);
            auto workspace = tensor->get_workspace();
            auto* gpu_workspace           = (uint8_t*)workspace->gpu(size_matrix + size_image);
            auto* affine_matrix_device    = (float*)gpu_workspace;
            uint8_t* image_device         = size_matrix + gpu_workspace;

            auto* cpu_workspace           = (uint8_t*)workspace->cpu(size_matrix + size_image);
            auto* affine_matrix_host      = (float*)cpu_workspace;
            uint8_t* image_host           = size_matrix + cpu_workspace;

            // speed up
            memcpy(image_host, image.data, size_image);
            memcpy(affine_matrix_host, affineMatrix.d2i, sizeof(affineMatrix.d2i));
            checkCudaRuntime(cudaMemcpyAsync(image_device, image_host, size_image, cudaMemcpyHostToDevice, preprocess_stream));
            checkCudaRuntime(cudaMemcpyAsync(affine_matrix_device, affine_matrix_host, sizeof(affineMatrix.d2i), cudaMemcpyHostToDevice, preprocess_stream));

            CUDAKernel::warp_affine_bilinear_and_normalize_plane(
                    image_device, image.cols * 3, image.cols, image.rows,
                    tensor->gpu<float>(), input_width_, input_height_,
                    affine_matrix_device, 114,
                    normalize_, preprocess_stream
            );

            return true;
        }

    private:
        int input_width_ = 0;
        int input_height_ = 0;
        int gpu_ = 0;
        CUDAKernel::Norm normalize_;
        TRT::CUStream stream_ = nullptr;
        shared_ptr<TRT::Infer> model_;
        shared_ptr<TRT::Tensor> input_;
        shared_ptr<TRT::Tensor> output_;
        shared_ptr<TRT::Tensor> affine_matrix_device_;
        shared_ptr<TRT::Tensor> invert_affine_matrix_device_;
    };

    shared_ptr<ReID> create_reid(const string& engine_file, int gpuid){
        shared_ptr<ReIDImpl> instance(new ReIDImpl{});
        if(!instance->startup(engine_file, gpuid)) instance.reset();
        return instance;
    }

}
