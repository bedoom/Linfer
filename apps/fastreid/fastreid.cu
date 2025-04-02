#include "trt_common/infer.hpp"
#include "trt_common/preprocess_kernel.cuh"
#include "trt_common/cuda_tools.hpp"
#include "trt_common/ilogger.hpp"
#include "fastreid.hpp"

namespace FastReID {

using namespace std;

#define GPU_BLOCK_THREADS 512
#define checkRuntime(call)                                                                 \
  do {                                                                                     \
    auto ___call__ret_code__ = (call);                                                     \
    if (___call__ret_code__ != cudaSuccess) {                                              \
      INFO("CUDA Runtime error💥 %s # %s, code = %s [ %d ]", #call,                         \
           cudaGetErrorString(___call__ret_code__), cudaGetErrorName(___call__ret_code__), \
           ___call__ret_code__);                                                           \
      abort();                                                                             \
    }                                                                                      \
  } while (0)

#define checkKernel(...)                 \
  do {                                   \
    { (__VA_ARGS__); }                   \
    checkRuntime(cudaPeekAtLastError()); \
  } while (0)


static __host__ __device__ void affine_project(float *matrix, float x, float y, float *ox,
                                               float *oy) {
  *ox = matrix[0] * x + matrix[1] * y + matrix[2];
  *oy = matrix[3] * x + matrix[4] * y + matrix[5];
}

static __global__ void decode_kernel_common() {
    return;
}

static __global__ void decode_kernel_v8() {
    return;
}

static void decode_kernel_invoker() {
    return;
}

const char *type_name(Type type) {
  switch (type) {
    case Type::BOT:
      return "FastReID-BOT";
    default:
      return "Unknow";
  }
}

struct AffineMatrix {
    float i2d[6];  // image to dst(network), 2x3 matrix
    float d2i[6];  // dst to image, 2x3 matrix

    void compute(const std::tuple<int, int> &from, const std::tuple<int, int> &to) {
        float scale_x = get<0>(to) / (float)get<0>(from);
        float scale_y = get<1>(to) / (float)get<1>(from);
        float scale = std::min(scale_x, scale_y);
        i2d[0] = scale;
        i2d[1] = 0;
        i2d[2] = -scale * get<0>(from) * 0.5 + get<0>(to) * 0.5 + scale * 0.5 - 0.5;
        i2d[3] = 0;
        i2d[4] = scale;
        i2d[5] = -scale * get<1>(from) * 0.5 + get<1>(to) * 0.5 + scale * 0.5 - 0.5;

        double D = i2d[0] * i2d[4] - i2d[1] * i2d[3];
        D = D != 0. ? double(1.) / D : double(0.);
        double A11 = i2d[4] * D, A22 = i2d[0] * D, A12 = -i2d[1] * D, A21 = -i2d[3] * D;
        double b1 = -A11 * i2d[2] - A12 * i2d[5];
        double b2 = -A21 * i2d[2] - A22 * i2d[5];

        d2i[0] = A11;
        d2i[1] = A12;
        d2i[2] = b1;
        d2i[3] = A21;
        d2i[4] = A22;
        d2i[5] = b2;
    }
};

class InferImpl : public Infer {
 public:
  shared_ptr<trt::Infer> trt_;
  string engine_file_;
  Type type_;
  vector<shared_ptr<trt::Memory<unsigned char>>> preprocess_buffers_;
  trt::Memory<float> input_buffer_, feature_output_;
  int network_input_width_, network_input_height_;
  CUDAKernel::Norm normalize_;
  int feature_dim;
  bool isdynamic_model_ = false;

  virtual ~InferImpl() = default;

  void adjust_memory(int batch_size) {
    std::size_t input_numel = network_input_width_ * network_input_height_ * 3;
    input_buffer_.gpu(batch_size * input_numel);
    feature_output_.gpu(batch_size * feature_dim);
    feature_output_.cpu(batch_size * feature_dim);

    if ((int)preprocess_buffers_.size() < batch_size) {
      for (int i = preprocess_buffers_.size(); i < batch_size; ++i)
        preprocess_buffers_.push_back(make_shared<trt::Memory<unsigned char>>());
    }
  }

  void preprocess(int ibatch, const Image &image,
                  shared_ptr<trt::Memory<unsigned char>> preprocess_buffer, AffineMatrix &affine,
                  void *stream = nullptr) {
    
    affine.compute(make_tuple(image.width, image.height),
                   make_tuple(network_input_width_, network_input_height_));

    std::size_t input_numel = network_input_width_ * network_input_height_ * 3;
    float *input_device = input_buffer_.gpu() + ibatch * input_numel;
    std::size_t size_image = image.width * image.height * 3;
    std::size_t size_matrix = iLogger::upbound(sizeof(affine.d2i), 32);
    uint8_t *gpu_workspace = preprocess_buffer->gpu(size_matrix + size_image);
    float *affine_matrix_device = (float *)gpu_workspace;
    uint8_t *image_device = gpu_workspace + size_matrix;

    uint8_t *cpu_workspace = preprocess_buffer->cpu(size_matrix + size_image);
    float *affine_matrix_host = (float *)cpu_workspace;
    uint8_t *image_host = cpu_workspace + size_matrix;

    // speed up
    cudaStream_t stream_ = (cudaStream_t)stream;
    memcpy(image_host, image.bgrptr, size_image);
    memcpy(affine_matrix_host, affine.d2i, sizeof(affine.d2i));
    checkRuntime(
        cudaMemcpyAsync(image_device, image_host, size_image, cudaMemcpyHostToDevice, stream_));
    checkRuntime(cudaMemcpyAsync(affine_matrix_device, affine_matrix_host, sizeof(affine.d2i),
                                 cudaMemcpyHostToDevice, stream_));

    warp_affine_bilinear_and_normalize_plane(image_device, image.width * 3, image.width,
                                             image.height, input_device, network_input_width_,
                                             network_input_height_, affine_matrix_device, 114,
                                             normalize_, stream_);
  }

  bool load(const string &engine_file, Type type) {
    trt_ = trt::load(engine_file);
    if (trt_ == nullptr) return false;

    trt_->print();

    this->type_ = type;

    auto input_dim = trt_->static_dims(0);
    network_input_width_ = input_dim[3];
    network_input_height_ = input_dim[2];
    isdynamic_model_ = trt_->has_dynamic_dim();

    feature_dim = trt_->static_dims(1)[1];  // 特征向量维度
    if(type == Type::BOT) {
        // normalize_ = CUDAKernel::Norm::alpha_beta(1 / 255.0f, 0.0f, CUDAKernel::ChannelType::Invert);
        normalize_ = CUDAKernel::Norm::None();
    } else {
      INFO("Unsupport type %d", type);
    }
    return true;
  }

  virtual ReIDResult forward(const Image &image, void *stream = nullptr) override {
    auto output = forwards({image}, stream);
    if (output.empty()) return {};
    return output[0];
  }

  virtual vector<ReIDResult> forwards(const vector<Image> &images, void *stream = nullptr) override {
    int num_image = images.size();
    if (num_image == 0) return {};

    auto input_dims = trt_->static_dims(0);
    int infer_batch_size = input_dims[0];
    if (infer_batch_size != num_image) {
      if (isdynamic_model_) {
        infer_batch_size = num_image;
        input_dims[0] = num_image;
        if (!trt_->set_run_dims(0, input_dims)) return {};
      } else {
        INFO(
            "When using static shape model, number of images[%d] must be "
            "less than or equal to the maximum batch[%d].",
            num_image, infer_batch_size);
        return {};
      }
    }
    adjust_memory(infer_batch_size);

    vector<AffineMatrix> affine_matrixs(num_image);
    cudaStream_t stream_ = (cudaStream_t)stream;
    for (int i = 0; i < num_image; ++i)
      preprocess(i, images[i], preprocess_buffers_[i], affine_matrixs[i], stream);

    float *feature_output_device = feature_output_.gpu();
    vector<void *> bindings{input_buffer_.gpu(), feature_output_device};

    if (!trt_->forward(bindings, stream)) {
      INFO("Failed to tensorRT forward.");
      return {};
    }

    checkRuntime(cudaMemcpyAsync(feature_output_.cpu(), feature_output_.gpu(),
                                 feature_output_.gpu_bytes(), cudaMemcpyDeviceToHost, stream_));
    checkRuntime(cudaStreamSynchronize(stream_));

    vector<ReIDResult> arrout(num_image);
    int imemory = 0;
    for (int ib = 0; ib < num_image; ++ib) {
      float *features = feature_output_.cpu() + ib * feature_dim;
      arrout[ib].features.assign(features, features + feature_dim);
    }

    return arrout;
  }
};

Infer *loadraw(const std::string &engine_file, Type type) {
  InferImpl *impl = new InferImpl();
  if (!impl->load(engine_file, type)) {
    delete impl;
    impl = nullptr;
  }
  return impl;
}

shared_ptr<Infer> load(const string &engine_file, Type type) {
  return std::shared_ptr<InferImpl>(
      (InferImpl *)loadraw(engine_file, type));
}

}