
#ifndef FASTREID_HPP
#define FASTREID_HPP


#include <vector>
#include <memory>
#include <string>
#include <opencv2/opencv.hpp>
#include "trt_common/trt_tensor.hpp"


namespace FastReID {

    using namespace std;

    class ReID{
    public:
        virtual bool reid(const cv::Mat& image, vector<float32_t>& out) = 0;
    };

    shared_ptr<ReID> create_reid(const string& engine_file, int gpuid = 0);
}


#endif //FASTREID_HPP
