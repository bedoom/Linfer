#include "trt_common/ilogger.hpp"
#include "trt_common/cpm.hpp"
#include "trt_common/infer.hpp"
#include "yolov8/yolov8.hpp"
#include <opencv2/opencv.hpp>
#include "bytetrack/BYTETracker.h"
#include <cstdio>

using namespace std;

static yolo::Image cvimg(const cv::Mat &image) { return yolo::Image(image.data, image.cols, image.rows); }

template<typename Cond>
static vector<Object> det2tracks(const yolo::BoxArray& array, const Cond& cond){
    vector<Object> outputs;
    for(int i = 0; i < array.size(); ++i){
        auto& abox = array[i];

        if(!cond(abox)) continue;

        Object obox;
        obox.prob = abox.confidence;
        obox.label = abox.class_label;
        obox.rect[0] = abox.left;
        obox.rect[1] = abox.top;
        obox.rect[2] = abox.right - abox.left;
        obox.rect[3] = abox.bottom - abox.top;
        outputs.emplace_back(obox);
    }
    return outputs;
}


void inference_bytetrack(const string& engine_file, int gpuid, yolo::Type type, const string& video_file){
    cpm::Instance<yolo::BoxArray, yolo::Image, yolo::Infer> cpmi;
    bool ok = cpmi.start([=] { return yolo::load(engine_file, yolo::Type::V8); },
                       1);
    if(ok == false){
        INFOE("Engine is nullptr");
        return;
    }

    vector<cv::String> files_;
    files_.reserve(100);
    cv::glob("imgs/c1/*.jpg", files_, true);
    vector<string> files(files_.begin(), files_.end());

    vector<cv::Mat> images;
    for(const auto& file : files){
        auto image = cv::imread(file);
        images.emplace_back(image);
    }
    int width = images[0].cols;
    int height = images[0].rows;
    int fps = 10;  // 假设帧率为10
    double all_time = 0;

    BYTETracker tracker;
    cv::Mat image;
    cv::Mat prev_image;
    tracker.config().set_initiate_state({0.1,  0.1,  0.1,  0.1,
                                                0.2,  0.2,  1,    0.2}
                                        ).set_per_frame_motion({0.1,  0.1,  0.1,  0.1,
                                                                       0.2,  0.2,  1,    0.2}
                                        ).set_max_time_lost(150);

    cv::VideoWriter writer("videos/res_mot.mp4", cv::VideoWriter::fourcc('M', 'P', 'E', 'G'), fps, cv::Size(width, height));
    auto cond = [](const yolo::Box& b){return b.class_label == 0;};

    shared_future<yolo::BoxArray> prev_fut;
    int t = 0;
    #if 0
    while(cap.read(image)){
        t++;
        /// 高性能的关键：
        /// 先缓存一帧图像，使得读图预处理和推理后处理的时序图有重叠
        if(prev_fut.valid()){
            const auto& boxes = prev_fut.get();
            auto tracks = tracker.update(det2tracks(boxes, cond));
            for(auto& track : tracks){

                vector<float> tlwh = track.tlwh;
                // 通过宽高比和面积过滤掉
                bool vertical = tlwh[2] / tlwh[3] > 1.6;
                if (tlwh[2] * tlwh[3] > 20 && !vertical)
                {
                    auto s = tracker.get_color(track.track_id);
                    putText(prev_image, cv::format("%d", track.track_id), cv::Point(tlwh[0], tlwh[1] - 10),
                            0, 2, cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
                    rectangle(prev_image, cv::Rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]),
                              cv::Scalar(get<0>(s), get<1>(s), get<2>(s)), 3);
                }
            }
            writer.write(prev_image);
            printf("process.\n");
        }

        image.copyTo(prev_image);
        prev_fut = engine->commit(image);

    }
    #else
    for(int i = 0; i < images.size(); i++) {
        image = images[i];
        t++;
        /// 高性能的关键：
        /// 先缓存一帧图像，使得读图预处理和推理后处理的时序图有重叠
        if(prev_fut.valid()){
            const auto& boxes = prev_fut.get();
            auto tracks = tracker.update(det2tracks(boxes, cond));
            for(auto& track : tracks){

                vector<float> tlwh = track.tlwh;
                // 通过宽高比和面积过滤掉
                bool vertical = tlwh[2] / tlwh[3] > 1.6;
                if (tlwh[2] * tlwh[3] > 20 && !vertical)
                {
                    auto s = tracker.get_color(track.track_id);
                    putText(prev_image, cv::format("%d", track.track_id), cv::Point(tlwh[0], tlwh[1] - 10),
                            0, 2, cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
                    rectangle(prev_image, cv::Rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]),
                              cv::Scalar(get<0>(s), get<1>(s), get<2>(s)), 3);
                }
            }
            writer.write(prev_image);
            printf("process.\n");
        }

        image.copyTo(prev_image);
        prev_fut = cpmi.commit(cvimg(image));

    }
    #endif
    writer.release();
    printf("Done.\n");
}

