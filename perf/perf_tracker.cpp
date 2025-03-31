#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>

#include "trt_common/ilogger.hpp"
#include "trt_common/cpm.hpp"
#include "trt_common/infer.hpp"
#include "apps/yolov8/yolov8.hpp"
#include "apps/bytetrack/BYTETracker.h"

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


void precision_bytetrack_inference(const std::string& img_folder, const string& result_path, const std::string& engine_path, yolo::Type type){
    cpm::Instance<yolo::BoxArray, yolo::Image, yolo::Infer> cpmi;
    bool ok = cpmi.start([=] { return yolo::load(engine_path, yolo::Type::V8); }, 1);
    if (!ok) {
        cerr << "Failed to load model: " << engine_path << endl;
        return;
    }

    vector<string> all_paths;
    cv::glob(img_folder, all_paths, true);  // 遍历图像路径（递归）
    cout << "Load " << all_paths.size() << " images." << endl;

    ofstream outfile(result_path);
    vector<string> result_buffer;

    BYTETracker tracker;
    tracker.config()
        .set_initiate_state({0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 1, 0.2})
        .set_per_frame_motion({0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 1, 0.2})
        .set_max_time_lost(30);

    auto cond = [](const yolo::Box& b){ return b.class_label == 0; };

    std::shared_future<yolo::BoxArray> prev_fut;
    cv::Mat image, prev_image;
    int frame_id = 0;

    // 创建结果文件（跟踪结果）
    std::ofstream result_file("track_results.txt");
    if (!result_file.is_open()) {
        INFOE("Failed to open result file.");
        return;
    }

    for(int i = 0; i < all_paths.size(); ++i) {
        image = cv::imread(all_paths[i]);

        if(prev_fut.valid()){
            const auto& boxes = prev_fut.get();
            auto tracks = tracker.update(det2tracks(boxes, cond));

            for(const auto& track : tracks){
                const auto& tlwh = track.tlwh;

                // ✅ 写入跟踪结果（MOT格式）：frame, id, x, y, w, h, score, class, visibility
                result_file << frame_id << "," << track.track_id << ","
                            << tlwh[0] << "," << tlwh[1] << "," << tlwh[2] << "," << tlwh[3] << ","
                            << track.score << "," << 0 << "," << 1.0 << "\n";
            }
            if(frame_id % 1000 == 0)
                printf("Processed frame %d\n", frame_id);
            frame_id++;
        }

        image.copyTo(prev_image);
        prev_fut = cpmi.commit(cvimg(image));
    }
    result_file.close();
    printf("Done. Tracking results saved to track_results.txt\n");
}
