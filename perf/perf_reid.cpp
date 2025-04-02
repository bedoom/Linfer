#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include "trt_common/cpm.hpp"
#include "trt_common/infer.hpp"
#include "apps/fastreid/fastreid.hpp"

using json = nlohmann::json;
using namespace std;

static FastReID::Image cvimg(const cv::Mat &image) { return FastReID::Image(image.data, image.cols, image.rows); }

#if 0
void performance_reid(const string& engine_file, const int &max_infer_batch){
    // int max_infer_batch = 8;
    int batch = max_infer_batch;

    auto image = cv::imread("imgs/000000.jpg");
    std::vector<cv::Mat> images(batch, image);

    cpm::Instance<FastReID::ReIDResult, FastReID::Image, FastReID::Infer> cpmi;
    bool ok = cpmi.start([=] { return FastReID::load(engine_file, FastReID::Type::BOT); },
                       max_infer_batch);

    if(!ok) {
        printf("cpmi 初始化失败\n");
        return;
    }
    std::vector<FastReID::Image> yoloimages(images.size());
    std::transform(images.begin(), images.end(), yoloimages.begin(), cvimg);

    // warmup
    for(int i = 0; i < 10; ++i) 
        cpmi.commits(yoloimages).back().get();

    trt::Timer timer;
    string batch_str;
    // 测试 400 张图片
    const int ntest = 400 / batch;
    timer.start();
    for(int i  = 0; i < ntest; ++i)
        cpmi.commits(yoloimages).back().get();
    batch_str = "BATCH"+to_string(batch);
    timer.stop(batch_str.c_str());
}
#endif

#if 1
void extract_images(cpm::Instance<FastReID::ReIDResult, FastReID::Image, FastReID::Infer>& predictor, const string& path, const string save_path) {
    vector<cv::String> files_;
    cv::glob(path+"/*.jpg", files_, true);
    vector<string> files(files_.begin(), files_.end());

    // map<string, vector<vector<float> > > reid_features;
    map<string, vector<vector<float>>> reid_features;

    for(const auto& file : files){
        string filename = file.substr(file.find_last_of("/\\") + 1);
        size_t pos = filename.find("_");
        if (pos == string::npos) {
            printf("警告: 无法解析身份 ID, 文件: %s，跳过。\n", filename.c_str());
            continue;
        }

        string reid = filename.substr(0, pos);

        auto origin_image = cv::imread(file);
        cv::Mat image;
        cv::cvtColor(origin_image, image, cv::COLOR_BGR2RGB);

        // 提取 ReID 
        auto result = predictor.commit(cvimg(image)).get();
        reid_features[reid].push_back(result.features);
    }

    json result_json;
    for (const auto& item : reid_features) {
        json person_json;
        person_json["reid"] = item.first;
        person_json["features"] = item.second;
        result_json.push_back(person_json);
    }
        std::ofstream out_file(save_path);
    if (out_file.is_open()) {
        out_file << result_json.dump(4);
        out_file.close();
        printf("特征数据已保存到: %s\n", save_path.c_str());
    } else {
        printf("错误: 无法打开文件 %s\n", save_path.c_str());
    }

    return;
} 

void extract_and_save_json(const string& engine_file, const int& max_infer_batch) {
    cpm::Instance<FastReID::ReIDResult, FastReID::Image, FastReID::Infer> predictor;
    bool ok = predictor.start([=] { return FastReID::load(engine_file, FastReID::Type::BOT); },
                       max_infer_batch);

    if(!ok) {
        printf("cpmi 初始化失败\n");
        return;
    }

    string gallery_path = "/home/xk/Linfer/workspace/gallery";
    string query_path = "/home/xk/Linfer/workspace/query";
    string save_galary_path = "gallery_reid.json";
    string save_query_path = "query_reid.json";

    extract_images(predictor, gallery_path, save_galary_path);
    extract_images(predictor, query_path, save_query_path);




}

#endif

