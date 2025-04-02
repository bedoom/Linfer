#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include "trt_common/cpm.hpp"
#include "trt_common/infer.hpp"
#include "fastreid/fastreid.hpp"

using json = nlohmann::json;
using namespace std;

static FastReID::Image cvimg(const cv::Mat &image) { return FastReID::Image(image.data, image.cols, image.rows); }

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

void normalize(vector<float>& features) {
    float norm = 0.0f;
    for (size_t i = 0; i < features.size(); ++i) {
        norm += features[i] * features[i];
    }
    norm = sqrt(norm);
    
    if (norm > 0) {
        for (size_t i = 0; i < features.size(); ++i) {
            features[i] /= norm;
        }
    }
}

float cosine_similarity(const vector<float>& a, const vector<float>& b) {
    // Normalize features before calculating cosine similarity
    vector<float> a_normalized = a;
    vector<float> b_normalized = b;
    normalize(a_normalized);
    normalize(b_normalized);

    float dot = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        dot += a_normalized[i] * b_normalized[i];
    }
    return dot;
}

#if 1
void extract_and_save_gallery(const string& engine_file, const string& gallery_path, const string& save_path, const int& max_infer_batch) {
    cpm::Instance<FastReID::ReIDResult, FastReID::Image, FastReID::Infer> predictor;
    bool ok = predictor.start([=] { return FastReID::load(engine_file, FastReID::Type::BOT); },
                       max_infer_batch);

    if(!ok) {
        printf("cpmi 初始化失败\n");
        return;
    }

    vector<cv::String> files_;
    files_.reserve(200);
    cv::glob(gallery_path+"/*.jpg", files_, true);
    vector<string> files(files_.begin(), files_.end());

    // 处理时间统计
    double total_time_ms = 0;
    int processed_images = 0;

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

}

bool load_gallery_features(const string& json_path, map<string, vector<vector<float> > >& gallery_reid_features) {
    ifstream in_file(json_path);
    if (!in_file.is_open()) {
        cerr << "Error: Unable to open JSON file: " << json_path << endl;
        return false;
    }

    json data;
    in_file >> data;
    in_file.close();

    for (const auto& entry : data) {
        string reid = entry["reid"];
        vector<vector<float> > features = entry["features"];
        gallery_reid_features[reid] = features;
    }

    cout << "Loaded " << gallery_reid_features.size() << " identities from gallery.\n";
    return true;
}

// Load query images and perform ReID
void load_gallery_and_reid(const string& engine_file, const string& json_path, const string& query_path, const int& max_infer_batch) {
    // 1. Load gallery features using Eigen vectors
    map<string, vector<vector<float> > > gallery_reid_features;
    if (!load_gallery_features(json_path, gallery_reid_features)) {
        return;
    }

    // 2️. Load ReID model
    cpm::Instance<FastReID::ReIDResult, FastReID::Image, FastReID::Infer> predictor;
    bool ok = predictor.start([=] { return FastReID::load(engine_file, FastReID::Type::BOT); },
                       max_infer_batch);

    if(!ok) {
        printf("cpmi 初始化失败\n");
        return;
    }

    // 3️. Load query images
    vector<cv::String> query_files;
    cv::glob(query_path + "/*.jpg", query_files, true);

    if (query_files.empty()) {
        cerr << "3. Error: No query images found in " << query_path << endl;
        return;
    }

    cout << "Processing " << query_files.size() << " query images...\n";

    for (const auto& file : query_files) {
        cv::Mat origin_image = cv::imread(file);
        cv::Mat image;
        cv::cvtColor(origin_image, image, cv::COLOR_BGR2RGB);

        if (image.empty()) {
            cerr << "Warning: Unable to read image: " << file << endl;
            continue;
        }

        // 4️. Extract features using FastReID
        auto res = predictor.commit(cvimg(image)).get();

        // 5️. Find the best match in the gallery
        priority_queue<pair<float, string>, vector<pair<float, string>>, greater<pair<float, string>>> top_matches;

        for (const auto& [gallery_reid, feature_list] : gallery_reid_features) {
            for (const auto& gallery_feature : feature_list) {
                float similarity = cosine_similarity(res.features, gallery_feature);
                top_matches.emplace(similarity, gallery_reid);
                if (top_matches.size() > 3) {
                    top_matches.pop(); // Remove the lowest-ranked match
                }
            }
        }

        // 6️. Output result
        vector<pair<float, string>> best_matches;
        while (!top_matches.empty()) {
            best_matches.emplace_back(top_matches.top());
            top_matches.pop();
        }
        reverse(best_matches.begin(), best_matches.end());

        string filename = file.substr(file.find_last_of("/\\") + 1);
        cout << "Query Image: " << filename << " -> Top 3 Matches:" << endl;
        for (size_t i = 0; i < best_matches.size(); ++i) 
            cout << "   " << i + 1 << ". " << best_matches[i].second
                 << " (Similarity: " << best_matches[i].first << ")" << endl;
    }
}

#endif

