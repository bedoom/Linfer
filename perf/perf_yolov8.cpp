#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

#include "trt_common/cpm.hpp"
#include "trt_common/infer.hpp"
#include "apps/yolov8/yolov8.hpp"


using namespace std;

static const char *cocolabels[] = {"person",        "bicycle",      "car",
                                   "motorcycle",    "airplane",     "bus",
                                   "train",         "truck",        "boat",
                                   "traffic light", "fire hydrant", "stop sign",
                                   "parking meter", "bench",        "bird",
                                   "cat",           "dog",          "horse",
                                   "sheep",         "cow",          "elephant",
                                   "bear",          "zebra",        "giraffe",
                                   "backpack",      "umbrella",     "handbag",
                                   "tie",           "suitcase",     "frisbee",
                                   "skis",          "snowboard",    "sports ball",
                                   "kite",          "baseball bat", "baseball glove",
                                   "skateboard",    "surfboard",    "tennis racket",
                                   "bottle",        "wine glass",   "cup",
                                   "fork",          "knife",        "spoon",
                                   "bowl",          "banana",       "apple",
                                   "sandwich",      "orange",       "broccoli",
                                   "carrot",        "hot dog",      "pizza",
                                   "donut",         "cake",         "chair",
                                   "couch",         "potted plant", "bed",
                                   "dining table",  "toilet",       "tv",
                                   "laptop",        "mouse",        "remote",
                                   "keyboard",      "cell phone",   "microwave",
                                   "oven",          "toaster",      "sink",
                                   "refrigerator",  "book",         "clock",
                                   "vase",          "scissors",     "teddy bear",
                                   "hair drier",    "toothbrush"};

static yolo::Image cvimg(const cv::Mat &image) { return yolo::Image(image.data, image.cols, image.rows); }

string extract_image_id(const string& path) {
    size_t last_slash = path.find_last_of("/\\");
    string filename = (last_slash == string::npos) ? path : path.substr(last_slash + 1);
    size_t dot = filename.find_last_of('.');
    return (dot == string::npos) ? filename : filename.substr(0, dot);
}


// 写入 result.txt
void flush_result(ofstream& out, const vector<string>& buffer) {
    for (const auto& line : buffer)
        out << line << "\n";
    out.flush();
}

void delay_batch_inference(const std::string& engine_file, const int& batch_size, yolo::Type type) {
  std::vector<cv::Mat> origin_images{cv::imread("inference/car.jpg"), cv::imread("inference/gril.jpg"),
                              cv::imread("inference/group.jpg")};
  auto yolo = yolo::load(engine_file, yolo::Type::V8);
  if (yolo == nullptr) return;

  std::vector<cv::Mat> images;
  for (int i = images.size(); i < batch_size; ++i)
    images.push_back(origin_images[i % 3]);

  std::vector<yolo::Image> yoloimages(images.size());
  std::transform(images.begin(), images.end(), yoloimages.begin(), cvimg);

  std::vector<yolo::BoxArray> batched_result;
  for(int i = 0; i < 10; ++i)
    batched_result = yolo->forwards(yoloimages);
  batched_result.clear();

  const int ntest = 400 / batch_size;
  auto start = std::chrono::steady_clock::now();
  for(int i  = 0; i < ntest; ++i)
    batched_result = yolo->forwards(yoloimages);

  std::chrono::duration<double> during = std::chrono::steady_clock::now() - start;
  double all_time = 1000.0 * during.count();
  float avg_time = all_time / ntest / images.size();
  printf("[Batch %d] Average time: %.2f ms, FPS: %.2f\n", batch_size, avg_time, 1000 / avg_time);
}

void precision_batch_inference(const std::string& img_folder, const string& result_path, const std::string& engine_path, int batch_size = 8, int flush_interval = 100) {
    auto yolo = yolo::load(engine_path, yolo::Type::V8);
    if (!yolo) {
        cerr << "Failed to load model: " << engine_path << endl;
        return;
    }

    vector<string> all_paths;
    cv::glob(img_folder, all_paths, true);  // 遍历图像路径（递归）
    cout << "Load " << all_paths.size() << " images." << endl;

    ofstream outfile(result_path);
    vector<string> result_buffer;

    for (size_t i = 0; i < all_paths.size(); i += batch_size) {
        vector<cv::Mat> images;
        vector<string> image_ids;

        for (int j = 0; j < batch_size && i + j < all_paths.size(); ++j) {
            const string& path = all_paths[i + j];
            cv::Mat img = cv::imread(path);
            if (img.empty()) continue;

            images.push_back(img);
            image_ids.push_back(extract_image_id(path));
        }

        if (images.empty()) continue;

        vector<yolo::Image> yolo_images(images.size());
        transform(images.begin(), images.end(), yolo_images.begin(), cvimg);
        vector<yolo::BoxArray> results = yolo->forwards(yolo_images);

        for (size_t k = 0; k < results.size(); ++k) {
            const string& id = image_ids[k];
            for (const auto& box : results[k]) {
                if (box.class_label != 0) continue;

                float x = box.left;
                float y = box.top;
                float w = box.right - box.left;
                float h = box.bottom - box.top;

                char line[200];
                snprintf(line, sizeof(line), "%s,%d,%.4f,%.2f,%.2f,%.2f,%.2f",
                         id.c_str(), box.class_label, box.confidence, x, y, w, h);
                result_buffer.push_back(line);
            }
        }

        if ((i + batch_size) % flush_interval == 0) {
            flush_result(outfile, result_buffer);
            result_buffer.clear();
            cout << "Processed " << min(i + batch_size, all_paths.size()) << " images." << endl;
        }
    }

    if (!result_buffer.empty()) {
        flush_result(outfile, result_buffer);
    }

    outfile.close();
    cout << "All done. Results saved to result.txt" << endl;
}
