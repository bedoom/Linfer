
#ifndef __FASTREID_HPP
#define __FASTREID_HPP

#include <future>
#include <memory>
#include <string>
#include <vector>


namespace FastReID {

enum class Type : int {
    BOT = 0
};

struct ReIDResult {
    int id;
    std::vector<float> features;

    ReIDResult() = default;
    ReIDResult(int person_id, std::vector<float> features)
        : id(person_id),
            features(features) {}
};

struct Image {
  const void *bgrptr = nullptr;
  int width = 0, height = 0;

  Image() = default;
  Image(const void *bgrptr, int width, int height) : bgrptr(bgrptr), width(width), height(height) {}
};

typedef std::vector<ReIDResult> ReIDArray;

class Infer {
    public:
    virtual ReIDResult forward(const Image &image, void *stream = nullptr) = 0;
    virtual std::vector<ReIDResult> forwards(const std::vector<Image> &images,
                                         void *stream = nullptr) = 0;
};

std::shared_ptr<Infer> load(const std::string &engine_file, Type type);

const char *type_name(Type type);

};
#endif //__FASTREID_HPP
