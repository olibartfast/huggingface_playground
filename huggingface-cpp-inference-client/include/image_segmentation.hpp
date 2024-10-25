#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include "huggingface_task.hpp"
#include <opencv2/opencv.hpp>

struct SegmentationResult {
    std::string label;
    double score;
    cv::Mat mask;
};

class ImageSegmentation : public HuggingFaceTask {
private:
    std::string imagePath;
    double maskThreshold;
    double overlapMaskAreaThreshold;
    std::string subtask;
    double threshold;
    int targetSize;
    bool resize;

protected:
    nlohmann::json preparePayload() const override;

public:
    ImageSegmentation(const std::string& endpoint, const std::string& authToken, 
                      const std::string& imagePath, 
                      double maskThreshold = 0, 
                      double overlapMaskAreaThreshold = 0, 
                      const std::string& subtask = "", 
                      double threshold = 0,
                      int targetSize = 1024,
                      bool resize = false);

    std::string execute();
};