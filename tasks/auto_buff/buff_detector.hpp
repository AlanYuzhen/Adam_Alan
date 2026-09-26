#ifndef AUTO_BUFF__TRACK_HPP
#define AUTO_BUFF__TRACK_HPP

#include <yaml-cpp/yaml.h>

#include <deque>
#include <memory>
#include <optional>

#include "buff_type.hpp"
#include "tools/img_tools.hpp"
const int LOSE_MAX = 20;  // 丢失的阙值

// SZU RuneModel 在全局命名空间（network_deployment_interface.hpp）
class RuneModel;

namespace auto_buff
{

class Buff_Detector
{
public:
  explicit Buff_Detector(const std::string & config);
  ~Buff_Detector();

  std::optional<PowerRune> detect_24(cv::Mat & bgr_img);

  std::optional<PowerRune> detect(cv::Mat & bgr_img);

  std::optional<PowerRune> detect_debug(cv::Mat & bgr_img, cv::Point2f v);

private:
  void handle_lose();

  std::unique_ptr<RuneModel> rune_model_;
  Track_status status_;
  int lose_;  // 丢失的次数
  std::optional<PowerRune> last_powerrune_ = std::nullopt;
};
}  // namespace auto_buff
#endif  // DETECTOR_HPP
