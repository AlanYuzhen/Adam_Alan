#include "buff_detector.hpp"

#include <opencv2/core.hpp>

#include "network_deployment_interface.hpp"
#include "tools/logger.hpp"

namespace auto_buff
{
Buff_Detector::Buff_Detector(const std::string & config) : status_(LOSE), lose_(0)
{
  auto yaml = YAML::LoadFile(config);
  const std::string model_path = yaml["rune_model"]["path"].as<std::string>();
  // SZU RuneModel：OpenVINO 后端，Intel 集显用 GPU device
  rune_model_ = std::make_unique<RuneModel>(model_path, "async", "openvino", "GPU");
}

Buff_Detector::~Buff_Detector() = default;

void Buff_Detector::handle_lose()
{
  lose_++;
  if (lose_ >= LOSE_MAX) {
    status_ = LOSE;
    last_powerrune_ = std::nullopt;
  }
  status_ = TEM_LOSE;
}

std::optional<PowerRune> Buff_Detector::detect_24(cv::Mat & bgr_img)
{
  return detect(bgr_img);
}

std::optional<PowerRune> Buff_Detector::detect(cv::Mat & bgr_img)
{
  auto results = rune_model_->netProcess(bgr_img);
  if (results.empty()) {
    handle_lose();
    return std::nullopt;
  }

  // 取置信度最高的一个结果（单能量机关目标）
  const auto & r = results[0];

  // SZU 5 点: top, left, R, right, bottom；PnP 只需 4 角点（上/左/下/右）
  const cv::Point2f top(r.top), left(r.left), bottom(r.bottom), right(r.right);
  const cv::Point2f point_r(r.point_R);
  std::vector<cv::Point2f> kpt = {top, left, bottom, right};
  std::vector<FanBlade> fanblades;
  fanblades.emplace_back(FanBlade(kpt, point_r, _light));

  /// 生成PowerRune（r_center 直接使用模型输出的 R 字中心）
  PowerRune powerrune(fanblades, point_r, last_powerrune_);

  /// handle error
  if (powerrune.is_unsolve()) {
    handle_lose();
    return std::nullopt;
  }

  status_ = TRACK;
  lose_ = 0;
  std::optional<PowerRune> P;
  P.emplace(powerrune);
  last_powerrune_ = P;
  return P;
}

std::optional<PowerRune> Buff_Detector::detect_debug(cv::Mat & bgr_img, cv::Point2f)
{
  return detect(bgr_img);
}

}  // namespace auto_buff
