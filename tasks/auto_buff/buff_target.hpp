#ifndef AUTO_BUFF__TARGET_HPP
#define AUTO_BUFF__TARGET_HPP

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "buff_algo/predictor/buff_predictor.hpp"
#include "buff_algo/tracker/buff_tracker.hpp"
#include "buff_detector.hpp"
#include "buff_type.hpp"
#include "tools/math_tools.hpp"

namespace auto_buff
{
/// Target 基类（合并 HW：内部使用 BuffTracker + BuffPredictor）

class Target
{
public:
  Target();
  virtual ~Target() = default;
  virtual void get_target(
    const std::optional<PowerRune> & p,
    std::chrono::steady_clock::time_point & timestamp) = 0;  // 纯虚函数

  virtual void predict(double dt) = 0;  // 纯虚函数

  // 返回 HW predictor 预测的世界系打击点（aimer 弹道迭代使用）
  Eigen::Vector3d point_buff2world(const Eigen::Vector3d & point_in_buff) const;

  bool is_unsolve() const;

  // 兼容旧接口：x[4]=yaw, x[5]=roll, x[6]=roll_velocity（调试/aimer 使用）
  Eigen::VectorXd ekf_x() const;

  double spd = 0;  // 调试用

protected:
  // HW 三件套（shared_ptr 以便 target 可拷贝，拷贝为浅拷贝共享状态）
  std::shared_ptr<buff_algo::BuffTracker> tracker_;
  std::shared_ptr<buff_algo::BuffPredictor> predictor_;

  buff_algo::BuffState state_;
  Eigen::Matrix3d R_gimbal2world_ = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t_camera2gimbal_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d aim_point_world_ = Eigen::Vector3d::Zero();

  bool unsolvable_ = true;
  int lost_cn_ = 0;
  bool first_in_ = true;
};

/// SmallTarget（小符：SMALL_BUFF 模式）

class SmallTarget : public Target
{
public:
  SmallTarget();  // 无配置构造（tracker 为空，不工作）
  explicit SmallTarget(const std::string & config_path);  // 读 hw_buff.config_dir

  void get_target(
    const std::optional<PowerRune> & p, std::chrono::steady_clock::time_point & timestamp) override;

  void predict(double dt) override;
};

/// BigTarget（大符：BIG_BUFF 模式，RANSAC 正弦拟合）

class BigTarget : public Target
{
public:
  BigTarget();
  explicit BigTarget(const std::string & config_path);

  void get_target(
    const std::optional<PowerRune> & p, std::chrono::steady_clock::time_point & timestamp) override;

  void predict(double dt) override;
};

}  // namespace auto_buff
#endif
