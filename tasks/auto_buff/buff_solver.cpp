#include "buff_solver.hpp"
namespace auto_buff
{
cv::Matx33f Solver::rotation_matrix(double angle) const
{
  return cv::Matx33f(
    1, 0, 0, 0, std::cos(angle), -std::sin(angle), 0, std::sin(angle), std::cos(angle));
}

void Solver::compute_rotated_points(std::vector<std::vector<cv::Point3f>> & object_points)
{
  const std::vector<cv::Point3f> & base_points = object_points[0];
  for (int i = 1; i < 5; ++i) {
    double angle = i * THETA;
    cv::Matx33f R = rotation_matrix(angle);
    std::vector<cv::Point3f> rotated_points;
    for (const auto & point : base_points) {
      cv::Vec3f vec(point.x, point.y, point.z);
      cv::Vec3f rotated_vec = R * vec;
      rotated_points.emplace_back(rotated_vec[0], rotated_vec[1], rotated_vec[2]);
    }
    object_points[i] = rotated_points;
  }
}

Solver::Solver(const std::string & config_path) : R_gimbal2world_(Eigen::Matrix3d::Identity())
{
  auto yaml = YAML::LoadFile(config_path);

  auto R_gimbal2imubody_data = yaml["R_gimbal2imubody"].as<std::vector<double>>();
  auto R_camera2gimbal_data = yaml["R_camera2gimbal"].as<std::vector<double>>();
  auto t_camera2gimbal_data = yaml["t_camera2gimbal"].as<std::vector<double>>();
  R_gimbal2imubody_ = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(R_gimbal2imubody_data.data());
  R_camera2gimbal_ = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(R_camera2gimbal_data.data());
  t_camera2gimbal_ = Eigen::Matrix<double, 3, 1>(t_camera2gimbal_data.data());

  auto camera_matrix_data = yaml["camera_matrix"].as<std::vector<double>>();
  auto distort_coeffs_data = yaml["distort_coeffs"].as<std::vector<double>>();
  Eigen::Matrix<double, 3, 3, Eigen::RowMajor> camera_matrix(camera_matrix_data.data());
  Eigen::Matrix<double, 1, 5> distort_coeffs(distort_coeffs_data.data());
  cv::eigen2cv(camera_matrix, camera_matrix_);
  cv::eigen2cv(distort_coeffs, distort_coeffs_);

  // HW PnP 需要相机内参
  pnp_solver_.set_camera_matrix(camera_matrix_, distort_coeffs_);
}

Eigen::Matrix3d Solver::R_gimbal2world() const { return R_gimbal2world_; }

void Solver::set_R_gimbal2world(const Eigen::Quaterniond & q)
{
  Eigen::Matrix3d R_imubody2imuabs = q.toRotationMatrix();
  R_gimbal2world_ = R_gimbal2imubody_.transpose() * R_imubody2imuabs * R_gimbal2imubody_;
}

void Solver::solve(std::optional<PowerRune> & ps) const
{
  if (!ps.has_value()) return;
  PowerRune & p = ps.value();

  // 检测输出 4 角点顺序固定：上/左/下/右（t/l/b/r）
  const auto & pts = p.target().points;
  if (pts.size() < 4) return;

  buff_algo::BuffDetection det;
  det.t = {pts[0].x, pts[0].y};
  det.l = {pts[1].x, pts[1].y};
  det.b = {pts[2].x, pts[2].y};
  det.r = {pts[3].x, pts[3].y};
  det.confidence = 1.0f;

  auto poses = pnp_solver_.solve_pnp({det});
  if (poses.empty()) return;
  const auto & pose = poses[0];  // 相机系（前x左y上z）

  // 存相机系位姿 + 外参，供 target 的 HW tracker 使用
  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  T.block<3, 3>(0, 0) = pose.rotation.toRotationMatrix().cast<double>();
  T.block<3, 1>(0, 3) = pose.translation.cast<double>();
  p.pose_camera = T;
  p.R_gimbal2world = R_gimbal2world_;
  p.t_camera2gimbal = t_camera2gimbal_;

  // HW PnP 内部已做 cv_to_tf（=R_camera2gimbal），输出已在"前x左y上z"相机系，
  // 到云台系只需加平移，绝不能再乘 R_camera2gimbal
  Eigen::Vector3d xyz_in_camera = pose.translation.cast<double>();
  Eigen::Vector3d xyz_in_gimbal = xyz_in_camera + t_camera2gimbal_;

  // gimbal -> world
  p.xyz_in_world = R_gimbal2world_ * xyz_in_gimbal;
  p.ypd_in_world = tools::xyz2ypd(p.xyz_in_world);

  // R 字中心（扇叶面板中心向 z 负方向 BUFF_RADIUS=0.7m）
  Eigen::Vector3d R_center_cam =
    pose.rotation.toRotationMatrix().cast<double>() * Eigen::Vector3d(0, 0, -0.7) +
    xyz_in_camera;
  Eigen::Vector3d R_center_gimbal = R_center_cam + t_camera2gimbal_;
  p.blade_xyz_in_world = R_gimbal2world_ * R_center_gimbal;
  p.blade_ypd_in_world = tools::xyz2ypd(p.blade_xyz_in_world);

  // ypr_in_world：R_buff2world 的欧拉角
  Eigen::Matrix3d R_buff2gimbal = pose.rotation.toRotationMatrix().cast<double>();
  Eigen::Matrix3d R_buff2world = R_gimbal2world_ * R_buff2gimbal;
  p.ypr_in_world = tools::eulers(R_buff2world, 2, 1, 0);
}

// 调试用
cv::Point2f Solver::point_buff2pixel(cv::Point3f x)
{
  // buff坐标系(单位:m)到像素坐标系
  std::vector<cv::Point3d> world_points;
  std::vector<cv::Point2d> image_points;
  world_points.push_back(x);
  cv::projectPoints(world_points, rvec_, tvec_, camera_matrix_, distort_coeffs_, image_points);
  return image_points.back();
}

// xyz_in_world2xyz_in_pix
std::vector<cv::Point2f> Solver::reproject_buff(
  const Eigen::Vector3d & xyz_in_world, double yaw, double row) const
{
  auto R_buff2world = tools::rotation_matrix(Eigen::Vector3d(yaw, 0.0, row));

  // get R_buff2camera t_buff2camera
  const Eigen::Vector3d & t_buff2world = xyz_in_world;
  Eigen::Matrix3d R_buff2camera =
    R_camera2gimbal_.transpose() * R_gimbal2world_.transpose() * R_buff2world;
  Eigen::Vector3d t_buff2camera =
    R_camera2gimbal_.transpose() * (R_gimbal2world_.transpose() * t_buff2world - t_camera2gimbal_);

  // get rvec tvec
  cv::Vec3d rvec;
  cv::Mat R_buff2camera_cv;
  cv::eigen2cv(R_buff2camera, R_buff2camera_cv);
  cv::Rodrigues(R_buff2camera_cv, rvec);
  cv::Vec3d tvec(t_buff2camera[0], t_buff2camera[1], t_buff2camera[2]);

  // reproject
  std::vector<cv::Point2f> image_points;
  cv::projectPoints(OBJECT_POINTS, rvec, tvec, camera_matrix_, distort_coeffs_, image_points);
  return image_points;
}
}  // namespace auto_buff
