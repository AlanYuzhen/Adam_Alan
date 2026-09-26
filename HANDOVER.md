# 2027_preview 打符模块交接文档（SZU 检测 + HW 算法链合入）

> 交接日期：2026-09-26
> 交接范围：`tasks/auto_buff` 原模块被替换为合并算法链（**非独立程序**）
> Git 提交：`31d8b00`（作者 AlanYuzhen，仓库历史已重建，仅保留本次提交）

---

## 1. 背景

NUC 为 **Intel Iris Xe 集显（i5-1135G7，Ubuntu 24.04，无 NVIDIA GPU）**，原 `auto_buff` 的 yolo11_buff 检测 + 自研 EKF 打符方案效果不理想。经讨论定案：

- **检测**：SZU `26_NNDeployment_Lib_and_Detection_Models` 的 `RuneModel`（OpenVINO，Intel 集显可跑）
- **PnP / 跟踪 / 预测**：`HWauto_buff2026` 的 `BuffPnPSolver` / `BuffTracker` / `BuffPredictor`
- **弹道**：沿用 2027_preview 自带 `tools::Trajectory`
- **形态**：真正合入 `tasks/auto_buff` 原模块，对外接口不变，调用方零改动

---

## 2. 最终算法链

```
图像帧
  │
  ▼
Buff_Detector::detect(img)        ← SZU RuneModel::netProcess（OpenVINO GPU，5 关键点）
  │   输出 PowerRune（像素 4 角点 上/左/下/右 + R 字中心）
  ▼
Solver::solve(power_runes)        ← HW BuffPnPSolver（IPPE，BUFF_WIDTH=0.114m，内置 cv_to_tf）
  │   填充 pose_camera / xyz_in_world / ypr_in_world / R_gimbal2world / t_camera2gimbal
  ▼
SmallTarget / BigTarget::get_target(power_runes, ts)   ← HW BuffTracker push/update + BuffPredictor set_state
  │   内部计算世界系打击点 aim_point_world_
  ▼
Aimer::aim / mpc_aim              ← 两次 Trajectory 迭代弹道闭环（含阻力/重力）
  │   target.predict(dt) → HW BuffPredictor::predict_position(dt) → 世界系打击点
  ▼
io::Command / auto_aim::Plan
```

对外接口签名保持不变：
`Buff_Detector::detect / detect_24 / detect_debug` → `Solver::solve` → `SmallTarget/BigTarget::get_target/predict/point_buff2world/ekf_x/is_unsolve` → `Aimer::aim/mpc_aim`。

---

## 3. 文件改动清单

### 3.1 核心模块（tasks/auto_buff/）
| 文件 | 改动 |
|---|---|
| `buff_detector.hpp/cpp` | 重写：内部改为 SZU `RuneModel`（pimpl 前置声明，无 SZU 头依赖）；`detect` 取置信度最高结果，5 点转 `FanBlade`（上/左/下/右）+ R 中心 |
| `buff_solver.hpp/cpp` | 重写：`solve` 改用 HW `BuffPnPSolver`；`pnp_solver_` 为 `mutable`（solve 为 const）；填充 `pose_camera`/`R_gimbal2world`/`t_camera2gimbal` 供 target 使用；`OBJECT_POINTS` 保留给 `reproject_buff` 调试 |
| `buff_target.hpp/cpp` | 重写：内部改为 HW `BuffTracker` + `BuffPredictor`（shared_ptr，可拷贝）；`get_target` 从 PowerRune 取位姿喂 tracker；`predict(dt)` 返回世界系打击点；`ekf_x()` 兼容旧接口（x[4]=yaw, x[5]=roll, x[6]=roll_velocity） |
| `buff_type.hpp` | `PowerRune` 增加 `pose_camera`(4x4)、`R_gimbal2world`、`t_camera2gimbal` 三个字段（solver 填、target 读） |
| `buff_aimer.hpp/cpp` | **未改动**（弹道闭环已内建，完全兼容新 Target） |
| `CMakeLists.txt` | 源列表移除 `yolo11_buff.cpp`；链接 `auto_aim openvino::runtime ${CERES_LIBRARIES} NNdeployment_lib buff_algorithm` |

### 3.2 调用方（src/ 与 tests/）
| 文件 | 改动 |
|---|---|
| `src/auto_buff_debug.cpp` | `SmallTarget target;` → `SmallTarget target(config_path);` |
| `src/auto_buff_debug_mpc.cpp` | 同上 |
| `src/standard_mpc.cpp` | 两个 target 构造传 `config_path` |
| `src/uav.cpp` | 同上 |
| `src/mt_standard.cpp` | 同上 |
| `tests/auto_buff_test.cpp` | `BigTarget target(config_path);` |

### 3.3 构建与配置
| 文件 | 改动 |
|---|---|
| 主 `CMakeLists.txt` | 新增 `add_subdirectory(../26_NNDeployment_Lib_and_Detection_Models ...)`、`add_subdirectory(../HWauto_buff2026 ...)`；所有链接 auto_buff 的 target 追加 `NNdeployment_lib buff_algorithm`；`set(BUFF_ALGO_BUILD_EXAMPLES OFF ...)` 禁用 HW 自带 example（依赖被过滤的 TensorRT detector） |
| `configs/*.yaml`（sentry/uav/standard3/standard4/demo/mvs/ascento） | 补齐 `hw_buff:` 与 `rune_model:` 段 |
| `readme.md` | 末尾新增「打符算法合并说明」 |
| `HANDOVER.md` | 本文档 |

### 3.4 已删除 / 不参与编译的文件
- `src/auto_buff_szu_hw.cpp`、`src/auto_buff_hw.cpp`（独立程序形态，已废弃删除）
- `tasks/auto_buff/yolo11_buff.cpp/hpp`、`buff_predict.hpp`：**仍在目录中但已不参与编译**（死代码，见 §8）
- `tasks/auto_buff.bak_20260926/`：原模块整目录备份（回滚安全网），已加入 `.gitignore`，不进仓库

---

## 4. 关键设计决策

### 4.1 坐标系桥（最重要，勿改）
HW `BuffPnPSolver` **内部已经做了 cv_to_tf 旋转**（四元数 (-0.5,0.5,-0.5,0.5)，等价于 `R_camera2gimbal`），输出的 `pose.translation` 已经在相机**"前 x 左 y 上 z"** 系。

因此 solver 到 gimbal 系 **只加平移**：
```cpp
Eigen::Vector3d xyz_in_gimbal = pose.translation.cast<double>() + t_camera2gimbal_;
```
**绝对不要**再乘 `R_camera2gimbal`——否则双重旋转，y/z 含义全反，R 心、roll、打击点全错（这是之前 bug 的根因，已修）。

到世界系：`xyz_in_world = R_gimbal2world_ * xyz_in_gimbal`（`R_gimbal2world` 由 `set_R_gimbal2world(q)` 每帧更新）。

### 4.2 弹道闭环
`Aimer::get_send_angle` 用 2027 自带 `tools::Trajectory`（RK4，含阻力，重力 9.781）做**两次迭代闭环**：
1. `target.predict(predict_time)` → 取打击点 → `Trajectory0(v0, d, h)` 得 fly_time
2. `target.predict(trajectory0.fly_time)` → 取新打击点 → `Trajectory1` 得 pitch

每次 `predict(dt)` 调用 HW `BuffPredictor::predict_position(dt)`，HW predictor 内部补偿 state age，即"HW predictor 被弹道迭代反复调用"。

### 4.3 时间戳
统一 `std::chrono::steady_clock`，转 double 秒：
```cpp
const double ts = std::chrono::duration<double>(timestamp.time_since_epoch()).count();
```
get_target 里 `predictor_->set_state(state, ts, ts)`（base_ts = state_ts = ts），后续 `predict(dt)` 即预测 dt 秒后位置。

### 4.4 颜色 / 模式
从 config 读（不是从 ECS 串口收）：
- `hw_buff.buff_mode`: 1=小符(SMALL_BUFF), 2=大符(BIG_BUFF) —— 决定构造 SmallTarget 还是 BigTarget
- `hw_buff.buff_color`: 0=蓝方, 1=红方（当前红方）

### 4.5 丢失保护
- 连续丢失 > 15 帧 → `tracker_->reset()` 并置 unsolvable
- 可解判定：`tracker->status() != buff_algo::StatusType::LOST`（降级保护；大符 RANSAC 未 ready 时处于 CONVERGING 仍视为可解，见 §7）
- `Target::spd` 在 `predict()` 中更新（ekf_x 为 const，不能赋值）

---

## 5. 配置文件

`configs/sentry.yaml`（其它 yaml 同结构）：
```yaml
hw_buff:
  config_dir: "/home/asus/Alan/HWauto_buff2026/config"
  buff_color: 1        # 0=蓝方, 1=红方
  buff_mode: 2         # 1=小符, 2=大符
  confidence_threshold: 0.5
  nms_threshold: 0.45
rune_model:
  path: "/home/asus/Alan/26_NNDeployment_Lib_and_Detection_Models/所有模型/openvino/Rune-v8n-fp16-20260624-D14367-B16/Rune-v8n-fp16-20260624-D14367-B16.xml"
```

HW 内部参数（勿随意改，与算法强耦合）：
- `tracker.yaml`：small（switch_buff_angle=45.8°, R_center_filter_ratio=0.1）、big（RANSAC min_inliers=100, omega 1.884~2.0, A 0.78~1.045）、temp_lost_return_frames=5
- `predictor.yaml`：空节点

外参（各 yaml 独立）：`camera_matrix` / `distort_coeffs` / `R_camera2gimbal` / `t_camera2gimbal` / `R_gimbal2imubody` 按各自标定值。

---

## 6. 构建与运行

```bash
# 构建（需外部依赖目录：../26_NNDeployment_Lib_and_Detection_Models、../HWauto_buff2026 与 2027_preview 同级）
cmake -B build
make -C build -j$(nproc)

# 离线验证（打符视频）
./build/auto_buff_test <视频路径(不含扩展名)> --config-path=configs/sentry.yaml --start-index=0 --end-index=N

# 实车（示例）
./build/standard configs/sentry.yaml
```

**注意**：OpenCV `CommandLineParser` 对 `-c xxx` 短参数解析有 bug（会把 config-path 解析成 "true"），必须用 `--config-path=xxx` 等号形式。

---

## 7. 验证结果

- ✅ 全量 `make -j4` 编译通过（MAKE_EXIT=0）
- ✅ 离线冒烟：`auto_buff_test` 跑 demo 视频正常退出（EXIT=0），RuneModel 加载成功、tracker/predictor/aimer 链路无崩溃
- ✅ 打符视频实测（SZU `测试视频/符.avi`，200 帧采样）：**detect_hit=82**，检测到后全部可解；全片 800 帧命中 671（前 118 帧为视频开头无目标段）
- ✅ 模型在 Intel Iris Xe GPU 上正常运行（OpenVINO device="GPU"）

**待改进（已讨论未定）**：
- SZU 提示标准打符应用 **sync 推理**（当前 async）；若改用 sync 需在 `buff_detector.cpp` 构造处把 `"async"` 改为 `"sync"` 并复测
- 大符 RANSAC 冷启动保护：当前用 `status() != LOST` 降级；更严格方案是给 `BuffTracker` 加 public `ransac_ready()` 转发（需改 HW 库）
- 模型输入分辨率：当前 SZU 默认 640×384；如换模型需同步确认后处理

---

## 8. 遗留项 / 待办

| 项 | 说明 | 建议 |
|---|---|---|
| `yolo11_buff.cpp/hpp`、`buff_predict.hpp` | 死代码，不参与编译 | 确认不再需要后删除 |
| demo.avi 59.75MB | 超过 GitHub 50MB 推荐上限（推送仅 warning） | 移出仓库或 Git LFS |
| 大符 RANSAC 保护 | 现为降级方案 | 按 §7 加 `ransac_ready()` |
| 模型精度 | Rune-v8n-fp16 已可用 | 有更高精度模型可替换 `rune_model.path` 后复测 |
| 其它 main 的配置 | 已补 6 个 yaml | 新增兵种 main 需同步补 `hw_buff`/`rune_model` 段 |

---

## 9. 回滚方案

原模块完整备份在 `tasks/auto_buff.bak_20260926/`（含 yolo11_buff，可随时还原）：
```bash
# 备份当前合并版
mv tasks/auto_buff tasks/auto_buff.szu_hw
# 还原原版
mv tasks/auto_buff.bak_20260926 tasks/auto_buff
# 还原 CMakeLists 与调用方构造行（见 git 历史）
```

---

## 10. 外部依赖

| 仓库 | 路径（与 2027_preview 同级） | 用途 |
|---|---|---|
| `26_NNDeployment_Lib_and_Detection_Models` | `/home/asus/Alan/26_NNDeployment_Lib_and_Detection_Models` | RuneModel 检测（OpenVINO） |
| `HWauto_buff2026` | `/home/asus/Alan/HWauto_buff2026` | PnP / tracker / predictor（TensorRT detector 已过滤） |

依赖通过主 CMakeLists 的 `add_subdirectory(../...)` 引用，**不随本仓库提交**；新环境需保证三个仓库同级摆放。
