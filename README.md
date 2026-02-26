# 3D-Circle-Recon: 高性能双目 3D 圆重建系统

[![C++](https://img.shields.io/badge/C++-17-blue.svg)](https://isocpp.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![Eigen](https://img.shields.io/badge/Eigen-3.x-orange.svg)](https://eigen.tuxfamily.org/)

本项目实现了一套从双目图像中提取并重建 3D 圆的高性能算法系统。核心流程涵盖了亚像素级椭圆检测、基于极线几何的椭圆匹配、初值几何估算以及非线性平差优化。该系统能够快速、高精度地获取空间中圆形的 3D 位姿（中心、法线、半径）。

---

## 🚀 核心功能

- **亚像素椭圆检测**：
  - 基于 ED (Edge Drawing) 思路的高效边缘提取。
  - 通过径向梯度纠正与 Gauss-Newton 优化实现亚像素精度。
  - 引入 Tukey Biweight 权重函数，对噪声和遮挡具有极强的鲁棒性。
- **鲁棒双目匹配**：
  - 利用基础矩阵 (Fundamental Matrix) 约束进行极线对候选点的过滤。
  - 通过动态规划或单调性约束确保匹配的一致性，支持复杂场景下的多目标匹配。
- **高精度 3D 重建**：
  - **初值估算**：采用圆锥分解法 (Cone Decomposition) 结合极线约束，从单对匹配椭圆中解析提取 3D 圆初值。
  - **非线性优化**：构建基于重投影误差的 Levenberg-Marquardt (LM) 优化模型，精修 3D 圆的 6 自由度参数（中心、方向、半径）。
- **畸变纠正**：内置完整的去畸变流程，确保在大畸变镜头下仍能保持极高的重建精度。
- **多维产出**：
  - 结构化的 YAML 结果输出（3D 位姿、重建误差）。
  - 可视化的 PLY 点云模型（圆周点云）。
  - 重投影对比图（检测椭圆 vs. 重投影椭圆）。

---

## 🛠️ 算法逻辑流程

系统的核心处理管线如下：

### 1. 图像预处理与边缘提取
对左右图像分别进行去畸变处理。采用 EDSF (Edge Drawing based Subpixel Fitting) 算法提取连续边缘段，通过凸性筛选与曼哈顿距离闭合判定提取椭圆候选。

### 2. 双目椭圆匹配
对于左图中检测到的每个椭圆，根据相机的外参（R, T）和内参构建极线约束。在右图中搜索满足几何一致性（如中心点位于极线附近、长短轴比例相近）的候选对象进行配对。

### 3. 3D 几何解析初算
利用匹配的两个椭圆方程（二系像平面圆锥曲线），结合双目几何推导出对应的 3D 圆初值。该步骤解决了解析解中的歧义性问题，并为后续优化提供高质量的初值。

### 4. LM 非线性平差精修 (核心)
定义代价函数为 3D 圆投影到左右像平面后的重投影误差。使用 Levenberg-Marquardt 算法最小化残差：
- **状态向量**：3D 中心 $(x, y, z)$，法线方向（球面坐标系参数化），半径 $r$。
- **鲁棒核函数**：对偏离点进行惩罚，确保在高噪声环境下收敛。

---

## 📦 环境依赖

- **编译器**: 支持 C++17 的编译器（如 GCC 9+）
- **CMake**: 3.16+
- **OpenCV**: 4.0+ (需包含 `highgui`, `imgproc`, `calib3d` 模块)
- **Eigen**: 3.3+ (用于高效的矩阵运算与非线性优化)

---

## 🔧 编译与运行

### 1. 编译项目
```bash
cmake -B build
cmake --build build -j$(nproc)
```

### 2. 运行 3D 重建
程序需要一个包含特定结构的目录作为输入：
```bash
./build/ellipse_detector -f <folder_path> [选项]
```

**常用选项**:
- `-f, --folder <路径>`: 指定数据集目录（需包含 `left/`, `right/` 和 `calibration.yaml`）。
- `-v, --visualize`: 开启可视化界面，实时查看重建与重投影效果。
- `-d, --debug`: 输出中间过程诊断图像（生成在 `debug_diag/` 下）。

### 3. 测试示例
```bash
./build/ellipse_detector -f dataset/laser_01 -v
```

---

## 📂 数据格式说明

### 数据集目录结构
输入目录应遵循以下结构：
```text
dataset/example/
├── left/               # 左相机图像文件夹 (.jpg, .png, .bmp)
├── right/              # 右相机图像文件夹 (需与左图同名)
└── calibration.yaml    # 双目相机标定参数
```

### calibration.yaml 格式
标定文件应包含以下 OpenCV `FileStorage` 兼容的节点：
- `camera_matrix_left` & `camera_matrix_right`: 3x3 内参矩阵。
- `dist_coeffs_left` & `dist_coeffs_right`: 畸变系数。
- `R`: 右相机相对于左相机的 3x3 旋转矩阵。
- `T`: 右相机相对于左相机的 3x1 平移向量 (单位通常为 mm)。

---

## 📊 结果产出与说明

处理结果将存储在数据集目录下的 `results/` 文件夹中：
1. **`*_circles.yaml`**: 记录了每个重建圆的 3D 参数（Center, Normal, Radius）以及重投影误差 (error) 和置信度得分 (score)。
2. **`*_edges.ply`**: 输出空间圆周的采样点云，可直接使用 MeshLab 或 CloudCompare 查看 3D 效果。
3. **`*_vis.png`**: 渲染重投影结果，红色为 3D 圆反投影到图像的预测位置，绿色为检测到的初始椭圆，两者重合度越高代表精度越高。

---

## 📝 后续开发建议

- **核心代码导读**:
  - `src/ellipse_detector.cpp`: 椭圆检测核心逻辑，若需调整检测灵敏度，请修改 `param.h` 中的 `gradThresh`。
  - `src/circle_optimizer.cpp`: 3D 重建与 LM 优化的数学实现。
  - `src/stereo_geometry.cpp`: 极线几何计算与圆锥分解法实现。
- **待优化项**:
  - 核心思想是“假设生成->验证->优化”，而不是"匹配->生成->优化"，采用的是解析法，但严重依赖于椭圆参数$a, b, \phi$，实际上我们只能确保椭圆的中心提取精度，并不能确保$a, b, \phi$精度，所以后续应该采用“优化法（网格搜索+微调）”可能会更好。
  - 既然已知“中心点”是唯一高度可信的特征，我们就应该以高精度三角化的 3D 中心点作为无条件锚点。将法向量和半径作为未知数，在空间中生成圆，并让它的 3D 边缘点投影回2D图像，比较“重投影坐标”与“提取边缘”的点对点距离 (Sampson / L2 欧氏距离)。
