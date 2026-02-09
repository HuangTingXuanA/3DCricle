#pragma once

/**
 * @file stereo_geometry.h
 * @brief 双目几何核心库 - 第一性原理重构版
 * "Simplicity is the ultimate sophistication."
 */

#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <vector>
#include "ellipse_detector.h" // Assuming Ellipse struct is defined here

namespace stereo {

// ============================================================================
// 基础数据结构
// ============================================================================

/**
 * @brief 3D圆
 */
struct Circle3D {
    Eigen::Vector3d center;
    Eigen::Vector3d normal;
    double radius;
    double confidence = 0.0; // 置信度分数

    Circle3D() : center(0,0,0), normal(0,0,1), radius(0) {}
    Circle3D(const Eigen::Vector3d& c, const Eigen::Vector3d& n, double r)
        : center(c), normal(n.normalized()), radius(r) {}
};

/**
 * @brief 双目相机参数
 */
struct StereoParams {
    Eigen::Matrix3d K_L, K_R;
    Eigen::Matrix<double, 1, 5> dist_L, dist_R;
    Eigen::Matrix3d R;
    Eigen::Vector3d T;
    Eigen::Matrix3d F; // Fundamental Matrix
    
    // 预计算投影矩阵
    Eigen::Matrix<double, 3, 4> P_L, P_R;

    StereoParams() {
        K_L.setIdentity(); K_R.setIdentity();
        R.setIdentity(); T.setZero(); F.setZero();
    }
    
    void init(); // 计算 F, P_L, P_R
};

// ============================================================================
// 核心几何算法 - 第一性原理
// ============================================================================

/**
 * @brief 从单目椭圆恢复圆法向量 (Geometry from Shape)
 * 
 * 基于透视投影原理，一个圆投影为椭圆，其法向量只有两个解（二义性）。
 * 参考: "3D Construction of Circles from Their Ellipse Images"
 * 
 * @param ellipse 检测到的椭圆
 * @param K 相机内参
 * @return 两个可能的归一化法向量 (相对于相机坐标系)
 */
std::pair<Eigen::Vector3d, Eigen::Vector3d> estimateCircleNormal(
    const Ellipse& ellipse, 
    const Eigen::Matrix3d& K
);

/**
 * @brief 3D圆重投影为2D椭圆 (Verification)
 * @return 投影得到的椭圆参数
 */
Ellipse projectCircle(
    const Circle3D& circle,
    const Eigen::Matrix3d& K,
    const Eigen::Matrix<double, 3, 4>& P  //虽然P包含K，但为了接口清晰保留K用于归一化平面计算（如果需要）
    // 实际上通常 P = K[R|t]，直接用P投影点即可
);

/**
 * @brief 计算点到椭圆的 Sampson 距离 (Robust Error Metric)
 * @return 距离值（像素）
 */
double computeSampsonDistance(
    const cv::Point2d& pt, 
    const Ellipse& ellipse
);

/**
 * @brief 计算两个椭圆参数之间的简单误差
 * sum of center dist + axis diffs
 */
double ellipseParamError(const Ellipse& e1, const Ellipse& e2);

/**
 * @brief 三角测量 (Triangulate)
 * @return 3D点 (左相机坐标系)
 */
Eigen::Vector3d triangulate(
    const Eigen::Vector2d& ptL, 
    const Eigen::Vector2d& ptR, 
    const StereoParams& stereo
);

/**
 * @brief 计算极线距离 (Epipolar Constraint)
 */
double distanceFromEpiline(
    const Eigen::Vector2d& ptR, 
    const Eigen::Vector2d& ptL, 
    const Eigen::Matrix3d& F
);

/**
 * @brief 求解3D圆 (Solve 3D Circle)
 * 给定圆心和法向量，计算最佳半径
 */
double optimizeRadius(
    const Eigen::Vector3d& center,
    const Eigen::Vector3d& normal,
    const Ellipse& eL,
    const Ellipse& eR,
    const StereoParams& stereo
);

/**
 * @brief 在3D圆上均匀采样点 (便于ply显示)
 * @return 采样点列表
 */
std::vector<Eigen::Vector3d> sampleCircleEdgePoints(const Circle3D& circle, int numPoints = 100);

} // namespace stereo
