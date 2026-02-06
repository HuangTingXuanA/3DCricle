#pragma once

/**
 * @file stereo_geometry.h
 * @brief 双目几何核心模块 - 3D圆投影与重建的数学基础
 */

#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include "ellipse_detector.h"

namespace stereo {

// ============================================================================
// 数据结构
// ============================================================================

/**
 * @brief 3D空间圆结构体
 */
struct Circle3D {
    Eigen::Vector3d center;   // 圆心坐标 (mm)
    Eigen::Vector3d normal;   // 法向量 (单位向量)
    double radius;            // 半径 (mm)
    
    Circle3D() : center(Eigen::Vector3d::Zero()), 
                 normal(Eigen::Vector3d::UnitZ()), 
                 radius(0) {}
    
    Circle3D(const Eigen::Vector3d& c, const Eigen::Vector3d& n, double r)
        : center(c), normal(n.normalized()), radius(r) {}
};

/**
 * @brief 双目相机标定参数
 */
struct StereoParams {
    Eigen::Matrix3d K_L;                      // 左相机内参
    Eigen::Matrix3d K_R;                      // 右相机内参
    Eigen::Matrix<double, 1, 5> dist_L;       // 左相机畸变系数 (k1,k2,p1,p2,k3)
    Eigen::Matrix<double, 1, 5> dist_R;       // 右相机畸变系数
    Eigen::Matrix3d R;                        // 右相机相对左相机的旋转
    Eigen::Vector3d T;                        // 右相机相对左相机的平移 (mm)
    
    // 基础矩阵 F (可由 K, R, T 计算)
    Eigen::Matrix3d F;
    
    StereoParams() {
        K_L.setIdentity();
        K_R.setIdentity();
        dist_L.setZero();
        dist_R.setZero();
        R.setIdentity();
        T.setZero();
        F.setZero();
    }
    
    /**
     * @brief 计算基础矩阵 F
     * F = K_R^{-T} * [T]_x * R * K_L^{-1}
     */
    void computeFundamentalMatrix();
    
    /**
     * @brief 获取左相机投影矩阵 P_L = K_L * [I | 0]
     */
    Eigen::Matrix<double, 3, 4> getProjectionMatrixL() const;
    
    /**
     * @brief 获取右相机投影矩阵 P_R = K_R * [R | T]
     */
    Eigen::Matrix<double, 3, 4> getProjectionMatrixR() const;
};

// ============================================================================
// 核心几何函数
// ============================================================================

/**
 * @brief 构造3D圆的对偶二次曲面矩阵 Q* (4x4)
 * 
 * 对偶二次曲面表示圆所在的平面与圆本身的几何约束
 * 
 * @param center 圆心
 * @param normal 法向量 (单位向量)
 * @param radius 半径
 * @return 4x4对偶二次曲面矩阵
 */
Eigen::Matrix4d buildDualQuadric(const Eigen::Vector3d& center,
                                  const Eigen::Vector3d& normal,
                                  double radius);

/**
 * @brief 将3D圆投影为图像椭圆
 * 
 * 核心公式: C* = P * Q* * P^T
 * 
 * @param circle 3D圆
 * @param K 相机内参
 * @param P 3x4投影矩阵
 * @return 图像椭圆参数
 */
Ellipse projectCircleToEllipse(const Circle3D& circle,
                                const Eigen::Matrix3d& K,
                                const Eigen::Matrix<double, 3, 4>& P);

/**
 * @brief 从对偶二次曲线矩阵提取椭圆参数
 * 
 * @param C_star 3x3对偶二次曲线矩阵
 * @return 椭圆参数 (center, a, b, phi)
 */
Ellipse conicMatrixToEllipse(const Eigen::Matrix3d& C_star);

/**
 * @brief 将椭圆参数转换为二次曲线矩阵
 */
Eigen::Matrix3d ellipseToConicMatrix(const Ellipse& ellipse);

/**
 * @brief 三角测量：从左右图像点恢复3D点
 * 
 * @param pt_L 左图像点 (像素)
 * @param pt_R 右图像点 (像素)
 * @param stereo 双目参数
 * @return 3D点坐标 (mm, 左相机坐标系)
 */
Eigen::Vector3d triangulatePoint(const Eigen::Vector2d& pt_L,
                                  const Eigen::Vector2d& pt_R,
                                  const StereoParams& stereo);

/**
 * @brief 计算点到极线的距离
 * 
 * @param pt 图像点
 * @param epiline 极线 ax + by + c = 0 (归一化)
 * @return 距离 (像素)
 */
double pointToEpilineDistance(const Eigen::Vector2d& pt,
                               const Eigen::Vector3d& epiline);

/**
 * @brief 计算左图点对应的右图极线
 */
Eigen::Vector3d computeEpilineR(const Eigen::Vector2d& pt_L,
                                 const Eigen::Matrix3d& F);

/**
 * @brief 计算右图点对应的左图极线
 */
Eigen::Vector3d computeEpilineL(const Eigen::Vector2d& pt_R,
                                 const Eigen::Matrix3d& F);

/**
 * @brief 去畸变单点
 */
Eigen::Vector2d undistortPoint(const Eigen::Vector2d& pt,
                                const Eigen::Matrix3d& K,
                                const Eigen::Matrix<double, 1, 5>& dist);

/**
 * @brief 计算两个椭圆参数之间的误差
 * 
 * @param e1 椭圆1
 * @param e2 椭圆2
 * @param weights 权重 [w_center, w_axis, w_angle]
 * @return 加权误差
 */
double ellipseParamError(const Ellipse& e1, const Ellipse& e2,
                          const Eigen::Vector3d& weights = Eigen::Vector3d(1.0, 1.0, 0.5));

/**
 * @brief 计算点到椭圆边缘的Sampson距离
 * 
 * Sampson距离是代数距离除以梯度范数，近似几何距离
 * 
 * @param pt 2D点
 * @param ellipse 椭圆参数
 * @return 距离 (像素)
 */
double pointToEllipseDistance(const cv::Point2d& pt, const Ellipse& ellipse);

} // namespace stereo
