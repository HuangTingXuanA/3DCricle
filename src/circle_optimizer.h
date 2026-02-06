#pragma once

/**
 * @file circle_optimizer.h
 * @brief 3D圆重建优化器 - 基于非线性优化的圆形重建算法
 */

#include "stereo_geometry.h"
#include "ellipse_detector.h"
#include <vector>

namespace stereo {

// ============================================================================
// 参数与结果结构体
// ============================================================================

/**
 * @brief 优化器参数
 */
struct OptimizerParams {
    // LM 优化参数
    double convergeTol = 1e-8;        // 收敛阈值
    int maxIter = 100;                // 最大迭代次数
    double lambdaInit = 1e-3;         // 初始阻尼系数
    double lambdaUp = 10.0;           // 阻尼增大因子
    double lambdaDown = 0.1;          // 阻尼减小因子
    
    // 椭圆匹配参数（进一步放宽以提高重建率）
    double epipolarThresh = 20.0;     // 极线约束阈值 (像素) [10→20]
    double areaRatioMin = 0.2;        // 面积比最小值 [0.3→0.2]
    double areaRatioMax = 5.0;        // 面积比最大值 [3.0→5.0]
    double axisRatioTol = 0.6;        // 长短轴比容差 [0.5→0.6]
    
    // 代价函数权重
    double weightCenter = 1.0;        // 中心权重
    double weightAxis = 0.5;          // 轴长权重
    double weightAngle = 0.2;         // 角度权重
    
    // 几何约束
    double minRadius = 0.5;           // 最小半径 (mm)
    double maxRadius = 500.0;         // 最大半径 (mm)
    double minDepth = 100.0;           // 最小深度 (mm)
    double maxDepth = 3000.0;        // 最大深度 (mm)
    
    // 验证阈值
    double maxReprojError = 3.0;      // 最大重投影误差 (像素) [2.0→3.0]
    
    // 边缘采样
    int numEdgePoints = 360;          // 边缘采样点数
    bool useEdgeRefinement = true;    // 是否使用边缘精化
};

/**
 * @brief 优化结果
 */
struct OptimizeResult {
    Circle3D circle;                           // 重建的3D圆
    std::vector<Eigen::Vector3d> edgePoints;   // 边缘3D点 (360个)
    
    double finalCost;                          // 最终代价
    double reprojErrorL;                       // 左图重投影误差 (像素)
    double reprojErrorR;                       // 右图重投影误差 (像素)
    
    int iterations;                            // 迭代次数
    bool converged;                            // 是否收敛
    bool valid;                                // 是否有效
    
    int ellipseIdxL;                           // 匹配的左图椭圆索引
    int ellipseIdxR;                           // 匹配的右图椭圆索引
    
    OptimizeResult() : finalCost(std::numeric_limits<double>::max()),
                       reprojErrorL(0), reprojErrorR(0),
                       iterations(0), converged(false), valid(false),
                       ellipseIdxL(-1), ellipseIdxR(-1) {}
};

// ============================================================================
// 优化器类
// ============================================================================

/**
 * @brief 3D圆优化器
 * 
 * 采用 Levenberg-Marquardt 算法最小化重投影误差
 */
class CircleOptimizer {
public:
    /**
     * @brief 构造函数
     * @param stereo 双目相机参数
     * @param params 优化器参数
     */
    CircleOptimizer(const StereoParams& stereo, 
                    const OptimizerParams& params = OptimizerParams());
    
    /**
     * @brief 优化单对椭圆 → 3D圆
     * 
     * @param ellipseL 左图椭圆
     * @param ellipseR 右图椭圆
     * @return 优化结果
     */
    OptimizeResult optimize(const Ellipse& ellipseL, const Ellipse& ellipseR);
    
    /**
     * @brief 重建所有3D圆（自动匹配）
     * 
     * 流程:
     * 1. 极线约束预筛选
     * 2. 逐对优化
     * 3. 重投影误差验证
     * 
     * @param ellipsesL 左图椭圆列表
     * @param ellipsesR 右图椭圆列表
     * @return 所有有效的重建结果
     */
    std::vector<OptimizeResult> reconstructAll(
        const std::vector<Ellipse>& ellipsesL,
        const std::vector<Ellipse>& ellipsesR);

private:
    // 获取优化初始值
    Circle3D getInitialGuess(const Ellipse& ellipseL, const Ellipse& ellipseR);
    
    // 阶段一：椭圆参数优化 (LM)
    Circle3D optimizeEllipseParams(const Circle3D& init,
                                    const Ellipse& ellipseL,
                                    const Ellipse& ellipseR,
                                    int& iterations,
                                    bool& converged);
    
    // 阶段二：边缘点精化 (可选)
    Circle3D refineWithEdgePoints(const Circle3D& init,
                                   const Ellipse& ellipseL,
                                   const Ellipse& ellipseR);
    
    // 计算代价函数
    double computeCost(const Circle3D& circle,
                       const Ellipse& ellipseL,
                       const Ellipse& ellipseR);
    
    // 计算雅可比矩阵和残差
    void computeJacobianAndResidual(const Circle3D& circle,
                                     const Ellipse& ellipseL,
                                     const Ellipse& ellipseR,
                                     Eigen::Matrix<double, 10, 6>& J,
                                     Eigen::Matrix<double, 10, 1>& r);
    
    // 参数化3D圆 (6自由度)
    Eigen::Matrix<double, 6, 1> circleToParams(const Circle3D& circle);
    Circle3D paramsToCircle(const Eigen::Matrix<double, 6, 1>& params);
    
    // 计算单侧重投影误差
    double computeReprojError(const Circle3D& circle, 
                              const Ellipse& ellipse,
                              bool isLeft);
    
    // 生成边缘3D点
    std::vector<Eigen::Vector3d> sampleEdgePoints(const Circle3D& circle);
    
    // 检查极线约束
    bool passEpipolarCheck(const Ellipse& ellipseL, const Ellipse& ellipseR);
    
    // 检查面积比约束
    bool passAreaRatioCheck(const Ellipse& ellipseL, const Ellipse& ellipseR);
    
    // 检查形状相似度
    bool passShapeCheck(const Ellipse& ellipseL, const Ellipse& ellipseR);

private:
    StereoParams stereo_;
    OptimizerParams params_;
    
    // 缓存投影矩阵
    Eigen::Matrix<double, 3, 4> P_L_;
    Eigen::Matrix<double, 3, 4> P_R_;
};

} // namespace stereo
