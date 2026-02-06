/**
 * @file circle_optimizer.cpp
 * @brief 3D圆重建优化器实现
 */

#include "circle_optimizer.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <set>
#include <cmath>

namespace stereo {

// ============================================================================
// 构造函数
// ============================================================================

CircleOptimizer::CircleOptimizer(const StereoParams& stereo, 
                                  const OptimizerParams& params)
    : stereo_(stereo), params_(params) {
    // 缓存投影矩阵
    P_L_ = stereo_.getProjectionMatrixL();
    P_R_ = stereo_.getProjectionMatrixR();
}

// ============================================================================
// 公开接口
// ============================================================================

OptimizeResult CircleOptimizer::optimize(const Ellipse& ellipseL, 
                                          const Ellipse& ellipseR) {
    OptimizeResult result;
    
    // 1. 获取初始值
    Circle3D init = getInitialGuess(ellipseL, ellipseR);
    
    // 检查初始值有效性
    if (init.center.z() < params_.minDepth || init.center.z() > params_.maxDepth) {
        result.valid = false;
        return result;
    }
    if (init.radius < params_.minRadius || init.radius > params_.maxRadius) {
        result.valid = false;
        return result;
    }
    
    // 2. 阶段一：椭圆参数优化
    int iterations = 0;
    bool converged = false;
    Circle3D optimized = optimizeEllipseParams(init, ellipseL, ellipseR, 
                                                iterations, converged);
    
    // 3. 阶段二：边缘精化（可选）
    if (params_.useEdgeRefinement && converged) {
        optimized = refineWithEdgePoints(optimized, ellipseL, ellipseR);
    }
    
    // 4. 计算最终误差
    result.circle = optimized;
    result.finalCost = computeCost(optimized, ellipseL, ellipseR);
    result.reprojErrorL = computeReprojError(optimized, ellipseL, true);
    result.reprojErrorR = computeReprojError(optimized, ellipseR, false);
    result.iterations = iterations;
    result.converged = converged;
    
    // 5. 验证结果
    result.valid = converged && 
                   result.reprojErrorL < params_.maxReprojError &&
                   result.reprojErrorR < params_.maxReprojError &&
                   optimized.radius > params_.minRadius &&
                   optimized.radius < params_.maxRadius;
    
    // 6. 生成边缘点
    if (result.valid) {
        result.edgePoints = sampleEdgePoints(optimized);
    }
    
    return result;
}

std::vector<OptimizeResult> CircleOptimizer::reconstructAll(
    const std::vector<Ellipse>& ellipsesL,
    const std::vector<Ellipse>& ellipsesR) {
    
    std::vector<OptimizeResult> results;
    std::set<int> usedL, usedR;
    
    // 收集候选对
    struct Candidate {
        int idxL, idxR;
        double epipolarDist;
        double xL, xR;  // 中心x坐标，用于顺序约束
    };
    std::vector<Candidate> candidates;
    int passEpiCount = 0, failAreaCount = 0, failShapeCount = 0, failDispCount = 0;
    
    for (int i = 0; i < static_cast<int>(ellipsesL.size()); ++i) {
        for (int j = 0; j < static_cast<int>(ellipsesR.size()); ++j) {
            // 极线距离检查
            Eigen::Vector2d centerL(ellipsesL[i].center.x, ellipsesL[i].center.y);
            Eigen::Vector3d epiline = computeEpilineR(centerL, stereo_.F);
            Eigen::Vector2d centerR(ellipsesR[j].center.x, ellipsesR[j].center.y);
            double epiDist = pointToEpilineDistance(centerR, epiline);
            
            if (epiDist >= params_.epipolarThresh) {
                ++passEpiCount;
                continue;
            }
            
            // 面积比检查
            double areaL = M_PI * ellipsesL[i].a * ellipsesL[i].b;
            double areaR = M_PI * ellipsesR[j].a * ellipsesR[j].b;
            double areaRatio = (areaR > 1e-6) ? areaL / areaR : 0;
            if (areaRatio < params_.areaRatioMin || areaRatio > params_.areaRatioMax) {
                ++failAreaCount;
                continue;
            }
            
            // 形状相似度检查
            double ratioL = ellipsesL[i].b / ellipsesL[i].a;
            double ratioR = ellipsesR[j].b / ellipsesR[j].a;
            double shapeDiff = std::abs(ratioL - ratioR);
            if (shapeDiff >= params_.axisRatioTol) {
                ++failShapeCount;
                continue;
            }
            
            // 视差检查（放宽）
            double disparity = ellipsesL[i].center.x - ellipsesR[j].center.x;
            if (disparity < -100) {
                ++failDispCount;
                continue;  // 视差过负，不太可能是正确匹配
            }
            
            candidates.push_back({i, j, epiDist, ellipsesL[i].center.x, ellipsesR[j].center.x});
        }
    }
    
    // 输出诊断信息
    static int diagCount = 0;
    if (diagCount < 2) {
        std::cout << "  [诊断] 极线滤除=" << passEpiCount 
                  << " 面积滤除=" << failAreaCount 
                  << " 形状滤除=" << failShapeCount 
                  << " 视差滤除=" << failDispCount << std::endl;
        diagCount++;
    }
    
    std::cout << "  预筛选候选对: " << candidates.size() << " / " 
              << ellipsesL.size() * ellipsesR.size() << std::endl;
    
    // 按极线距离和左图x坐标复合排序（优先极线距离，同极线内按x排序）
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) {
                  if (std::abs(a.epipolarDist - b.epipolarDist) < 0.5) {
                      return a.xL < b.xL;  // 同一极线上按x排序
                  }
                  return a.epipolarDist < b.epipolarDist;
              });
    
    // 顺序约束：已匹配的椭圆对应形成的映射必须保持x坐标单调性
    // 记录已接受匹配的 (xL, xR) 对
    std::vector<std::pair<double, double>> acceptedMatches;
    
    // 逐对优化，同时应用顺序约束
    for (const auto& cand : candidates) {
        if (usedL.count(cand.idxL) || usedR.count(cand.idxR)) {
            continue;
        }
        
        // 顺序约束检查：新匹配不应打破已有匹配的单调性
        bool orderViolation = false;
        for (const auto& accepted : acceptedMatches) {
            // 如果新匹配的xL > 已有的xL，则新匹配的xR也应该 > 已有的xR
            // 如果新匹配的xL < 已有的xL，则新匹配的xR也应该 < 已有的xR
            if ((cand.xL > accepted.first && cand.xR < accepted.second) ||
                (cand.xL < accepted.first && cand.xR > accepted.second)) {
                // 允许一定容差（相近的点可能由于噪声交叉）
                if (std::abs(cand.xL - accepted.first) > 20 && 
                    std::abs(cand.xR - accepted.second) > 20) {
                    orderViolation = true;
                    break;
                }
            }
        }
        
        if (orderViolation) {
            continue;  // 跳过违反顺序约束的匹配
        }
        
        OptimizeResult result = optimize(ellipsesL[cand.idxL], ellipsesR[cand.idxR]);
        result.ellipseIdxL = cand.idxL;
        result.ellipseIdxR = cand.idxR;
        
        if (result.valid) {
            results.push_back(result);
            usedL.insert(cand.idxL);
            usedR.insert(cand.idxR);
            acceptedMatches.push_back({cand.xL, cand.xR});
        }
    }
    
    std::cout << "  候选匹配对: " << candidates.size() 
              << " 顺序约束后接受: " << acceptedMatches.size() << std::endl;
    
    return results;
}

// ============================================================================
// 私有方法
// ============================================================================

Circle3D CircleOptimizer::getInitialGuess(const Ellipse& ellipseL, 
                                           const Ellipse& ellipseR) {
    // 1. 三角测量圆心初始值
    Eigen::Vector2d centerL(ellipseL.center.x, ellipseL.center.y);
    Eigen::Vector2d centerR(ellipseR.center.x, ellipseR.center.y);
    
    Eigen::Vector3d C_init = triangulatePoint(centerL, centerR, stereo_);
    
    // 2. 从椭圆形状估计法向量
    // 原理：椭圆轴比 b/a = |cos(α)|，其中α是法向量与视线的夹角
    //       椭圆长轴方向垂直于法向量在图像平面的投影
    
    // 左相机视线方向（从相机指向圆心）
    Eigen::Vector3d viewL = C_init.normalized();
    
    // 从左图椭圆估计倾斜角
    double axisRatioL = ellipseL.b / ellipseL.a;
    double alphaL = std::acos(std::clamp(axisRatioL, 0.01, 0.99));  // 避免0和1
    
    // 椭圆长轴方向（图像坐标系）
    double phi_L = ellipseL.phi;
    
    // 图像平面上法向量投影方向（垂直于长轴）
    Eigen::Vector2d projDir2D(-std::sin(phi_L), std::cos(phi_L));
    
    // 转换到归一化相机坐标系
    double fx = stereo_.K_L(0, 0);
    double fy = stereo_.K_L(1, 1);
    double cx = stereo_.K_L(0, 2);
    double cy = stereo_.K_L(1, 2);
    
    // 椭圆中心的归一化坐标
    double x_norm = (ellipseL.center.x - cx) / fx;
    double y_norm = (ellipseL.center.y - cy) / fy;
    
    // 法向量投影方向在相机坐标系中
    Eigen::Vector3d projDir3D(projDir2D.x() / fx, projDir2D.y() / fy, 0);
    projDir3D.normalize();
    
    // 构建与视线垂直的平面
    Eigen::Vector3d right = viewL.cross(Eigen::Vector3d::UnitZ());
    if (right.norm() < 0.1) {
        right = viewL.cross(Eigen::Vector3d::UnitX());
    }
    right.normalize();
    Eigen::Vector3d up = right.cross(viewL);
    up.normalize();
    
    // 法向量初值：视线方向 + 倾斜修正
    // 使用椭圆方向确定倾斜轴
    double tiltAngle = M_PI / 2 - alphaL;  // 从视线偏离的角度
    Eigen::Vector3d tiltAxis = std::cos(phi_L) * right + std::sin(phi_L) * up;
    
    // 使用Rodrigues旋转公式
    Eigen::Vector3d n_init = viewL * std::cos(tiltAngle) + 
                              tiltAxis.cross(viewL) * std::sin(tiltAngle) +
                              tiltAxis * tiltAxis.dot(viewL) * (1 - std::cos(tiltAngle));
    n_init = -n_init.normalized();  // 法向量指向相机
    
    // 3. 半径初始值：考虑倾斜角度的修正
    // r = a * z / (f * cos(α))，但如果接近正对则直接用 a * z / f
    double cosAlpha = std::max(0.3, axisRatioL);  // 防止除以接近0的值
    double f = (stereo_.K_L(0, 0) + stereo_.K_L(1, 1)) / 2.0;
    double r_init = ellipseL.a * C_init.z() / (f * std::sqrt(cosAlpha));
    
    // 限制半径范围
    r_init = std::clamp(r_init, params_.minRadius, params_.maxRadius);
    
    return Circle3D(C_init, n_init, r_init);
}

Circle3D CircleOptimizer::optimizeEllipseParams(const Circle3D& init,
                                                 const Ellipse& ellipseL,
                                                 const Ellipse& ellipseR,
                                                 int& iterations,
                                                 bool& converged) {
    // Levenberg-Marquardt 优化
    Eigen::Matrix<double, 6, 1> params = circleToParams(init);
    double lambda = params_.lambdaInit;
    double prevCost = computeCost(init, ellipseL, ellipseR);
    
    converged = false;
    
    for (iterations = 0; iterations < params_.maxIter; ++iterations) {
        Circle3D current = paramsToCircle(params);
        
        // 计算雅可比矩阵和残差
        Eigen::Matrix<double, 10, 6> J;
        Eigen::Matrix<double, 10, 1> r;
        computeJacobianAndResidual(current, ellipseL, ellipseR, J, r);
        
        // LM 更新: (J^T J + λI) Δp = J^T r
        Eigen::Matrix<double, 6, 6> JtJ = J.transpose() * J;
        Eigen::Matrix<double, 6, 1> Jtr = J.transpose() * r;
        
        // 添加阻尼
        Eigen::Matrix<double, 6, 6> H = JtJ + lambda * Eigen::Matrix<double, 6, 6>::Identity();
        
        // 求解增量
        Eigen::Matrix<double, 6, 1> delta = H.ldlt().solve(Jtr);
        
        // 尝试更新
        Eigen::Matrix<double, 6, 1> newParams = params - delta;
        Circle3D newCircle = paramsToCircle(newParams);
        double newCost = computeCost(newCircle, ellipseL, ellipseR);
        
        if (newCost < prevCost) {
            // 接受更新
            params = newParams;
            prevCost = newCost;
            lambda *= params_.lambdaDown;
            
            // 检查收敛
            if (delta.norm() < params_.convergeTol) {
                converged = true;
                break;
            }
        } else {
            // 拒绝更新，增大阻尼
            lambda *= params_.lambdaUp;
            
            if (lambda > 1e10) {
                break;  // 阻尼过大，停止
            }
        }
    }
    
    return paramsToCircle(params);
}

Circle3D CircleOptimizer::refineWithEdgePoints(const Circle3D& init,
                                                const Ellipse& ellipseL,
                                                const Ellipse& ellipseR) {
    // TODO: 实现边缘点精化
    // 目前直接返回初始值
    return init;
}

double CircleOptimizer::computeCost(const Circle3D& circle,
                                     const Ellipse& ellipseL,
                                     const Ellipse& ellipseR) {
    // 在3D圆上采样点，投影到2D，计算到检测椭圆的距离
    const int numSamples = 12;  // 使用较少采样点以提高速度
    
    Eigen::Vector3d n = circle.normal.normalized();
    Eigen::Vector3d u, v;
    if (std::abs(n.x()) < 0.9) {
        u = Eigen::Vector3d::UnitX().cross(n).normalized();
    } else {
        u = Eigen::Vector3d::UnitY().cross(n).normalized();
    }
    v = n.cross(u);
    
    double totalCost = 0;
    
    for (int i = 0; i < numSamples; ++i) {
        double theta = 2 * M_PI * i / numSamples;
        Eigen::Vector3d pt3D = circle.center + circle.radius * (std::cos(theta) * u + std::sin(theta) * v);
        
        // 投影到左图
        Eigen::Vector4d pt_homo;
        pt_homo << pt3D, 1.0;
        Eigen::Vector3d projL = P_L_ * pt_homo;
        if (std::abs(projL(2)) > 1e-10) {
            cv::Point2d pt2dL(projL(0) / projL(2), projL(1) / projL(2));
            double distL = pointToEllipseDistance(pt2dL, ellipseL);
            totalCost += distL * distL;
        }
        
        // 投影到右图
        Eigen::Vector3d projR = P_R_ * pt_homo;
        if (std::abs(projR(2)) > 1e-10) {
            cv::Point2d pt2dR(projR(0) / projR(2), projR(1) / projR(2));
            double distR = pointToEllipseDistance(pt2dR, ellipseR);
            totalCost += distR * distR;
        }
    }
    
    return totalCost;
}

void CircleOptimizer::computeJacobianAndResidual(const Circle3D& circle,
                                                  const Ellipse& ellipseL,
                                                  const Ellipse& ellipseR,
                                                  Eigen::Matrix<double, 10, 6>& J,
                                                  Eigen::Matrix<double, 10, 1>& r) {
    // 采样10个点（左右各5个），计算残差和雅可比
    const int numSamplesPerSide = 5;
    
    Eigen::Vector3d n = circle.normal.normalized();
    Eigen::Vector3d u, v;
    if (std::abs(n.x()) < 0.9) {
        u = Eigen::Vector3d::UnitX().cross(n).normalized();
    } else {
        u = Eigen::Vector3d::UnitY().cross(n).normalized();
    }
    v = n.cross(u);
    
    // 计算当前残差
    r.setZero();
    for (int i = 0; i < numSamplesPerSide; ++i) {
        double theta = 2 * M_PI * i / numSamplesPerSide;
        Eigen::Vector3d pt3D = circle.center + circle.radius * (std::cos(theta) * u + std::sin(theta) * v);
        
        Eigen::Vector4d pt_homo;
        pt_homo << pt3D, 1.0;
        
        // 左图残差
        Eigen::Vector3d projL = P_L_ * pt_homo;
        if (std::abs(projL(2)) > 1e-10) {
            cv::Point2d pt2dL(projL(0) / projL(2), projL(1) / projL(2));
            r(i) = pointToEllipseDistance(pt2dL, ellipseL);
        }
        
        // 右图残差
        Eigen::Vector3d projR = P_R_ * pt_homo;
        if (std::abs(projR(2)) > 1e-10) {
            cv::Point2d pt2dR(projR(0) / projR(2), projR(1) / projR(2));
            r(numSamplesPerSide + i) = pointToEllipseDistance(pt2dR, ellipseR);
        }
    }
    
    // 数值求导
    const double eps = 1e-6;
    Eigen::Matrix<double, 6, 1> params = circleToParams(circle);
    
    for (int j = 0; j < 6; ++j) {
        Eigen::Matrix<double, 6, 1> paramsPert = params;
        paramsPert(j) += eps;
        Circle3D circPert = paramsToCircle(paramsPert);
        
        Eigen::Vector3d n_pert = circPert.normal.normalized();
        Eigen::Vector3d u_pert, v_pert;
        if (std::abs(n_pert.x()) < 0.9) {
            u_pert = Eigen::Vector3d::UnitX().cross(n_pert).normalized();
        } else {
            u_pert = Eigen::Vector3d::UnitY().cross(n_pert).normalized();
        }
        v_pert = n_pert.cross(u_pert);
        
        Eigen::Matrix<double, 10, 1> r_pert;
        r_pert.setZero();
        
        for (int i = 0; i < numSamplesPerSide; ++i) {
            double theta = 2 * M_PI * i / numSamplesPerSide;
            Eigen::Vector3d pt3D = circPert.center + circPert.radius * (std::cos(theta) * u_pert + std::sin(theta) * v_pert);
            
            Eigen::Vector4d pt_homo;
            pt_homo << pt3D, 1.0;
            
            Eigen::Vector3d projL = P_L_ * pt_homo;
            if (std::abs(projL(2)) > 1e-10) {
                cv::Point2d pt2dL(projL(0) / projL(2), projL(1) / projL(2));
                r_pert(i) = pointToEllipseDistance(pt2dL, ellipseL);
            }
            
            Eigen::Vector3d projR = P_R_ * pt_homo;
            if (std::abs(projR(2)) > 1e-10) {
                cv::Point2d pt2dR(projR(0) / projR(2), projR(1) / projR(2));
                r_pert(numSamplesPerSide + i) = pointToEllipseDistance(pt2dR, ellipseR);
            }
        }
        
        J.col(j) = (r_pert - r) / eps;
    }
}

Eigen::Matrix<double, 6, 1> CircleOptimizer::circleToParams(const Circle3D& circle) {
    // 参数化: [Cx, Cy, Cz, theta, phi, r]
    // 其中 normal = [sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)]
    
    Eigen::Matrix<double, 6, 1> params;
    params(0) = circle.center.x();
    params(1) = circle.center.y();
    params(2) = circle.center.z();
    
    // 从法向量提取球坐标
    double theta = std::acos(std::clamp(circle.normal.z(), -1.0, 1.0));
    double phi = std::atan2(circle.normal.y(), circle.normal.x());
    
    params(3) = theta;
    params(4) = phi;
    params(5) = circle.radius;
    
    return params;
}

Circle3D CircleOptimizer::paramsToCircle(const Eigen::Matrix<double, 6, 1>& params) {
    Eigen::Vector3d center(params(0), params(1), params(2));
    
    double theta = params(3);
    double phi = params(4);
    Eigen::Vector3d normal(
        std::sin(theta) * std::cos(phi),
        std::sin(theta) * std::sin(phi),
        std::cos(theta)
    );
    
    double radius = std::abs(params(5));  // 确保半径为正
    
    return Circle3D(center, normal, radius);
}

double CircleOptimizer::computeReprojError(const Circle3D& circle, 
                                            const Ellipse& ellipse,
                                            bool isLeft) {
    // 投影3D圆
    Ellipse proj;
    if (isLeft) {
        proj = projectCircleToEllipse(circle, stereo_.K_L, P_L_);
    } else {
        proj = projectCircleToEllipse(circle, stereo_.K_R, P_R_);
    }
    
    // 计算中心距离
    double dx = proj.center.x - ellipse.center.x;
    double dy = proj.center.y - ellipse.center.y;
    return std::sqrt(dx * dx + dy * dy);
}

std::vector<Eigen::Vector3d> CircleOptimizer::sampleEdgePoints(const Circle3D& circle) {
    // 构造圆平面内的正交基
    Eigen::Vector3d n = circle.normal.normalized();
    Eigen::Vector3d u, v;
    
    if (std::abs(n.x()) < 0.9) {
        u = Eigen::Vector3d::UnitX().cross(n).normalized();
    } else {
        u = Eigen::Vector3d::UnitY().cross(n).normalized();
    }
    v = n.cross(u);
    
    // 采样360个点
    std::vector<Eigen::Vector3d> points(params_.numEdgePoints);
    for (int i = 0; i < params_.numEdgePoints; ++i) {
        double theta = 2 * M_PI * i / params_.numEdgePoints;
        points[i] = circle.center + circle.radius * (std::cos(theta) * u + std::sin(theta) * v);
    }
    
    return points;
}

bool CircleOptimizer::passEpipolarCheck(const Ellipse& ellipseL, 
                                         const Ellipse& ellipseR) {
    Eigen::Vector2d centerL(ellipseL.center.x, ellipseL.center.y);
    Eigen::Vector3d epiline = computeEpilineR(centerL, stereo_.F);
    Eigen::Vector2d centerR(ellipseR.center.x, ellipseR.center.y);
    
    double dist = pointToEpilineDistance(centerR, epiline);
    return dist < params_.epipolarThresh;
}

bool CircleOptimizer::passAreaRatioCheck(const Ellipse& ellipseL, 
                                          const Ellipse& ellipseR) {
    double areaL = M_PI * ellipseL.a * ellipseL.b;
    double areaR = M_PI * ellipseR.a * ellipseR.b;
    
    if (areaL < 1e-6 || areaR < 1e-6) return false;
    
    double ratio = areaL / areaR;
    return ratio >= params_.areaRatioMin && ratio <= params_.areaRatioMax;
}

bool CircleOptimizer::passShapeCheck(const Ellipse& ellipseL, 
                                      const Ellipse& ellipseR) {
    // 长短轴比相似
    double ratioL = ellipseL.b / ellipseL.a;
    double ratioR = ellipseR.b / ellipseR.a;
    
    return std::abs(ratioL - ratioR) < params_.axisRatioTol;
}

} // namespace stereo
