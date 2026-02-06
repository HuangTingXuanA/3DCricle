/**
 * @file stereo_geometry.cpp
 * @brief 双目几何核心模块实现
 */

#include "stereo_geometry.h"
#include <cmath>
#include <iostream>
#include <Eigen/Eigenvalues>

namespace stereo {

// ============================================================================
// StereoParams 成员函数
// ============================================================================

void StereoParams::computeFundamentalMatrix() {
    // 反对称矩阵 [T]_x
    Eigen::Matrix3d T_skew;
    T_skew << 0, -T(2), T(1),
              T(2), 0, -T(0),
              -T(1), T(0), 0;
    
    // 本质矩阵 E = [T]_x * R
    Eigen::Matrix3d E = T_skew * R;
    
    // 基础矩阵 F = K_R^{-T} * E * K_L^{-1}
    F = K_R.transpose().inverse() * E * K_L.inverse();
    
    // 归一化
    F /= F(2, 2);
}

Eigen::Matrix<double, 3, 4> StereoParams::getProjectionMatrixL() const {
    Eigen::Matrix<double, 3, 4> P;
    P.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    P.col(3).setZero();
    return K_L * P;
}

Eigen::Matrix<double, 3, 4> StereoParams::getProjectionMatrixR() const {
    Eigen::Matrix<double, 3, 4> P;
    P.block<3, 3>(0, 0) = R;
    P.col(3) = T;
    return K_R * P;
}

// ============================================================================
// 核心几何函数
// ============================================================================

Eigen::Matrix4d buildDualQuadric(const Eigen::Vector3d& center,
                                  const Eigen::Vector3d& normal,
                                  double radius) {
    // 3D圆的对偶二次曲面构造
    // 原理：圆 = 球体 ∩ 平面
    // - 球体: |X-C|² = r²
    // - 平面: n·(X-C) = 0
    
    Eigen::Vector3d n = normal.normalized();
    double d = n.dot(center);

    // 1. 构造经过圆心、半径为r的球体的对偶二次曲面 S*
    // 球体方程: (X-C)^T(X-C) = r² => X^T X - 2C^T X + C^T C - r² = 0
    // 齐次形式矩阵:
    // S = [ I,    -C   ]
    //     [-C^T, C·C-r²]
    // 对偶形式 S* = adj(S)，对于球体 S* ∝ S^{-1}，但直接用S的形式
    // 实际上对于圆的投影，我们需要的是：
    Eigen::Matrix4d S_star = Eigen::Matrix4d::Identity();
    S_star.block<3, 1>(0, 3) = -center;
    S_star.block<1, 3>(3, 0) = -center.transpose();
    S_star(3, 3) = center.dot(center) - radius * radius;

    // 2. 构造圆所在平面的齐次表示 π = [n; -d]
    Eigen::Vector4d pi;
    pi << n, -d;
    
    // 3. 圆的对偶二次曲面 Q*
    // 要求：Q* 是秩3的退化二次曲面，且满足 Q* π = 0
    // 通过从 S* 中减去其在 π 方向上的分量实现：
    // Q* = S* - (1/λ) * (S* π)(S* π)^T
    // 其中 λ = π^T S* π
    
    Eigen::Vector4d S_star_pi = S_star * pi;
    double lambda = pi.dot(S_star_pi);
    
    Eigen::Matrix4d Q_star = S_star;
    if (std::abs(lambda) > 1e-10) {
        Q_star = S_star - (1.0 / lambda) * S_star_pi * S_star_pi.transpose();
    }
    
    return Q_star;
}

Ellipse projectCircleToEllipse(const Circle3D& circle,
                                const Eigen::Matrix3d& K,
                                const Eigen::Matrix<double, 3, 4>& P) {
    // 使用边界点采样投影法：
    // 1. 在3D圆边界上均匀采样点
    // 2. 将采样点投影到2D
    // 3. 用投影点拟合椭圆
    // 
    // 这种方法比对偶二次曲面更稳定可靠
    
    Eigen::Vector3d n = circle.normal.normalized();
    Eigen::Vector3d u, v;
    if (std::abs(n.x()) < 0.9) {
        u = Eigen::Vector3d::UnitX().cross(n).normalized();
    } else {
        u = Eigen::Vector3d::UnitY().cross(n).normalized();
    }
    v = n.cross(u);
    
    // 采样36个边界点
    const int numSamples = 36;
    std::vector<cv::Point2f> points2D;
    points2D.reserve(numSamples);
    
    for (int i = 0; i < numSamples; ++i) {
        double theta = 2 * M_PI * i / numSamples;
        Eigen::Vector3d pt3D = circle.center + circle.radius * (std::cos(theta) * u + std::sin(theta) * v);
        
        // 投影到2D
        Eigen::Vector4d pt_homo;
        pt_homo << pt3D, 1.0;
        Eigen::Vector3d pt2D_homo = P * pt_homo;
        
        if (std::abs(pt2D_homo(2)) > 1e-10) {
            float px = static_cast<float>(pt2D_homo(0) / pt2D_homo(2));
            float py = static_cast<float>(pt2D_homo(1) / pt2D_homo(2));
            points2D.emplace_back(px, py);
        }
    }
    
    if (points2D.size() < 5) {
        return Ellipse(cv::Point2d(0, 0), 1.0, 1.0, 0.0);
    }
    
    // 用OpenCV拟合椭圆
    cv::RotatedRect fitted = cv::fitEllipse(points2D);
    
    double a = fitted.size.width / 2.0;
    double b = fitted.size.height / 2.0;
    double phi = fitted.angle * M_PI / 180.0;
    
    // 确保 a >= b
    if (a < b) {
        std::swap(a, b);
        phi += M_PI / 2.0;
    }
    
    // 归一化角度
    while (phi < 0) phi += M_PI;
    while (phi >= M_PI) phi -= M_PI;
    
    return Ellipse(cv::Point2d(fitted.center.x, fitted.center.y), a, b, phi);
}

Ellipse conicMatrixToEllipse(const Eigen::Matrix3d& C_star) {
    // 对偶二次曲线 C* 直接包含椭圆信息
    // 对于椭圆中心和半轴，可以直接从 C* 的结构提取
    //
    // C* 的形式为:
    // C* = [M,     m  ]
    //      [m^T,   c  ]
    //
    // 椭圆中心: center = -M^{-1} * m / c' (归一化到齐次坐标)
    // 使用 C* 的共焦矩阵 (adjugate) 方法
    
    // 归一化使最大元素为1以提高数值稳定性
    double maxVal = C_star.cwiseAbs().maxCoeff();
    if (maxVal < 1e-15) {
        return Ellipse(cv::Point2d(0, 0), 1.0, 1.0, 0.0);
    }
    Eigen::Matrix3d C = C_star / maxVal;
    
    // 对于对偶圆锥 C*，椭圆中心是 C* 的最后一行/列对应的点
    // 使用特征分解找到中心
    //
    // 另一种方法：C* 的相应点形式需要计算伴随矩阵
    // C = adj(C*) = det(C*) * (C*)^{-1}
    //
    // 但对于圆锥投影，C* 可能是退化的，需要特殊处理
    
    double det = C.determinant();
    
    // 如果 C* 可逆，转换为点形式
    if (std::abs(det) > 1e-12) {
        // 标准情况：C* 满秩
        Eigen::Matrix3d C_point = C.inverse();
        
        // 归一化
        C_point = C_point / C_point(2, 2);
        C_point = (C_point + C_point.transpose()) / 2.0;
        
        double A = C_point(0, 0);
        double B = 2 * C_point(0, 1);
        double C_coef = C_point(1, 1);
        double D = 2 * C_point(0, 2);
        double E = 2 * C_point(1, 2);
        double F = C_point(2, 2);
        
        double discriminant = B * B - 4 * A * C_coef;
        
        // 临时调试
        static int conicDebug = 0;
        if (conicDebug < 3) {
            std::cout << "  [Conic " << conicDebug << "] A=" << A << " B=" << B << " C_coef=" << C_coef 
                      << " disc=" << discriminant << "\n";
            std::cout << "    D=" << D << " E=" << E << " F=" << F << "\n";
        }
        
        if (discriminant >= 0) {
            if (conicDebug < 3) std::cout << "    FAIL: disc >= 0\n";
            conicDebug++;
            return Ellipse(cv::Point2d(0, 0), 1.0, 1.0, 0.0);
        }
        
        double cx = (2 * C_coef * D - B * E) / (-discriminant);
        double cy = (2 * A * E - B * D) / (-discriminant);
        
        Eigen::Matrix2d M;
        M << A, B/2, B/2, C_coef;
        
        double G = A*cx*cx + B*cx*cy + C_coef*cy*cy + D*cx + E*cy + F;
        if (G >= 0) {
            if (conicDebug < 3) std::cout << "    FAIL: G=" << G << " >= 0\n";
            conicDebug++;
            return Ellipse(cv::Point2d(0, 0), 1.0, 1.0, 0.0);
        }
        if (conicDebug < 3) {
            std::cout << "    cx=" << cx << " cy=" << cy << " G=" << G << "\n";
            conicDebug++;
        }
        
        Eigen::Matrix2d M_norm = M / (-G);
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(M_norm);
        Eigen::Vector2d eigenvalues = solver.eigenvalues();
        Eigen::Matrix2d eigenvectors = solver.eigenvectors();
        
        if (eigenvalues(0) <= 0 || eigenvalues(1) <= 0) {
            return Ellipse(cv::Point2d(0, 0), 1.0, 1.0, 0.0);
        }
        
        double a = 1.0 / std::sqrt(eigenvalues(0));
        double b_axis = 1.0 / std::sqrt(eigenvalues(1));
        
        double phi;
        if (a < b_axis) {
            std::swap(a, b_axis);
            phi = std::atan2(eigenvectors(1, 0), eigenvectors(0, 0));
        } else {
            phi = std::atan2(eigenvectors(1, 1), eigenvectors(0, 1));
        }
        
        while (phi < 0) phi += M_PI;
        while (phi >= M_PI) phi -= M_PI;
        
        return Ellipse(cv::Point2d(cx, cy), a, b_axis, phi);
    }
    
    // C* 接近奇异：使用 SVD 找到零空间
    // 这对应于退化的锥面（点锥）
    // 零空间向量 v 满足 C* v = 0，v 就是椭圆中心的齐次坐标
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(C, Eigen::ComputeFullV);
    Eigen::Vector3d v = svd.matrixV().col(2);  // 最小奇异值对应的向量
    
    if (std::abs(v(2)) < 1e-12) {
        return Ellipse(cv::Point2d(0, 0), 1.0, 1.0, 0.0);
    }
    
    double cx = v(0) / v(2);
    double cy = v(1) / v(2);
    
    // 从奇异值估计半轴
    // 对于退化情况，使用前两个奇异值
    Eigen::Vector3d singularValues = svd.singularValues();
    double a = std::sqrt(singularValues(0) / maxVal);
    double b_axis = std::sqrt(singularValues(1) / maxVal);
    if (a < b_axis) std::swap(a, b_axis);
    
    // 简化：假设角度为0
    return Ellipse(cv::Point2d(cx, cy), a, b_axis, 0.0);
}

Eigen::Matrix3d ellipseToConicMatrix(const Ellipse& ellipse) {
    double cx = ellipse.center.x;
    double cy = ellipse.center.y;
    double a = ellipse.a;
    double b = ellipse.b;
    double phi = ellipse.phi;
    
    double cos_phi = std::cos(phi);
    double sin_phi = std::sin(phi);
    double a2 = a * a;
    double b2 = b * b;
    
    // 二次曲线系数
    double A = cos_phi * cos_phi / a2 + sin_phi * sin_phi / b2;
    double B = 2 * cos_phi * sin_phi * (1.0 / a2 - 1.0 / b2);
    double C = sin_phi * sin_phi / a2 + cos_phi * cos_phi / b2;
    double D = -2 * A * cx - B * cy;
    double E = -B * cx - 2 * C * cy;
    double F = A * cx * cx + B * cx * cy + C * cy * cy - 1;
    
    Eigen::Matrix3d M;
    M << A, B / 2, D / 2,
         B / 2, C, E / 2,
         D / 2, E / 2, F;
    
    return M;
}

Eigen::Vector3d triangulatePoint(const Eigen::Vector2d& pt_L,
                                  const Eigen::Vector2d& pt_R,
                                  const StereoParams& stereo) {
    // DLT 三角测量
    Eigen::Matrix<double, 3, 4> P_L = stereo.getProjectionMatrixL();
    Eigen::Matrix<double, 3, 4> P_R = stereo.getProjectionMatrixR();
    
    // 构造 A 矩阵
    Eigen::Matrix4d A;
    A.row(0) = pt_L(0) * P_L.row(2) - P_L.row(0);
    A.row(1) = pt_L(1) * P_L.row(2) - P_L.row(1);
    A.row(2) = pt_R(0) * P_R.row(2) - P_R.row(0);
    A.row(3) = pt_R(1) * P_R.row(2) - P_R.row(1);
    
    // SVD 求解
    Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullV);
    Eigen::Vector4d X_homo = svd.matrixV().col(3);
    
    // 齐次坐标转欧氏坐标
    return X_homo.head<3>() / X_homo(3);
}

double pointToEpilineDistance(const Eigen::Vector2d& pt,
                               const Eigen::Vector3d& epiline) {
    // 极线 ax + by + c = 0
    double a = epiline(0);
    double b = epiline(1);
    double c = epiline(2);
    
    // 归一化
    double norm = std::sqrt(a * a + b * b);
    if (norm < 1e-10) return std::numeric_limits<double>::max();
    
    return std::abs(a * pt(0) + b * pt(1) + c) / norm;
}

Eigen::Vector3d computeEpilineR(const Eigen::Vector2d& pt_L,
                                 const Eigen::Matrix3d& F) {
    // l_R = F * p_L
    Eigen::Vector3d pt_L_homo(pt_L(0), pt_L(1), 1.0);
    return F * pt_L_homo;
}

Eigen::Vector3d computeEpilineL(const Eigen::Vector2d& pt_R,
                                 const Eigen::Matrix3d& F) {
    // l_L = F^T * p_R
    Eigen::Vector3d pt_R_homo(pt_R(0), pt_R(1), 1.0);
    return F.transpose() * pt_R_homo;
}

Eigen::Vector2d undistortPoint(const Eigen::Vector2d& pt,
                                const Eigen::Matrix3d& K,
                                const Eigen::Matrix<double, 1, 5>& dist) {
    // 提取畸变系数
    double k1 = dist(0, 0);
    double k2 = dist(0, 1);
    double p1 = dist(0, 2);
    double p2 = dist(0, 3);
    double k3 = dist(0, 4);
    
    // 归一化坐标
    double fx = K(0, 0), fy = K(1, 1);
    double cx = K(0, 2), cy = K(1, 2);
    
    double x = (pt(0) - cx) / fx;
    double y = (pt(1) - cy) / fy;
    
    // 迭代去畸变
    double x0 = x, y0 = y;
    for (int i = 0; i < 10; ++i) {
        double r2 = x * x + y * y;
        double r4 = r2 * r2;
        double r6 = r4 * r2;
        
        double radial = 1 + k1 * r2 + k2 * r4 + k3 * r6;
        double dx = 2 * p1 * x * y + p2 * (r2 + 2 * x * x);
        double dy = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y;
        
        x = (x0 - dx) / radial;
        y = (y0 - dy) / radial;
    }
    
    // 返回像素坐标
    return Eigen::Vector2d(x * fx + cx, y * fy + cy);
}

double ellipseParamError(const Ellipse& e1, const Ellipse& e2,
                          const Eigen::Vector3d& weights) {
    // 中心误差
    double center_err = std::pow(e1.center.x - e2.center.x, 2) + 
                        std::pow(e1.center.y - e2.center.y, 2);
    
    // 轴长误差
    double axis_err = std::pow(e1.a - e2.a, 2) + std::pow(e1.b - e2.b, 2);
    
    // 角度误差 (考虑周期性)
    double angle_diff = e1.phi - e2.phi;
    while (angle_diff > M_PI / 2) angle_diff -= M_PI;
    while (angle_diff < -M_PI / 2) angle_diff += M_PI;
    double angle_err = angle_diff * angle_diff;
    
    return weights(0) * center_err + weights(1) * axis_err + weights(2) * angle_err;
}

double pointToEllipseDistance(const cv::Point2d& pt, const Ellipse& ellipse) {
    // 转换到椭圆中心坐标系
    double dx = pt.x - ellipse.center.x;
    double dy = pt.y - ellipse.center.y;
    
    // 旋转到椭圆主轴坐标系
    double cos_phi = std::cos(ellipse.phi);
    double sin_phi = std::sin(ellipse.phi);
    double x = dx * cos_phi + dy * sin_phi;
    double y = -dx * sin_phi + dy * cos_phi;
    
    // 椭圆方程: x²/a² + y²/b² = 1
    // F(x,y) = x²/a² + y²/b² - 1
    double a2 = ellipse.a * ellipse.a;
    double b2 = ellipse.b * ellipse.b;
    
    if (a2 < 1e-10 || b2 < 1e-10) {
        return 1e10;  // 无效椭圆
    }
    
    double F = x*x / a2 + y*y / b2 - 1.0;
    
    // 梯度: ∇F = (2x/a², 2y/b²)
    double grad_x = 2.0 * x / a2;
    double grad_y = 2.0 * y / b2;
    double grad_norm = std::sqrt(grad_x * grad_x + grad_y * grad_y);
    
    // Sampson距离 = |F| / ||∇F||
    if (grad_norm < 1e-10) {
        return std::abs(F);  // 退化情况（中心点）
    }
    
    return std::abs(F) / grad_norm;
}

} // namespace stereo
