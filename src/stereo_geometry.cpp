/**
 * @file stereo_geometry.cpp
 * @brief 双目几何核心实现
 */

#include "stereo_geometry.h"
#include <cmath>
#include <iostream>
#include <algorithm>

namespace stereo {

void StereoParams::init() {
    // 1. Calculate Fundamental Matrix F
    // E = [T]x R
    Eigen::Matrix3d Tx;
    Tx << 0, -T(2), T(1),
          T(2), 0, -T(0),
          -T(1), T(0), 0;
    Eigen::Matrix3d E = Tx * R;
    F = K_R.transpose().inverse() * E * K_L.inverse();
    if (std::abs(F(2,2)) > 1e-6)
        F /= F(2,2); // Normalize if possible

    // 2. Calculate Projection Matrices P
    P_L.setIdentity(); // [I|0]
    P_L = K_L * P_L;
    
    P_R.block<3,3>(0,0) = R;
    P_R.col(3) = T;
    P_R = K_R * P_R;
}

std::pair<Eigen::Vector3d, Eigen::Vector3d> estimateCircleNormal(
    const Ellipse& ellipse, 
    const Eigen::Matrix3d& K) 
{
    // Algorithm:
    // 1. Convert ellipse to normalized coordinates.
    // 2. Compute eigen-decomposition of the quadratic form.
    // 3. Derive two possible plane normals.
    
    // Ellipse center and axes in image plane
    double cx = ellipse.center.x;
    double cy = ellipse.center.y;
    double a = ellipse.a;
    double b = ellipse.b;
    double phi = ellipse.phi;

    // Construct Conic Matrix C in image P^2
    // Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0
    double cphi = std::cos(phi);
    double sphi = std::sin(phi);
    double a2 = a*a;
    double b2 = b*b;

    // Prevent division by zero
    if (a2 < 1e-6) a2 = 1e-6;
    if (b2 < 1e-6) b2 = 1e-6;

    double A = cphi*cphi/a2 + sphi*sphi/b2;
    double B = 2*cphi*sphi*(1.0/a2 - 1.0/b2);
    double C_val = sphi*sphi/a2 + cphi*cphi/b2;
    double D = -2*A*cx - B*cy;
    double E = -B*cx - 2*C_val*cy;
    double F_val = A*cx*cx + B*cx*cy + C_val*cy*cy - 1.0;

    Eigen::Matrix3d C_mat;
    C_mat << A, B/2, D/2,
             B/2, C_val, E/2,
             D/2, E/2, F_val;

    // Back-project to normalized cone: Q = K^T * C * K
    Eigen::Matrix3d Q = K.transpose() * C_mat * K;

    // Eigendecomposition of Q (symmetric)
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(Q);
    // Eigenvalues are sorted in increasing order
    Eigen::Vector3d eval = eig.eigenvalues(); 
    Eigen::Matrix3d evec = eig.eigenvectors();

    // Since it's a cone from a circle, eigenvalues should have signs (+, +, -) or (-, -, +)
    // We sort them so that lambda1 * lambda2 > 0 and lambda3 has opposite sign
    // Let's assume (l1, l2, l3) are sorted.
    // If l1, l2 positive and l3 negative.
    
    // Zhang's method or similar analytic reconstruction
    // Here we use the simplified heuristic from previous attempt:
    // Rotate ViewDir by +/- theta around TiltAxis.
    
    Eigen::Vector3d viewDir = (K.inverse() * Eigen::Vector3d(cx, cy, 1.0)).normalized();
    
    // Approximate slant angle from aspect ratio
    double axisRatio = b / a;
    double theta = std::acos(std::max(0.01, std::min(0.99, axisRatio))); // Slant
    
    Eigen::Vector3d up_world = Eigen::Vector3d::UnitY();
    if(std::abs(viewDir.dot(up_world)) > 0.9) up_world = Eigen::Vector3d::UnitX();
    
    Eigen::Vector3d right = viewDir.cross(up_world).normalized();
    
    // Ellipse phi is angle from x-axis. 
    // Tilt Axis (Major Axis) direction in this basis:
    Eigen::Vector3d axis_cam(std::cos(phi), std::sin(phi), 0);
    // Orthogonalize w.r.t viewDir
    axis_cam = (axis_cam - axis_cam.dot(viewDir)*viewDir).normalized();
    
    // Roate ViewDir around axis_cam by theta
    Eigen::AngleAxisd rot1(theta, axis_cam);
    Eigen::AngleAxisd rot2(-theta, axis_cam);
    
    Eigen::Vector3d n1 = rot1 * (-viewDir); // Normal points towards camera
    Eigen::Vector3d n2 = rot2 * (-viewDir);
    
    return {n1, n2};
}

Ellipse projectCircle(const Circle3D& circle, const Eigen::Matrix3d& K, const Eigen::Matrix<double, 3, 4>& P) {
    // Sampling based method - Robust and First Principles
    // Project N points from 3D circle to 2D image and fit ellipse.
    
    Eigen::Vector3d center = circle.center;
    double r = circle.radius;
    Eigen::Vector3d n = circle.normal;
    
    // Basis on the circle plane
    Eigen::Vector3d u = n.cross(Eigen::Vector3d::UnitY());
    if (u.norm() < 0.1) u = n.cross(Eigen::Vector3d::UnitX());
    u.normalize();
    Eigen::Vector3d v = n.cross(u).normalized();
    
    std::vector<cv::Point2f> pts;
    for (int i=0; i<36; ++i) {
        double ang = 2*M_PI*i/36.0;
        Eigen::Vector3d p3 = center + r*(std::cos(ang)*u + std::sin(ang)*v);
        
        Eigen::Vector4d p4(p3.x(), p3.y(), p3.z(), 1.0);
        Eigen::Vector3d p2 = P * p4;
        
        if (p2.z() > 0.1) {
            pts.emplace_back(p2.x()/p2.z(), p2.y()/p2.z());
        }
    }
    
    if (pts.size() < 6) return Ellipse(); // Invalid
    
    cv::RotatedRect rr = cv::fitEllipse(pts);
    
    // Convert cv::RotatedRect to Ellipse struct
    // RotatedRect angle is degrees clockwise from vertical? Or horizontal?
    // OpenCV docs: "angle is the rotation of the rectangle from the first side ... clockwise"
    // Usually standardizing is tricky.
    
    return Ellipse(rr); // Assuming Ellipse has constructor from RotatedRect
}

double computeSampsonDistance(const cv::Point2d& pt, const Ellipse& ellipse) {
    // Translate point to ellipse center
    double x = pt.x - ellipse.center.x;
    double y = pt.y - ellipse.center.y;
    
    // Rotate to align with axes
    double c = std::cos(ellipse.phi);
    double s = std::sin(ellipse.phi);
    
    double u = x*c + y*s;
    double v = -x*s + y*c;
    
    double a = ellipse.a;
    double b = ellipse.b;
    
    if (a < 1e-3 || b < 1e-3) return 1e9;
    
    // F = u^2/a^2 + v^2/b^2 - 1
    double F = (u*u)/(a*a) + (v*v)/(b*b) - 1.0;
    
    // Grad F = [2u/a^2, 2v/b^2]
    double dFdu = 2*u/(a*a);
    double dFdv = 2*v/(b*b);
    
    double gradNormSq = dFdu*dFdu + dFdv*dFdv;
    
    if (gradNormSq < 1e-8) return std::abs(F);
    
    return std::abs(F) / std::sqrt(gradNormSq);
}

double ellipseParamError(const Ellipse& e1, const Ellipse& e2) {
    double dC = cv::norm(e1.center - e2.center);
    double dA = std::abs(e1.a - e2.a);
    double dB = std::abs(e1.b - e2.b);
    
    // Angle difference
    double dPhi = std::abs(e1.phi - e2.phi);
    while (dPhi > M_PI) dPhi -= M_PI; // Standardize to [0, PI) range diff
    if (dPhi > M_PI/2) dPhi = M_PI - dPhi;
    
    // Weight angle less
    return dC + dA + dB + dPhi * 10.0;
}

Eigen::Vector3d triangulate(const Eigen::Vector2d& ptL, const Eigen::Vector2d& ptR, const StereoParams& stereo) {
    // Simple Linear Triangulation DLT
    Eigen::Matrix4d A;
    
    // row = x*P.row(2) - P.row(0)
    // row = y*P.row(2) - P.row(1)
    
    A.row(0) = ptL.x() * stereo.P_L.row(2) - stereo.P_L.row(0);
    A.row(1) = ptL.y() * stereo.P_L.row(2) - stereo.P_L.row(1);
    A.row(2) = ptR.x() * stereo.P_R.row(2) - stereo.P_R.row(0);
    A.row(3) = ptR.y() * stereo.P_R.row(2) - stereo.P_R.row(1);
    
    Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullV);
    Eigen::Vector4d X = svd.matrixV().col(3);
    
    return X.head<3>() / X(3);
}

double distanceFromEpiline(const Eigen::Vector2d& ptR, const Eigen::Vector2d& ptL, const Eigen::Matrix3d& F) {
    Eigen::Vector3d pL(ptL.x(), ptL.y(), 1.0);
    Eigen::Vector3d line = F * pL; // Epiline in Right image
    
    double num = std::abs(line.x()*ptR.x() + line.y()*ptR.y() + line.z());
    double den = std::sqrt(line.x()*line.x() + line.y()*line.y());
    
    return (den > 1e-6) ? num/den : num;
}

double optimizeRadius(const Eigen::Vector3d& center, const Eigen::Vector3d& normal, 
                     const Ellipse& eL, const Ellipse& eR, const StereoParams& stereo) 
{
    // Analytical or simple search for radius?
    // Given Center and Normal, the transformation from 3D circle to 2D ellipse is fixed except radius.
    // 2D projection size scales linearly with radius approximately.
    
    // Estimate radius from Left image major axis
    // Project unit radius circle
    Circle3D unitC(center, normal, 1.0);
    Ellipse projL = projectCircle(unitC, stereo.K_L, stereo.P_L);
    
    double rL = 0;
    if (projL.a > 1e-3) rL = eL.a / projL.a;
    
    // Estimate from Right
    Ellipse projR = projectCircle(unitC, stereo.K_R, stereo.P_R);
    double rR = 0;
    if (projR.a > 1e-3) rR = eR.a / projR.a;
    
    if (rL > 0 && rR > 0) return (rL + rR) / 2.0;
    if (rL > 0) return rL;
    if (rR > 0) return rR;
    
    return 0.0;
}

std::vector<Eigen::Vector3d> sampleCircleEdgePoints(const Circle3D& circle, int numPoints) {
    std::vector<Eigen::Vector3d> points;
    points.reserve(numPoints);
    
    Eigen::Vector3d center = circle.center;
    double r = circle.radius;
    Eigen::Vector3d n = circle.normal.normalized();
    
    // Basis on the circle plane
    Eigen::Vector3d u = n.cross(Eigen::Vector3d::UnitY());
    if (u.norm() < 0.1) u = n.cross(Eigen::Vector3d::UnitX());
    u.normalize();
    Eigen::Vector3d v = n.cross(u).normalized();
    
    for (int i=0; i<numPoints; ++i) {
        double ang = 2 * M_PI * i / numPoints;
        Eigen::Vector3d pt = center + r * (std::cos(ang) * u + std::sin(ang) * v);
        points.push_back(pt);
    }
    
    return points;
}

} // namespace stereo
