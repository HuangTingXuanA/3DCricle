/**
 * @file circle_optimizer.cpp
 * @brief 3D圆重建优化器实现
 */

#include "circle_optimizer.h"
#include <iostream>
#include <algorithm>
#include <cmath>

namespace stereo {

CircleOptimizer::CircleOptimizer(const StereoParams& stereo, const OptimizerParams& params)
    : stereo_(stereo), params_(params) {
    // Ensure stereo params are initialized
    // stereo_.init(); // Assuming caller did this or we do it here safely
    // Since init() is non-const but stereo_ is const member, we assume it's pre-initialized.
    // In strict C++, we should receive fully initialized params.
}

std::vector<ReconstructionResult> CircleOptimizer::reconstruct(
    const std::vector<Ellipse>& ellipsesL,
    const std::vector<Ellipse>& ellipsesR
) {
    std::vector<ReconstructionResult> candidates;
    
    // 1. Sparse Matching & Hypothesis Generation
    for (size_t i = 0; i < ellipsesL.size(); ++i) {
        const auto& eL = ellipsesL[i];
        
        // Compute epiline for eL center
        // Pre-filter R candidates? 
        // For simplicity, double loop. O(N*M) is fine for N,M < 100.
        
        for (size_t j = 0; j < ellipsesR.size(); ++j) {
            const auto& eR = ellipsesR[j];
            
            // Epipolar constraint
            double epiDist = distanceFromEpiline(
                Eigen::Vector2d(eR.center.x, eR.center.y),
                Eigen::Vector2d(eL.center.x, eL.center.y),
                stereo_.F
            );
            
            if (epiDist > params_.epipolarThresh) continue;
            
            // Shape similarity constraint (optional but helpful)
            double ratioL = eL.b / eL.a;
            double ratioR = eR.b / eR.a;
            if (std::abs(ratioL - ratioR) > params_.shapeSimilarityThresh) continue;
            
            // Process this pair
            ReconstructionResult res = processPair(eL, i, eR, j);
            if (res.valid()) {
                candidates.push_back(res);
            }
        }
    }
    
    // 2. Global Selection (Greedy Strategy)
    // Sort by score (descending)
    std::sort(candidates.begin(), candidates.end(), 
        [](const ReconstructionResult& a, const ReconstructionResult& b) {
            return a.score > b.score; // Higher score is better
        });
        
    std::vector<ReconstructionResult> finalResults;
    std::vector<bool> usedL(ellipsesL.size(), false);
    std::vector<bool> usedR(ellipsesR.size(), false);
    
    for (const auto& cand : candidates) {
        if (usedL[cand.ellipseIdxL] || usedR[cand.ellipseIdxR]) continue;
        
        finalResults.push_back(cand);
        usedL[cand.ellipseIdxL] = true;
        usedR[cand.ellipseIdxR] = true;
    }
    
    return finalResults;
}

ReconstructionResult CircleOptimizer::processPair(
    const Ellipse& eL, int idxL,
    const Ellipse& eR, int idxR
) {
    ReconstructionResult bestResult;
    bestResult.score = -1.0;
    bestResult.ellipseIdxL = idxL;
    bestResult.ellipseIdxR = idxR;
    
    // Hypothesis Generation Strategy:
    // 1. Triangulate centers roughly
    Eigen::Vector3d centerRough = triangulate(
        Eigen::Vector2d(eL.center.x, eL.center.y),
        Eigen::Vector2d(eR.center.x, eR.center.y),
        stereo_
    );
    
    // 2. Estimate normals from Left Image (2 candidates)
    auto [nL1, nL2] = estimateCircleNormal(eL, stereo_.K_L);
    
    // 3. Estimate normals from Right Image (2 candidates) 
    // Need to transform back to World (Left Camera) Frame
    auto [nR1_cam, nR2_cam] = estimateCircleNormal(eR, stereo_.K_R);
    Eigen::Vector3d nR1 = stereo_.R.transpose() * nR1_cam;
    Eigen::Vector3d nR2 = stereo_.R.transpose() * nR2_cam;
    
    // 4. Test all 4 normals
    std::vector<Eigen::Vector3d> normals = {nL1, nL2, nR1, nR2};
    
    // Also add "Facing Camera" normal as fallback?
    // normals.push_back(-centerRough.normalized());
    
    for (const auto& n : normals) {
        // Solve Radius
        // We assume Center is roughly correct from triangulation.
        // Actually, for a tilted circle, the center of ellipse != projection of circle center.
        // But for small tilt or far distance, it's close.
        // Iterating center could be an enhancement.
        
        // Optimize Radius
        double r = optimizeRadius(centerRough, n, eL, eR, stereo_); // You need to implement this in stereo_geometry
        
        if (r < params_.minRadius || r > params_.maxRadius) continue;
        
        Circle3D circle(centerRough, n, r);
        
        // Verify & Evaluate
        double error = 0;
        double score = evaluate(circle, eL, eR, error);
        
        if (score > params_.minScore && score > bestResult.score) {
            bestResult.circle = circle;
            bestResult.score = score;
            bestResult.reprojError = error;
        }
    }
    
    return bestResult;
}

double CircleOptimizer::evaluate(const Circle3D& circle, const Ellipse& eL, const Ellipse& eR, double& outError) {
    // Reprojection Error in Left
    Ellipse pL = projectCircle(circle, stereo_.K_L, stereo_.P_L);
    double errL = ellipseParamError(eL, pL); // Need this helper, or use Sampson
    
    // Reprojection Error in Right
    Ellipse pR = projectCircle(circle, stereo_.K_R, stereo_.P_R);
    double errR = ellipseParamError(eR, pR); // Need this helper
    
    // Or use Sampson Distance on sampled points (more robust)
    // Let's use Sampson on 8 points
    /*
    double sampL = 0, sampR = 0;
    // ... sample points on eL, compute distance to pL? No, other way around.
    // Error is difference between Observed and Projected.
    */
    
    // Simplified metric: intersection Over Union (IoU) of bounding boxes?
    // Or just center distance + axis difference.
    
    // Let's use simple center distance + axis length difference
    auto diff = [](const Ellipse& a, const Ellipse& b) {
        double dC = cv::norm(a.center - b.center);
        double dA = std::abs(a.a - b.a);
        double dB = std::abs(a.b - b.b);
        return dC + dA + dB;
    };
    
    errL = diff(eL, pL);
    errR = diff(eR, pR);
    
    outError = (errL + errR) / 2.0;
    
    if (outError > params_.maxReprojError) return 0.0;
    
    // Score: inverse of error, normalized
    return std::exp(-outError); // 1.0 is perfect, decay fast
}

} // namespace stereo
