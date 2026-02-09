#pragma once

/**
 * @file circle_optimizer.h
 * @brief 3D圆重建优化器 - 极简主义版
 * "Less is more."
 */

#include "stereo_geometry.h"
#include <vector>

namespace stereo {

struct OptimizerParams {
    double epipolarThresh = 3.0;      // 极线距离阈值 (pixels)
    double shapeSimilarityThresh = 0.5; // 形状相似度阈值 (axis ratio diff)
    double minRadius = 1.0;           // 最小半径 (mm)
    double maxRadius = 1000.0;        // 最大半径 (mm)
    double maxReprojError = 4.0;      // 最大重投影误差 (pixels)
    double minScore = 0.6;            // 最小置信度分数
};

struct ReconstructionResult {
    Circle3D circle;
    int ellipseIdxL;
    int ellipseIdxR;
    double score;
    double reprojError;
    
    bool valid() const { return score > 0; }
};

class CircleOptimizer {
public:
    CircleOptimizer(const StereoParams& stereo, const OptimizerParams& params = OptimizerParams());

    /**
     * @brief 执行重建
     * @param ellipsesL 左图椭圆
     * @param ellipsesR 右图椭圆
     * @return重建结果列表
     */
    std::vector<ReconstructionResult> reconstruct(
        const std::vector<Ellipse>& ellipsesL,
        const std::vector<Ellipse>& ellipsesR
    );

private:
    StereoParams stereo_;
    OptimizerParams params_;

    // Generate hypothesis for a pair of ellipses
    ReconstructionResult processPair(
        const Ellipse& eL, 
        int idxL,
        const Ellipse& eR,
        int idxR
    );
    
    // Evaluate a 3D circle hypothesis
    double evaluate(const Circle3D& circle, const Ellipse& eL, const Ellipse& eR, double& outError);
};

} // namespace stereo
