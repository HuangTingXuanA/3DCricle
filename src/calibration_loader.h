#pragma once

/**
 * @file calibration_loader.h
 * @brief 标定文件加载器
 */

#include "stereo_geometry.h"
#include <string>

namespace stereo {

/**
 * @brief 从 YAML 文件加载双目相机标定参数
 * 
 * 支持 OpenCV FileStorage 格式的 YAML 文件
 * 
 * @param yamlPath YAML 文件路径
 * @param params 输出的双目参数结构体
 * @return 成功返回 true
 */
bool loadCalibration(const std::string& yamlPath, StereoParams& params);

} // namespace stereo
