/**
 * @file calibration_loader.cpp
 * @brief 标定文件加载器实现
 */

#include "calibration_loader.h"
#include <opencv2/core.hpp>
#include <iostream>

namespace stereo {

bool loadCalibration(const std::string& yamlPath, StereoParams& params) {
    cv::FileStorage fs(yamlPath, cv::FileStorage::READ);
    
    if (!fs.isOpened()) {
        std::cerr << "[CalibLoader] 无法打开标定文件: " << yamlPath << std::endl;
        return false;
    }
    
    try {
        // 读取左相机内参
        cv::Mat K_L_cv;
        fs["camera_matrix_left"] >> K_L_cv;
        if (K_L_cv.empty()) {
            std::cerr << "[CalibLoader] 缺少 camera_matrix_left" << std::endl;
            return false;
        }
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                params.K_L(i, j) = K_L_cv.at<double>(i, j);
            }
        }
        
        // 读取右相机内参
        cv::Mat K_R_cv;
        fs["camera_matrix_right"] >> K_R_cv;
        if (K_R_cv.empty()) {
            std::cerr << "[CalibLoader] 缺少 camera_matrix_right" << std::endl;
            return false;
        }
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                params.K_R(i, j) = K_R_cv.at<double>(i, j);
            }
        }
        
        // 读取左相机畸变系数 (支持多种长度)
        cv::Mat dist_L_cv;
        fs["dist_coeffs_left"] >> dist_L_cv;
        params.dist_L.setZero();
        if (!dist_L_cv.empty()) {
            int n = std::min(static_cast<int>(dist_L_cv.total()), 5);
            for (int i = 0; i < n; ++i) {
                params.dist_L(0, i) = dist_L_cv.at<double>(i);
            }
        }
        
        // 读取右相机畸变系数
        cv::Mat dist_R_cv;
        fs["dist_coeffs_right"] >> dist_R_cv;
        params.dist_R.setZero();
        if (!dist_R_cv.empty()) {
            int n = std::min(static_cast<int>(dist_R_cv.total()), 5);
            for (int i = 0; i < n; ++i) {
                params.dist_R(0, i) = dist_R_cv.at<double>(i);
            }
        }
        
        // 读取旋转矩阵
        cv::Mat R_cv;
        fs["R"] >> R_cv;
        if (R_cv.empty()) {
            std::cerr << "[CalibLoader] 缺少 R" << std::endl;
            return false;
        }
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                params.R(i, j) = R_cv.at<double>(i, j);
            }
        }
        
        // 读取平移向量
        cv::Mat T_cv;
        fs["T"] >> T_cv;
        if (T_cv.empty()) {
            std::cerr << "[CalibLoader] 缺少 T" << std::endl;
            return false;
        }
        for (int i = 0; i < 3; ++i) {
            params.T(i) = T_cv.at<double>(i);
        }
        
        // 计算基础矩阵
        params.init();
        
        fs.release();
        
        std::cout << "[CalibLoader] 成功加载标定参数" << std::endl;
        std::cout << "  左相机焦距: (" << params.K_L(0, 0) << ", " << params.K_L(1, 1) << ")" << std::endl;
        std::cout << "  右相机焦距: (" << params.K_R(0, 0) << ", " << params.K_R(1, 1) << ")" << std::endl;
        std::cout << "  基线长度: " << params.T.norm() << " mm" << std::endl;
        
        return true;
        
    } catch (const cv::Exception& e) {
        std::cerr << "[CalibLoader] OpenCV 异常: " << e.what() << std::endl;
        return false;
    }
}

} // namespace stereo
