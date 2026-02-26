#include <iostream>
#include <chrono>
#include <iomanip>
#include <filesystem>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <algorithm>
#include <fstream>
#include "ellipse_detector.h"
#include "stereo_geometry.h"
#include "circle_optimizer.h"
#include "calibration_loader.h"
#include <opencv2/core/eigen.hpp>

namespace fs = std::filesystem;

// ============================================================================
// 可视化辅助函数
// ============================================================================

/**
 * 在图像上绘制检测到的椭圆（弧线和圆心）
 */
cv::Mat drawEllipses(const cv::Mat& image, const std::vector<Ellipse>& ellipses) {
    cv::Mat display;
    if (image.channels() == 1) {
        cv::cvtColor(image, display, cv::COLOR_GRAY2BGR);
    } else {
        display = image.clone();
    }
    
    std::vector<cv::Scalar> colors = {
        cv::Scalar(0, 255, 0),     // 绿色
        cv::Scalar(0, 0, 255),     // 红色
        cv::Scalar(255, 0, 0),     // 蓝色
        cv::Scalar(0, 255, 255),   // 黄色
        cv::Scalar(255, 0, 255),   // 品红
        cv::Scalar(255, 255, 0),   // 青色
    };
    
    for (size_t i = 0; i < ellipses.size(); i++) {
        const auto& e = ellipses[i];
        cv::Scalar color = colors[i % colors.size()];
        
        cv::ellipse(display, 
                    cv::Point(cvRound(e.center.x), cvRound(e.center.y)),
                    cv::Size(cvRound(e.a), cvRound(e.b)),
                    e.phi * 180.0 / CV_PI, 0, 360, color, 1);
        
        cv::circle(display, cv::Point(cvRound(e.center.x), cvRound(e.center.y)), 
                   2, color, -1);
        
        cv::putText(display, std::to_string(i+1),
                    cv::Point(cvRound(e.center.x - 15), cvRound(e.center.y - 5)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, color, 1);
    }
    
    return display;
}

// ============================================================================
// 3D圆重建结果保存
// ============================================================================

void saveReconstructResults(const std::string& outputPath,
                            const std::vector<stereo::ReconstructionResult>& results,
                            const std::string& imageName) {
    cv::FileStorage fs(outputPath, cv::FileStorage::WRITE);
    if (!fs.isOpened()) {
        std::cerr << "[Error] 无法创建输出文件: " << outputPath << std::endl;
        return;
    }
    
    fs << "image_name" << imageName;
    fs << "num_circles" << (int)results.size();
    
    fs << "circles" << "[";
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& res = results[i];
        fs << "{";
        fs << "id" << (int)i;
        fs << "center_x" << res.circle.center.x();
        fs << "center_y" << res.circle.center.y();
        fs << "center_z" << res.circle.center.z();
        fs << "normal_x" << res.circle.normal.x();
        fs << "normal_y" << res.circle.normal.y();
        fs << "normal_z" << res.circle.normal.z();
        fs << "radius" << res.circle.radius;
        fs << "error" << res.reprojError;
        fs << "score" << res.score;
        fs << "}";
    }
    fs << "]";
    
    fs.release();
}

void saveEdgePointsPLY(const std::string& outputPath,
                        const std::vector<stereo::ReconstructionResult>& results) {
    std::ofstream ofs(outputPath);
    if (!ofs.is_open()) {
        std::cerr << "[Error] 无法创建PLY文件: " << outputPath << std::endl;
        return;
    }
    
    int pointsPerCircle = 100;
    std::vector<Eigen::Vector3d> allPoints;
    std::vector<std::tuple<int, int, int>> pointColors;

    std::vector<std::tuple<int, int, int>> colors = {
        {255, 0, 0}, {0, 255, 0}, {0, 0, 255},
        {255, 255, 0}, {255, 0, 255}, {0, 255, 255}
    };
    
    for (size_t i = 0; i < results.size(); ++i) {
        auto pts = stereo::sampleCircleEdgePoints(results[i].circle, pointsPerCircle);
        auto color = colors[i % colors.size()];
        
        for (const auto& p : pts) {
            allPoints.push_back(p);
            pointColors.push_back(color);
        }
    }
    
    ofs << "ply\n";
    ofs << "format ascii 1.0\n";
    ofs << "element vertex " << allPoints.size() << "\n";
    ofs << "property float x\n";
    ofs << "property float y\n";
    ofs << "property float z\n";
    ofs << "property uchar red\n";
    ofs << "property uchar green\n";
    ofs << "property uchar blue\n";
    ofs << "end_header\n";
    
    for (size_t i = 0; i < allPoints.size(); ++i) {
        auto& pt = allPoints[i];
        auto& col = pointColors[i];
        ofs << std::fixed << std::setprecision(6)
            << pt.x() << " " << pt.y() << " " << pt.z() << " "
            << std::get<0>(col) << " " << std::get<1>(col) << " " << std::get<2>(col) << "\n";
    }
    
    ofs.close();
}

// ============================================================================
// 重投影可视化
// ============================================================================

cv::Mat drawReprojection(const cv::Mat& image,
                          const std::vector<Ellipse>& detectedEllipses,
                          const std::vector<stereo::ReconstructionResult>& results,
                          const stereo::StereoParams& stereo,
                          bool isLeft) {
    cv::Mat display;
    if (image.channels() == 1) {
        cv::cvtColor(image, display, cv::COLOR_GRAY2BGR);
    } else {
        display = image.clone();
    }
    
    // 绘制检测椭圆 (绿色)
    for (const auto& e : detectedEllipses) {
        cv::ellipse(display, 
                    cv::Point(cvRound(e.center.x), cvRound(e.center.y)),
                    cv::Size(cvRound(e.a), cvRound(e.b)),
                    e.phi * 180.0 / CV_PI, 0, 360, cv::Scalar(0, 255, 0), 1);
    }
    
    // 绘制重投影椭圆 (红色)
    Eigen::Matrix<double, 3, 4> P = isLeft ? stereo.P_L : stereo.P_R;
    Eigen::Matrix3d K = isLeft ? stereo.K_L : stereo.K_R;
    
    for (const auto& res : results) {
        if (!res.valid()) continue;
        
        Ellipse proj = stereo::projectCircle(res.circle, K, P);
        cv::ellipse(display,
                    cv::Point(cvRound(proj.center.x), cvRound(proj.center.y)),
                    cv::Size(cvRound(proj.a), cvRound(proj.b)),
                    proj.phi * 180.0 / CV_PI, 0, 360, cv::Scalar(0, 0, 255), 1);
        
        cv::circle(display, cv::Point(cvRound(proj.center.x), cvRound(proj.center.y)),
                   3, cv::Scalar(0, 0, 255), -1);
    }
    
    return display;
}

// ============================================================================
// 3D圆重建主函数
// ============================================================================

void reconstruct3DCircles(const std::string& inputDir, bool visualize, bool debug) {
    std::cout << "\n========== 3D圆重建模式 (Refactored) ==========\n";
    std::cout << "输入目录: " << inputDir << std::endl;
    
    fs::path basePath(inputDir);
    fs::path leftDir = basePath / "left";
    fs::path rightDir = basePath / "right";
    fs::path calibPath = basePath / "calibration.yaml";
    
    if (!fs::exists(leftDir) || !fs::exists(rightDir)) {
        std::cerr << "[Error] 目录必须包含 left/ 和 right/ 子目录" << std::endl;
        return;
    }
    
    if (!fs::exists(calibPath)) {
        std::cerr << "[Error] 未找到标定文件: " << calibPath << std::endl;
        return;
    }
    
    // 加载标定参数
    stereo::StereoParams stereoParams;
    if (!stereo::loadCalibration(calibPath.string(), stereoParams)) {
        std::cerr << "[Error] 标定文件加载失败" << std::endl;
        return;
    }
    stereoParams.init(); // 重要：初始化导出参数 P_L, P_R, F
    
    // 收集左图像文件
    std::vector<std::string> leftImages;
    for (const auto& entry : fs::directory_iterator(leftDir)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
                leftImages.push_back(entry.path().filename().string());
            }
        }
    }
    std::sort(leftImages.begin(), leftImages.end());
    
    if (leftImages.empty()) {
        std::cerr << "[Error] left/ 目录中没有图像文件" << std::endl;
        return;
    }
    
    std::cout << "找到 " << leftImages.size() << " 对图像\n";
    
    // 创建输出目录
    fs::path resultsDir = basePath / "results";
    fs::create_directories(resultsDir);
    
    // 初始化优化器
    stereo::OptimizerParams optParams;
    // 可以根据需要调整 optParams
    stereo::CircleOptimizer optimizer(stereoParams, optParams);
    
    // 检测参数
    DetectorParams detParams;
    detParams.debug = debug;
    // 确保检测足够敏感以产生候选项
    detParams.ellipse.minEllipseScore1 = 0.4; 
    
    // 显示窗口
    std::string windowName = "3D重建结果";
    if (visualize) {
        cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    }
    
    // 统计
    int totalCircles = 0;
    double totalTime = 0.0;
    
    // 处理每对图像
    for (size_t idx = 0; idx < leftImages.size(); ++idx) {
        const std::string& imgName = leftImages[idx];
        fs::path leftPath = leftDir / imgName;
        fs::path rightPath = rightDir / imgName;
        
        std::cout << "\n[" << (idx + 1) << "/" << leftImages.size() << "] " 
                  << imgName << std::endl;
        
        if (!fs::exists(rightPath)) {
            std::cerr << "  [Warning] 右图不存在，跳过" << std::endl;
            continue;
        }
        
        cv::Mat imageL = cv::imread(leftPath.string(), cv::IMREAD_GRAYSCALE);
        cv::Mat imageR = cv::imread(rightPath.string(), cv::IMREAD_GRAYSCALE);
        
        if (imageL.empty() || imageR.empty()) {
            std::cerr << "  [Warning] 无法加载图像，跳过" << std::endl;
            continue;
        }
        
        auto startTime = std::chrono::high_resolution_clock::now();
        
        // 在无畸变的相机模型下进行极线约束与几何重建
        cv::Mat cvK_L, cvD_L, cvK_R, cvD_R;
        cv::eigen2cv(stereoParams.K_L, cvK_L);
        cv::eigen2cv(stereoParams.dist_L, cvD_L);
        cv::eigen2cv(stereoParams.K_R, cvK_R);
        cv::eigen2cv(stereoParams.dist_R, cvD_R);
        
        cv::Mat undistL, undistR;
        cv::undistort(imageL, undistL, cvK_L, cvD_L);
        cv::undistort(imageR, undistR, cvK_R, cvD_R);
        
        // 椭圆检测 (在去畸变的图像上检测)
        std::vector<Ellipse> ellipsesL = detectEllipses(undistL, detParams);
        std::vector<Ellipse> ellipsesR = detectEllipses(undistR, detParams);
        
        std::cout << "  检测椭圆: 左=" << ellipsesL.size() 
                  << " 右=" << ellipsesR.size() << std::endl;
        
        // 3D重建
        std::vector<stereo::ReconstructionResult> results = 
            optimizer.reconstruct(ellipsesL, ellipsesR);
        
        auto endTime = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double, std::milli>(endTime - startTime).count();
        
        std::cout << "  重建圆数: " << results.size() 
                  << " 耗时: " << std::fixed << std::setprecision(1) << elapsed << " ms" << std::endl;
        
        for (size_t i = 0; i < results.size(); ++i) {
            const auto& res = results[i];
            std::cout << "    圆[" << i << "] 中心=(" 
                      << std::setprecision(2) << res.circle.center.x() << ", "
                      << res.circle.center.y() << ", " << res.circle.center.z()
                      << ") R=" << res.circle.radius << " mm"
                      << " Score=" << std::setprecision(3) << res.score << std::endl;
        }
        
        totalCircles += (int)results.size();
        totalTime += elapsed;
        
        // 保存结果
        fs::path yamlPath = resultsDir / (fs::path(imgName).stem().string() + "_circles.yaml");
        saveReconstructResults(yamlPath.string(), results, imgName);
        
        fs::path plyPath = resultsDir / (fs::path(imgName).stem().string() + "_edges.ply");
        saveEdgePointsPLY(plyPath.string(), results);
        
        // 可视化
        if (visualize) {
            cv::Mat visL = drawReprojection(undistL, ellipsesL, results, stereoParams, true);
            cv::Mat visR = drawReprojection(undistR, ellipsesR, results, stereoParams, false);
            
            cv::Mat combined;
            cv::hconcat(visL, visR, combined);
            
            std::ostringstream info;
            info << imgName << " | Circles: " << results.size() 
                 << " | Time: " << std::fixed << std::setprecision(1) << elapsed << "ms";
            cv::putText(combined, info.str(), cv::Point(10, 25),
                        cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 255), 2);
            cv::putText(combined, "Green=Detected, Red=Reprojected", cv::Point(10, 55),
                        cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 255), 2);
            
            fs::path visPath = resultsDir / (fs::path(imgName).stem().string() + "_vis.png");
            cv::imwrite(visPath.string(), combined);
            
            cv::resizeWindow(windowName, combined.size());
            cv::imshow(windowName, combined);
            int key = cv::waitKey(500);
            if (key == 27) break;
        }
    }
    
    std::cout << "\n========== 重建完成 ==========\n";
    std::cout << "处理图像对: " << leftImages.size() << std::endl;
    std::cout << "重建圆总数: " << totalCircles << std::endl;
    std::cout << "总耗时: " << std::fixed << std::setprecision(1) << totalTime << " ms" << std::endl;
    std::cout << "平均耗时: " << (totalTime / leftImages.size()) << " ms/对" << std::endl;
    std::cout << "结果保存在: " << resultsDir << std::endl;
    
    if (visualize) {
        cv::destroyAllWindows();
    }
}

// ============================================================================
// 主函数
// ============================================================================

void printUsage(const char* progName) {
    std::cout << "3D圆重建程序 (Refactored)\n" << std::endl;
    std::cout << "用法: " << progName << " -f <目录> [选项]\n" << std::endl;
    std::cout << "参数:" << std::endl;
    std::cout << "  -f, --folder <路径>  指定包含 left/, right/, calibration.yaml 的目录" << std::endl;
    std::cout << "  -v, --visualize      启用可视化显示" << std::endl;
    std::cout << "  -d, --debug          开启调试模式" << std::endl;
    std::cout << "  -h, --help           显示帮助信息\n" << std::endl;
    std::cout << "示例:" << std::endl;
    std::cout << "  " << progName << " -f laser_01          # 重建 laser_01 数据集" << std::endl;
    std::cout << "  " << progName << " -f laser_01 -v       # 重建并可视化" << std::endl;
}

int main(int argc, char** argv) {
    std::string inputDir;
    bool visualize = false;
    bool debug = false;
    
    // 解析命令行参数
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "-f" || arg == "--folder") {
            if (i + 1 < argc && argv[i+1][0] != '-') {
                inputDir = argv[++i];
            } else {
                std::cerr << "错误: -f 选项需要指定目录路径" << std::endl;
                printUsage(argv[0]);
                return 1;
            }
        } else if (arg == "-v" || arg == "--visualize") {
            visualize = true;
        } else if (arg == "-d" || arg == "--debug") {
            debug = true;
        } else {
            std::cerr << "错误: 未知参数 " << arg << std::endl;
            printUsage(argv[0]);
            return 1;
        }
    }
    
    if (inputDir.empty()) {
        std::cerr << "错误: 必须使用 -f 指定输入目录" << std::endl;
        printUsage(argv[0]);
        return 1;
    }
    
    reconstruct3DCircles(inputDir, visualize, debug);
    
    return 0;
}
