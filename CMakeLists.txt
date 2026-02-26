cmake_minimum_required(VERSION 3.16)
project(ellipse_detector LANGUAGES CXX)

# C++17标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 查找Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# 源文件
set(SOURCES
    src/ellipse_detector.cpp
    src/dsf.cpp
    src/main.cpp
    src/stereo_geometry.cpp
    src/calibration_loader.cpp
    src/circle_optimizer.cpp
)

# 头文件
set(HEADERS
    src/ellipse_detector.h
    src/dsf.h
    src/param.h
    src/stereo_geometry.h
    src/calibration_loader.h
    src/circle_optimizer.h
)

# 可执行目标
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# 链接库
target_link_libraries(${PROJECT_NAME} 
    ${OpenCV_LIBS}
    Eigen3::Eigen
)

# 编译选项
target_compile_options(${PROJECT_NAME} PRIVATE
    -Wall -Wextra -O2
)
