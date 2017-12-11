cmake_minimum_required(VERSION 2.8)

#プロジェクト名の指定
project(improved_shot_detection)

#Debug,Releaseの指定,MinSizeRelとRelWithDebInfoの選択肢を生成しない
set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "limited configs" FORCE)

#OpenCVのディレクトリの指定
set(OpenCV_DIR "C:/opencv-3.3.0/build")
include_directories("C:/opencv-3.3.0/build/include")

#パッケージの検索
find_package(OpenCV REQUIRED)

#インクルードパスの指定
include_directories( ${OpenCV_INCLUDE_DIRS})
add_executable(improved_shot_detection calc_block_histogram.cpp cut_detection_functions.cpp example_template_matching.cpp main.cpp improved_shot_detection.cpp test_calc_template_matching_diff.cpp)

#リリース版の設定
set(CMAKE_BUILD_TYPE Debug;Release)

#リンクするライブラリの追加
target_link_libraries(improved_shot_detection ${OpenCV_LIBS})