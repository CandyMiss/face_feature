cmake_minimum_required(VERSION 3.0.2)
project(face_feature)
set(PROJ_NAME face_feature)


option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)
#set(CMAKE_BUILD_TYPE Release)

find_package(CUDA REQUIRED)

set(CUDA_NVCC_PLAGS ${CUDA_NVCC_PLAGS};-std=c++11;-g;-G;-gencode;arch=compute_30;code=sm_30)

include_directories(/usr/include)
include_directories(/usr/include/glib-2.0)
include_directories(/usr/lib/glib-2.0/include)
# 区分嵌入式平台环境
include_directories(${PROJECT_SOURCE_DIR}/include)
if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    message("embed_platform on")
    include_directories(/usr/local/cuda/targets/aarch64-linux/include)
    link_directories(/usr/local/cuda/targets/aarch64-linux/lib)
    # link_directories(/usr/lib/aarch64-linux-gnu)    # 字体渲染库，libfreetype的位置
    include_directories(/usr/include/gstreamer-1.0)
    include_directories(/usr/lib/aarch64-linux-gnu/glib-2.0/include)
else()
    message("embed_platform off")
    include_directories(/usr/local/cuda/include)
    link_directories(/usr/local/cuda/lib64)
    # link_directories(/usr/lib/x86_64-linux-gnu/)
    # include_directories(/usr/lib/x86_64-linux-gnu/gstreamer-1.0)
    # include_directories(/usr/lib/x86_64-linux-gnu/gstreamer-1.0/include)
    # include_directories(/usr/lib/x86_64-linux-gnu/glib-2.0)
    # include_directories(/usr/lib/x86_64-linux-gnu/glib-2.0/include)
endif()

# 为了防止对 #pragma 语句的报警，加上 -Wno-unknown-pragmas 设置
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wno-unknown-pragmas -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

# 不明白工程里的问题，这里凡是使用了 CUDA代码，都要编译为动态链接库才能使用
# cuda_add_library(yololayer SHARED src/yolov5/yololayer.cu)
cuda_add_library(face_similarity SHARED ${PROJECT_SOURCE_DIR} src/arcface/face_similarity.cu)

add_executable(Generater src/generateFeature.cpp
        # src/utils/cfg_params.h src/utils/cfg_params.cpp
        # src/data/results.h src/data/results.cpp
        # src/pfld/pfld.h src/pfld/pfld.cpp
        # src/pfld/face_gesture.h src/pfld/face_gesture.cpp
        src/arcface/prelu.h src/arcface/prelu.cu
        src/arcface/arcface.cpp src/arcface/arcface.h
        # src/arcface/draw_mask.cpp src/arcface/draw_mask.h
        # src/yolov5/common.h src/yolov5/common.cpp
        # src/utils/faceId_match.h src/utils/faceId_match.cpp
        src/arcface/face_similarity.cu)


find_package(OpenCV)
include_directories(OpenCV_INCLUDE_DIRS)
target_link_libraries(Generater ${OpenCV_LIBS})


include_directories(/usr/lib)
# include_directories(/usr/lib/glib-2.0/include)
# include_directories(/usr/lib/x86_64-linux-gnu -lgio-2.0)

target_link_libraries(Generater nvinfer)
target_link_libraries(Generater cudart)
# target_link_libraries(Generater yololayer)
target_link_libraries(Generater face_similarity)
# target_link_libraries(Generater sqlite3)

include_directories(/usr/include/freetype2)
include_directories(/usr/include/libxml2)
target_link_libraries(Generater xml2)


add_definitions(-O2 -pthread)
