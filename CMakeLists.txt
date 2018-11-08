cmake_minimum_required(VERSION 3.7)
project(yolov3)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV)

add_executable(yolov3 src/yolov3.cpp src/yolov3.h)

include_directories(${OpenCV_INCLUDE_DIRS}
                    /home/leo/anaconda2/envs/yolo/include/python3.5m)
                   # /home/leo/anaconda2/envs/yolo/lib/python3.5/site-packages/tensorflow/include/tensorflow)

link_directories(/home/leo/anaconda2/envs/yolo/lib/python3.5)

target_link_libraries(yolov3 ${OpenCV_LIBS}
                             /home/leo/anaconda2/envs/yolo/lib/libpython3.5m.so)
                             #/home/leo/anaconda2/envs/yolo/lib/python3.5/site-packages/tensorflow/libtensorflow_framework.so)
