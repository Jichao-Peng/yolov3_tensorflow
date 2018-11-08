//
// Created by leo on 18-11-6.
//

#ifndef YOLOV3_YOLOV3_H
#define YOLOV3_YOLOV3_H

#include <Python.h>
#include <numpy/arrayobject.h>
//#include <opencv2/opencv.hpp>
//#include <opencv2/core.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <stdlib.h>
#include <malloc.h>


//using namespace cv;
using namespace std;

class TransMethod;

class YOLOSystem
{
    struct DetectResult
    {
        string mName;
        float mConfidence;
        vector<float> mvBoudingBox;
    };

public:
    YOLOSystem();
    ~YOLOSystem();
    void Test();
//    vector<YOLOSystem::DetectResult> Detect(Mat Image);

private:
    PyObject *mpDict;
    PyObject *mpModule;
    PyObject *mpClass;
    PyObject *mpInstance;
//    PyByteArrayObject* TransMat2Array(Mat Image);
};


#endif //YOLOV3_YOLOV3_H
