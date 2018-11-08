//
// Created by leo on 18-11-6.
//

#include "yolov3.h"

YOLOSystem::YOLOSystem()
{
    //Python初始化
    Py_Initialize();
    _import_array();

    //链接到YOLO对应的python接口文件上去
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('/home/leo/Desktop/yolov3_project/src/yolov3_tf/yolov3_tf/')");
    //导入python脚本
    mpModule = PyImport_ImportModule("yolov3");//导入Python脚本
    if (!mpModule)
    {
        cout << "Load python module failed" << endl;
    } else
    {
        cout << "Load python module successed!" << endl;
    }

    //导入python脚本中的类
    mpDict = PyModule_GetDict(mpModule);
    mpClass = PyDict_GetItemString(mpDict, "YOLO");
    mpInstance = PyInstanceMethod_New(mpClass);
    if (!PyInstanceMethod_Check(mpInstance))
    {
        cout << "Load python instance failed" << endl;
    } else
    {
        cout << "Load python instance successed!" << "\n";
    }
}

YOLOSystem::~YOLOSystem()
{
    Py_Finalize();
}

void YOLOSystem::Test()
{
    PyObject* pArg = PyTuple_New(0);
    PyObject_CallMethodObjArgs(mpInstance, Py_BuildValue("s","Test"), pArg, NULL);
}

PyByteArrayObject* YOLOSystem::TransMat2Array(Mat Image)
{
    cvtColor(Image,Image,CV_RGB2BGR);//讲RGB转化成BGR（python里面的图片格式）
    npy_intp ImageShape[1]= {Image.rows * Image.cols * 3};
    char* Data = new char[ImageShape[0]];
    memcpy(Data,Image.data,Image.total() * Image.elemSize() * sizeof(char));
    PyByteArrayObject *pArray = reinterpret_cast<PyByteArrayObject*>(PyArray_SimpleNewFromData(1, ImageShape, NPY_BYTE, reinterpret_cast<void *>(Data)));

    return pArray;
}

//检测图片
vector<YOLOSystem::DetectResult> YOLOSystem::Detect(Mat Image)
{
    PyByteArrayObject *pArray = TransMat2Array(Image);//讲Mat转化成传输到python的格式
    cout<<"Finish trans"<<endl;
    //申请python入参
    PyObject* pArg = PyTuple_New(1);
    //对python入参进行赋值
    PyTuple_SetItem(pArg, 0, reinterpret_cast<PyObject*>(pArray));
    //执行函数
    PyObject *pResult = PyObject_CallMethodObjArgs(mpInstance, Py_BuildValue("s","DetectImage"), pArg, NULL);

    if(pResult)
    {
        cout<<"Get result"<<endl;
        int ResultSize = (int)PyList_Size(pResult);
        cout<<ResultSize<<endl;
        PyObject* pClass = PyList_GetItem(pResult,0);
        int ClassSize = (int)PyTuple_Size(pClass);
        cout<<ClassSize<<endl;
        PyObject* pClassName = PyTuple_GetItem(pClass,0);
        //String Name = PyString_AsString(pClassName);
        //cout<<Name<<endl;
    }
}

int main()
{
    YOLOSystem YOLO;
    YOLO.Test();
    Mat mat = imread("/home/leo/Desktop/yolov3_project/src/yolov3_tf/yolov3_tf/data/girl.jpeg");
    YOLO.Detect(mat);
}
