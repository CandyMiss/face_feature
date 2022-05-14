#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <face_feature/FacePicMsg.h>
#include <face_feature/DriverIdMsg.h>
#include "font/CvxText.h"
#include <iostream>
#include <stdlib.h>
#include <fstream>

#include <queue>
#include <map>
#include <string.h>
#include <vector>

#include "./pfld/pfld.h"
#include "./arcface/arcface.h"
#include "data/results.h"
#include "database/SqliteOp.h"
#include "utils/faceId_match.h"

using std::cout;
using std::endl;
using std::map;
using std::string;
using std::ifstream;

// string _path = "/home/nvidia/wdq/ros_vision/src/face_feature/src/data/";     //存.txt文件的路径
string txt_path;
string cap_path = "/home/nvidia/Pictures/face_cap/";    //待识别特征的人脸图片的路径
string cap_pic_name;
string face_data_filename = "/home/nvidia/Documents/face_data/face.data";

ros::Publisher DriverInfoPub;

static int faceId = 0;         //tmp人脸id
int num = 0;                        //照片编号

// 当前结果寄存处
float tmpFaceFeature[ArcFace::FACE_FEATURE_DIMENSION]{0.0};//512维特征向量

float gallaryData[(ArcFace::GALLARY_NUM )* (ArcFace::FACE_FEATURE_DIMENSION)]{0.0};    //存特征库

inline void printVector(float *tmpFaceFeature){ //用于辅助测试特征是否被捕捉
    for(int i=0; i<ArcFace::FACE_FEATURE_DIMENSION; ++i){
        cout << tmpFaceFeature[i] << " ";
    }
    cout << endl;
}

void imageCallback(const face_feature::FacePicMsg::ConstPtr& msg)
{
    //sensor_msgs::Image ROS中image传递的消息形式
    try
    {
        boost::shared_ptr<void const> tracked_object;    //共享指针,原来初始化了：boost::shared_ptr<void const> tracked_object(&(msg->FaceImage))
        cv::Mat faceMat = cv_bridge::toCvShare(msg->FaceImage, tracked_object,"bgr8")->image;
        
        cv::imshow("view2", faceMat);    
        cv::waitKey(3); 
        
        if(msg->id > 0){
            ROS_INFO("hasFace");
        }else{
            ROS_INFO("no face");
        }

        //成功截取人脸
        if(msg->id > 0 ){
            //get tmp face feature
            memset(tmpFaceFeature, 0, sizeof(tmpFaceFeature));
            //printVector(tmpFaceFeature);
            ArcFace::GetFaceFeature(faceMat, tmpFaceFeature);
            // printVector(tmpFaceFeature);
            cout << "id:    " << msg->id << endl;
            
            // //do inference get face id
            // ArcFace::DetectFaceID(faceMat, faceId, tmpFaceFeature);
            // cout << "Face-----------------------------------------ID: " << faceId << endl << endl;
            // matchMaker.addFaceId(faceId);
            // // //得到人脸关键点
            // // PFLD::AnalyzeOneFace(faceMat, prob);   // 正式使用时，改为faceMat//使用前要初始化引擎
            // // driverResult.DealFaceResult(prob);
            // // cout << "prob" << (*prob)<< endl;
        }
        else{
            if(msg->id == -1)    //图片传输完毕，结束回调
            {
                return ;
            }
            else                            //截取的照片不可用
            {
                cout << "skip id: "<< msg->id << endl;
            }
            //没找到可识别的人脸目标，或者追踪失败
            // DoFaceIDTimes = 0;
            // faceId = 0;
            // driverResult.ResetPointState();
            // driver_msg.isDriver = false;
            // driver_msg.driverID = faceId;
            // matchMaker.addFaceId(0);
        }

        // bool getSuccess = matchMaker.detectFaceId(faceId);
        // if(getSuccess)
        // {
        //     driver_msg.isDriver = true;
        //     if (faceId == 0)
        //     {
        //         driver_msg.isDriver = false;
        //     }
        //     driver_msg.driverID = faceId;
        //     DriverInfoPub.publish(driver_msg);
        // }
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("Could not convert from image to 'bgr8'.");
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "image_identify_node");
    ros::NodeHandle node_generate;

    //防止崩溃，所有模型引擎优先初始化
    cudaSetDevice(DEVICE);
    ArcFace::InitArcFaceEngine();
    cout << "ArcFace 引擎序列化完成"  << std::endl;
    // PFLD::InitPFLDEngine();
    // cout << "PFLD 引擎序列化完成" <<  std::endl;
    //  // 初始化所有人脸数据
    // ArcFace::ReadFaceDataToGPU();
    // cout << "初始化人脸数据完成" <<  std::endl;   

    // //1.1消息机制，更新人脸图片
    // cv::namedWindow("view2",cv::WINDOW_NORMAL); 
    // ros::Subscriber sub = node_generate.subscribe("/face_picture", 1, imageCallback);    
    // ros::spin();   
    // cv::destroyWindow("view2");    //窗口

    //1.2直接读取本地的人脸照片
    txt_path = argv[1]; //第二个参数开始
    cout  << "argv[1]:  " << argv[1] << endl;

    // std::vector<cv::String> picture_names;
    // cv::glob(picture_path, picture_names);
    const std::string txt = "face_cap_name.txt";
    const std::string suffix = ".jpg";
    unsigned int suffix_len = suffix.length();
    ifstream infile;
    infile.open(txt_path + txt);   //将文件流对象与文件连接起来
    if(infile.is_open())
    {
            cout << "face_pic_name.txt opened successfully." << endl;
    }
    //assert(infile.is_open());   //若失败,则输出错误消息,并终止程序
    string line;                            //每次读取一行
    cv::Mat img;                        //存储读到的本地照片
    while(getline(infile, line))
    {
        cap_pic_name = line.substr(0, line.length() - suffix_len);
        num = atoi(cap_pic_name.c_str());
        cap_pic_name = cap_pic_name + ".jpg";
        img = cv::imread(cap_path + cap_pic_name, 1); 
        memset(tmpFaceFeature, 0, sizeof(tmpFaceFeature));
        //printVector(tmpFaceFeature);
        ArcFace::GetFaceFeature(img, tmpFaceFeature);
        memcpy((gallaryData+num*(ArcFace::FACE_FEATURE_DIMENSION)), tmpFaceFeature, (ArcFace::FACE_FEATURE_DIMENSION) * sizeof(float));
        // //test 
        // printVector(gallaryData+num*(ArcFace::FACE_FEATURE_DIMENSION));
        cout << "read " << cap_pic_name << " successfully." << endl;
    }

    //写到本地
    std::ofstream outFStream(face_data_filename.c_str(), std::ios::binary);
    if(!outFStream){
        std::cout << "Failed to open face data file." << std::endl;
    }
    else{
        std::cout<< "Face data file opened successfully." << std::endl;
    }
    outFStream.write((char *) &gallaryData, sizeof(gallaryData));
    outFStream.close();

    ArcFace::ReleasePFLDEngine();       //一样的名字
    return 0;

}
