// #include <ros/ros.h>
// #include <image_transport/image_transport.h>
// #include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
// #include <face_feature/FacePicMsg.h>
// #include <face_feature/DriverIdMsg.h>
#include "font/CvxText.h"
#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <string.h>

#include <queue>
#include <map>
#include <string>
#include <vector>

// #include "./pfld/pfld.h"
#include "./arcface/arcface.h"
// #include "data/results.h"
// #include "database/SqliteOp.h"
// #include "utils/faceId_match.h"
#ifndef DEVICE
    #define DEVICE 0
#endif

using std::cout;
using std::endl;
using std::map;
using std::ifstream;

// string _path = "/home/nvidia/wdq/ros_vision/src/face_feature/src/data/";     //存.txt文件的路径

const std::string suffix = ".jpg";
const std::string separator = "_";
// string cap_path = "/home/nvidia/Pictures/face_cap/";    //待识别特征的人脸图片的路径
// string cap_pic_name;
std::string face_data_path = "/home/nvidia/Documents/face_data/";
std::string map_path = "/home/nvidia/Documents/face_map/";

// ros::Publisher DriverInfoPub;

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

//a字符串减去b字符串，返回值存在result字符串，从尾部截去，ge:"123.jpg"-".jpg"='123'
inline void StringSub(const std::string &a, const std::string &b, std::string &result)
{
    int pos = a.find(b);
    if(pos != std::string::npos)
    {
        result = a.substr(0, pos);
    }
    else
    {
        cout << "string_b is not a substr of string_a." << endl;
    }
}

//拆分人名和id
inline void getIDName(const std::string &a, const std::string &separator, int &id, std::string &name)
{
    int pos = a.find(separator);
    if(pos != std::string::npos)
    {
        id = std::stoi(a.substr(0, pos).c_str());
        name = (a.substr(pos)).erase(0,separator.size());  //必须是非常量字符串
    }
    else
    {
        cout << "can  not find separator." << endl;
    }
}

// void imageCallback(const face_feature::FacePicMsg::ConstPtr& msg)
// {
//     //sensor_msgs::Image ROS中image传递的消息形式
//     try
//     {
//         boost::shared_ptr<void const> tracked_object;    //共享指针,原来初始化了：boost::shared_ptr<void const> tracked_object(&(msg->FaceImage))
//         cv::Mat faceMat = cv_bridge::toCvShare(msg->FaceImage, tracked_object,"bgr8")->image;
        
//         cv::imshow("view2", faceMat);    
//         cv::waitKey(3); 
        
//         if(msg->id > 0){
//             ROS_INFO("hasFace");
//         }else{
//             ROS_INFO("no face");
//         }

//         //成功截取人脸
//         if(msg->id > 0 ){
//             //get tmp face feature
//             memset(tmpFaceFeature, 0, sizeof(tmpFaceFeature));
//             //printVector(tmpFaceFeature);
//             ArcFace::GetFaceFeature(faceMat, tmpFaceFeature);
//             // printVector(tmpFaceFeature);
//             cout << "id:    " << msg->id << endl;
            
//             // //do inference get face id
//             // ArcFace::DetectFaceID(faceMat, faceId, tmpFaceFeature);
//             // cout << "Face-----------------------------------------ID: " << faceId << endl << endl;
//             // matchMaker.addFaceId(faceId);
//             // // //得到人脸关键点
//             // // PFLD::AnalyzeOneFace(faceMat, prob);   // 正式使用时，改为faceMat//使用前要初始化引擎
//             // // driverResult.DealFaceResult(prob);
//             // // cout << "prob" << (*prob)<< endl;
//         }
//         else{
//             if(msg->id == -1)    //图片传输完毕，结束回调
//             {
//                 return ;
//             }
//             else                            //截取的照片不可用
//             {
//                 cout << "skip id: "<< msg->id << endl;
//             }
//             //没找到可识别的人脸目标，或者追踪失败
//             // DoFaceIDTimes = 0;
//             // faceId = 0;
//             // driverResult.ResetPointState();
//             // driver_msg.isDriver = false;
//             // driver_msg.driverID = faceId;
//             // matchMaker.addFaceId(0);
//         }

//         // bool getSuccess = matchMaker.detectFaceId(faceId);
//         // if(getSuccess)
//         // {
//         //     driver_msg.isDriver = true;
//         //     if (faceId == 0)
//         //     {
//         //         driver_msg.isDriver = false;
//         //     }
//         //     driver_msg.driverID = faceId;
//         //     DriverInfoPub.publish(driver_msg);
//         // }
//     }
//     catch (cv_bridge::Exception& e)
//     {
//         ROS_ERROR("Could not convert from image to 'bgr8'.");
//     }
// }

int main(int argc, char **argv)
{
    // ros::init(argc, argv, "image_identify_node");
    // ros::NodeHandle node_generate;

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
    map<int, std::string> id_name_map;
    std::string picture_path;

    //路径
    if(argc < 2)
    {
        //std::cout << " 请传入3个地址参数，分别为存储照片路径，存储id映射表的路径，存储face.data文件的路径。"  << std::endl;
        std::cout << " 请至少传入1个地址参数：存储照片路径。"  << std::endl;
        return -2;
    }
    switch (argc)
    {
    case 2:
    {
        picture_path = argv[1]; //第二个参数开始
        break;
    }
    case 3:
    {
        picture_path = argv[1]; 
        map_path = argv[2];        
        break;
    }
    case 4:
    {
        picture_path = argv[1]; 
        map_path = argv[2];       
        face_data_path = argv[3];          
        break;
    }
    default:
    {
        cout << "wrong: too much parametor." << endl;
        break;
    }
    }


    //判断输入地址末尾必须为/
    if(*(picture_path.rbegin())!= '/' || *(map_path.rbegin())!= '/'|| *(face_data_path.rbegin())!= '/' )
    {    
        // cout << *(picture_path.rbegin()) << endl;
        cout << "The path must end with '/', please input a new path." << endl;
        return -1;
    }
    cout  << "picture path: "  << picture_path<< endl;
    cout << "map path: " << map_path<< endl;
    cout << "face data  path: " << face_data_path << endl;
    std::vector<cv::String> picture_names; 
    cv::glob(picture_path, picture_names);
    std::string name;    //存单个人名;
    int id;                 //照片编号

    //read pictures

    for(size_t i = 0; i < picture_names.size(); ++i)
    {
        // std::cout<<picture_names[i]<<std::endl;
        //照片正常读取
        cv::Mat src = cv::imread(picture_names[i]);
        if(!src.data)
        {
            std::cerr << "Problem loading image!!!" << std::endl;            
        }

        name = picture_names[i];
        name.erase(0, picture_path.size());     //去掉名字中的路径
        StringSub(name, suffix, name);
        getIDName(name, separator, id, name);
        // cout << "No: " << id << "   Name: " << name << endl;
        
        if(id >= ArcFace::GALLARY_NUM)
        {
            cout << "id: " << id << "is out of range." << endl;
        }
        else
        {
             if(id_name_map.find(id) != id_name_map.end())
            {
                cout << "insert wrong. the id already exists." << endl;
            }
            else
            {
                id_name_map[id] = name;
                // cout << id << ": " << id_name_map[id] << endl;
                //add to gallaryData
                memset(tmpFaceFeature, 0, sizeof(tmpFaceFeature));
                //printVector(tmpFaceFeature);
                ArcFace::GetFaceFeature(src, tmpFaceFeature);
                memcpy((gallaryData+id*(ArcFace::FACE_FEATURE_DIMENSION)), tmpFaceFeature, (ArcFace::FACE_FEATURE_DIMENSION) * sizeof(float));
                // //test 
                // printVector(gallaryData+id*(ArcFace::FACE_FEATURE_DIMENSION));
            }
        }
    }

    //写map到txt
    std::ofstream ous(map_path + "face_map.txt");
    for(map<int,std::string>::iterator iter = id_name_map.begin(); iter != id_name_map.end(); ++iter){
        // cout<<"key:"<<iter->first<<" value:"<<iter->second<<endl;
        ous << iter->first << " "<< iter->second << endl;
    }

    //写gallaryData到本地face.data
    face_data_path = face_data_path + "face.data";
    std::ofstream outFStream(face_data_path.c_str(), std::ios::binary);
    if(!outFStream){
        std::cout << "Failed to open face data file." << std::endl;
    }
    else{
        std::cout<< "Face data file  was opened successfully." << std::endl;
    }
    outFStream.write((char *) &gallaryData, sizeof(gallaryData));
    std::cout<< "Face data file  was written successfully." << std::endl;
    outFStream.close();

    // std::vector<cv::String> picture_names;
    // cv::glob(picture_path, picture_names);
    // const std::string txt = "face_cap_name.txt";
    // const std::string suffix = ".jpg";
    // unsigned int suffix_len = suffix.length();
    // ifstream infile;
    // infile.open(txt_path + txt);   //将文件流对象与文件连接起来
    // if(infile.is_open())
    // {
    //         cout << "face_pic_name.txt opened successfully." << endl;
    // }
    //assert(infile.is_open());   //若失败,则输出错误消息,并终止程序
    // string line;                            //每次读取一行
    // cv::Mat img;                        //存储读到的本地照片
    // while(getline(infile, line))
    // {
    //     cap_pic_name = line.substr(0, line.length() - suffix_len);
    //     num = atoi(cap_pic_name.c_str());
    //     cap_pic_name = cap_pic_name + ".jpg";
    //     img = cv::imread(cap_path + cap_pic_name, 1); 
    //     memset(tmpFaceFeature, 0, sizeof(tmpFaceFeature));
    //     //printVector(tmpFaceFeature);
    //     ArcFace::GetFaceFeature(img, tmpFaceFeature);
    //     memcpy((gallaryData+num*(ArcFace::FACE_FEATURE_DIMENSION)), tmpFaceFeature, (ArcFace::FACE_FEATURE_DIMENSION) * sizeof(float));
    //     // //test 
    //     // printVector(gallaryData+num*(ArcFace::FACE_FEATURE_DIMENSION));
    //     cout << "read " << cap_pic_name << " successfully." << endl;
    // }

    // //写到本地
    // std::ofstream outFStream(face_data_filename.c_str(), std::ios::binary);
    // if(!outFStream){
    //     std::cout << "Failed to open face data file." << std::endl;
    // }
    // else{
    //     std::cout<< "Face data file opened successfully." << std::endl;
    // }
    // outFStream.write((char *) &gallaryData, sizeof(gallaryData));
    // outFStream.close();

    ArcFace::ReleasePFLDEngine();       //一样的名字
    return 0;

}
