#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <face_feature/FacePicMsg.h>
#include <face_feature/FaceRecMsg.h>

#include "yolov5/yolov5.h"
#include "data/results.h"
#include "database/SqliteOp.h"

using namespace std;
using namespace std::chrono::_V2;

ros::Publisher FacePicPub;

ros::Time CurAnalyzeStamp;
ros::Publisher StampInfoPub;
ros::Publisher FaceInfoPub;

DriverResult CurFaceResult;

string cap_path = "/home/nvidia/Pictures/face_cap/";    //保存截取人脸图片的路径
string pic_name;    //图片名

void imageCallback(const face_feature::FacePicMsg::ConstPtr& msg)
 {
    //sensor_msgs::Image ROS中image传递的消息形式
    try
    {
        if(msg->id == -1)
        {
            cout << "-1,finish" << endl;           
            return;           
        }
        else if(msg->id == 1)
        {
            cout << "1" << endl;
        }

        boost::shared_ptr<void const> tracked_object;    //共享指针,原来初始化了：boost::shared_ptr<void const> tracked_object(&(msg->FaceImage))
        cv::Mat frame = cv_bridge::toCvShare(msg->FaceImage, tracked_object,"bgr8")->image;
        face_feature::FacePicMsg face_msg;

        CurFaceResult.FaceCaptured = false;
//         // result_msg.FaceGesture = CurFaceResult.FaceCaptured;

        // 得到分析结果
        vector<Yolo::Detection> result = YoloV5::AnalyzeOneShot(frame); //返回存储bundingbox的vector
        CurFaceResult.DealYoloResult(result);                           //分析目标检测的结果
        cv::Rect rect = YoloV5::get_rect(frame, CurFaceResult.RectFace); // 坐标转换，得到目标框在图中的位置

        // 下标容易越界，必须保护一下，不然会异常崩溃
        if (rect.x > 0 && rect.y > 0 && rect.width > 0 && rect.height > 0 &&
            (rect.x + rect.width) < frame.cols &&
            (rect.y + rect.height) < frame.rows)
        {
            // cout << "picture:   " << msg->id << "   get face rect successfully." << endl;
        }
        else
        {
            CurFaceResult.FaceCaptured = false;
        }

        cv::Mat face_frame;
        face_msg.id = msg->id;
        // 准备发布yolo检测结果，是否抓到人脸
        if(CurFaceResult.FaceCaptured && CurFaceResult.FaceNum == 1)
        {
            face_frame = frame(rect); //截出人脸的图像
            cout << "picture:   " << msg->id << "   get face rect successfully." << endl;
            //存储照片，照片名：序号.jpg
            pic_name = std::to_string(msg->id)+".jpg";
            cv::imwrite(cap_path + pic_name, face_frame);   //保存图片
        }
        else
        {
            cout << "picture:   " << msg->id << "   get face rect failed." << endl;
            face_frame =  frame;  //截出人脸的图像
            face_msg.id = -2;       //表示未识别到人脸
        }

        face_msg.FaceImage = *(cv_bridge::CvImage(std_msgs::Header(), "bgr8", face_frame ).toImageMsg());  //用cv_bridge转化mat


        //在消息回调函数里面发布另一个话题的消息
        FaceInfoPub.publish(face_msg);

    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("Could not convert from image to 'bgr8'.");
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "image_analyze_node");
    ros::NodeHandle node_analyze;

    //防止崩溃，所有模型引擎优先初始化
    cudaSetDevice(DEVICE);
    YoloV5::InitYoloV5Engine();
    cout << "YoloV5 引擎序列化完成" << endl;

    // CurAnalyzeStamp = ros::Time::now();
     //订阅
    ros::Subscriber sub = node_analyze.subscribe("/frames", 1, imageCallback);
    //发布
    FaceInfoPub = node_analyze.advertise<face_feature::FacePicMsg>("/face_picture", 1);

    ros::spin();

    YoloV5::ReleaseYoloV5Engine();
}
