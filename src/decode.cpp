#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <stdlib.h>
#include "face_feature/FacePicMsg.h"

using namespace std;
//设置gstreamer管道参数tx2s

string video_path = "/home/nvidia/Videos/";
string cam0_path = "/home/nvidia/wdq/picture/cam0/";   //存储摄像头拍摄的照片
string pic_path = "/home/nvidia/Pictures/face_pic/";    //读取待录入图片的路径
string txt_path = "/home/nvidia/wdq/ros_vision/src/face_feature/src/data/";     //存.txt文件的路径

int num = 0;    //给照片编号排序
string image_name;
string video_name;
string pic_name;

ros::Publisher FacePicPub;
face_feature::FacePicMsg face_pic_msg;

std::string gstreamer_pipeline (int capture_width, int capture_height, int display_width, int display_height, int framerate, int flip_method)
{
    return "nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=(int)" + std::to_string(capture_width) + ", height=(int)" +
           std::to_string(capture_height) + ", format=(string)NV12, framerate=(fraction)" + std::to_string(framerate) +
           "/1 ! nvvidconv flip-method=" + std::to_string(flip_method) + " ! video/x-raw, width=(int)" + std::to_string(display_width) + ", height=(int)" +
           std::to_string(display_height) + ", format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
}

// // //hk
// std::string gstreamer_pipeline(std::string uri, int latency, int display_width, int display_height) 
// {
//     return "rtspsrc location=" + uri +
//             " latency=" + std::to_string(latency) +
//             " ! rtph264depay ! h264parse ! omxh264dec ! nvvidconv ! video/x-raw, " + 
//             "width=(int)" + std::to_string(display_width) + 
//             ", height=(int)" + std::to_string(display_height) + 
//             ", format=(string)BGRx ! videoconvert ! appsink";
// }

int main(int argc, char **argv)
{
    ros::init(argc, argv, "pub_cam_node");  //argc:remapping 参数的个数，argv参数列表，运行时节点名
    ros::NodeHandle n;

//tx2
    int capture_width = 1280 ;
    int capture_height = 720 ;
    int display_width = 1280 ;
    int display_height = 720 ;
    int framerate = 10 ;
    int flip_method =2 ;

    std::string pipeline = gstreamer_pipeline(capture_width,
                           capture_height,
                           display_width,
                           display_height,
                           framerate,
                           flip_method);
    

    // //1 捕获视频
    // cv::VideoCapture capture;
    // capture.open(pipeline, cv::CAP_GSTREAMER);
    // if(!capture.isOpened())
    // {
    //     ROS_ERROR_STREAM("Failed to open camera!");
    //     ros::shutdown();
    // }

    // //1.2读取本地视频，用于获取录入素材
    // video_name = "IMG_4500.MOV";
    // cv::VideoCapture cap(video_path + video_name);
    // if(!cap.isOpened()){
	// 	std::cout<<"open video failed!"<<std::endl;
	// 	return -1;
	// }
	// else
	// std::cout<<"open video success!"<<std::endl;
    // bool isSuccess = true;
    // cv::Mat img;      //存储读取的本地视频的每一帧

    //1.3读取待录入的照片
    const std::string face_pic_name = "face_pic_name.txt";
    const std::string suffix = ".jpg";
    unsigned int suffix_len = suffix.length();
    ifstream infile;
    infile.open(txt_path + face_pic_name);   //将文件流对象与文件连接起来
    if(infile.is_open())
    {
            cout << "face_pic_name.txt opened successfully." << endl;
    }
    //assert(infile.is_open());   //若失败,则输出错误消息,并终止程序
    string line;                            //每次读取一行
    cv::Mat img;                        //存储读到的本地照片

    // //2 创建ROS中图像的发布者
    // image_transport::ImageTransport it(n);
    // image_transport::Publisher pub_image = it.advertise("/camera_csi0/frames", 1);

    // //cv_bridge功能包提供了ROS图像和OpenCV图像转换的接口，建立了一座桥梁
    // cv_bridge::CvImagePtr frame = boost::make_shared<cv_bridge::CvImage>();
    // frame->encoding = sensor_msgs::image_encodings::BGR8;

    //2.2创建发布者，自定义消息类型
    FacePicPub = n.advertise<face_feature::FacePicMsg>("/frames", 1);


    while(ros::ok())
    {
        // /1.1/摄像头
        // capture >> frame->image; //流的转换

        // if(frame->image.empty())
        // {
        //     ROS_ERROR_STREAM("Failed to capture frame!");
        //     ros::shutdown();
        // }
        // //打成ROS数据包
        // frame->header.stamp = ros::Time::now();
        // cv::Mat img = frame->image;

        // //1.2读取本地视频
        // isSuccess = cap.read(img);
		// if(!isSuccess){//if the video ends, then break
		// 	std::cout<<"video ends"<<std::endl;
		// 	break;
		// }

        //1.3读取本地照片
        if(getline(infile, line))
        {
            pic_name = line.substr(0, line.length() - suffix_len);
            num = atoi(pic_name.c_str());
            pic_name = pic_name + ".jpg";
            img = cv::imread(pic_path + pic_name, 1); 
            imshow("img: ",img);
            face_pic_msg.FaceImage =*(cv_bridge::CvImage(std_msgs::Header(), "bgr8", img ).toImageMsg());  //用cv_bridge转化mat
            face_pic_msg.id = num;
            cout << "num:   " << num << endl;
            //发布消息
            FacePicPub.publish(face_pic_msg);
        }
        else
        {
            infile.close();
            cout << "read face_pic_name.txt finish." << endl;
            face_pic_msg.id = -1;       //表示传输结束
            //发布消息
            FacePicPub.publish(face_pic_msg);
            break;
        }

        // //打成ROS数据包
        // frame->header.stamp = ros::Time::now();
        // frame->image = img;


        
        // //倒置画面，可以，但是球机上画面上的日期信息也会倒置，不方面查看
        // cv::Mat pubimg;     //存储倒置的画面
        // flip(img,pubimg,-1);
        // frame->image = pubimg;//需要倒置
        // //test
        // //imshow("pubimg:",pubimg);


        // //存储照片，照片名：序号.jpg
        // image_name = std::to_string(num)+".jpg";
        // cv::imwrite(cam0_path + image_name, img);   //保存图片
        // num++;
        
        // pub_image.publish(frame->toImageMsg());

        cv::waitKey(1000);//opencv刷新图像 3ms
        ros::spinOnce();
    }

    //capture.release();    //释放流
    // cap.release();
    return EXIT_SUCCESS;
}
