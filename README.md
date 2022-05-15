# face_feature
当前版本是用运行generate一个节点生成face.data
输入的图片是只截取了人脸的部分。
人脸照片命名格式为：数字_人名。eg:123_司机1
读取照片需要给出照片文件夹路径参数。
默认存储路径：
std::string face_data_path = "/home/nvidia/Documents/face_data/";   //文件夹下存储face.data，为人脸特征库
std::string map_path = "/home/nvidia/Documents/face_map/";  //文件夹下存储face_map.txt，为id映射表