#include "ros/ros.h"
#include "sensor_msgs/Image.h"
#include <std_msgs/UInt8.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>

using namespace std;
using namespace ros;

class ImageConverter
{
        NodeHandle gdd;
        Subscriber depth_sub;
        Publisher gray_pub;

        public:
        void imageCb(const sensor_msgs::Image::ConstPtr& );

        ImageConverter()
        {

                depth_sub = gdd.subscribe("/camera/depth_registered/image_raw", 1, &ImageConverter::imageCb, this);
                gray_pub = gdd.advertise<sensor_msgs::Image>("depthTOgreyscale_image", 1);
        }


};

void ImageConverter::imageCb(const sensor_msgs::ImageConstPtr& depth_image)
  {
        cv_bridge::CvImagePtr cv_ptr;
        try
        {
         cv_ptr=cv_bridge::toCvCopy(depth_image, sensor_msgs::image_encodings::TYPE_32FC1);
        }catch(cv_bridge::Exception& e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }
        cv::Mat cv_image = cv_ptr->image;
        //cv::Mat gray_image(depth_image->height,depth_image->width,CV_8UC1);

        int channels =cv_image.channels();
        int nRows = cv_image.rows;
        int nCols = cv_image.cols*channels;
        cv::Mat gray_image;
     //   if(cv_image.isContinuous()){
     //       nCols *=nRows;
     //       nRows=1;
     //   }
        ushort* p_gray;
        cv_image.convertTo(gray_image,CV_16UC1,255,0);
        int min_val=99999;
        int max_val=-99999;
        for (int i=0;i<nRows; i++){
            p_gray=gray_image.ptr<ushort>(i);
            for(int j=0; j<nCols;j++){
                max_val=max(max_val,(int)p_gray[j]);
              min_val=min(min_val,(int)p_gray[j]);
            }
        }
        ROS_INFO("max= %d,min=%d",max_val,min_val);
        for (int i=0;i<nRows; i++){
            p_gray=gray_image.ptr<ushort>(i);
            for(int j=0; j<nCols;j++){
                    if(p_gray[j]==0){
                        p_gray[j]=max_val;
                    }
               p_gray[j]=(log(abs(p_gray[j])-min_val+1)/log(max_val-min_val+1)*255);
              // p_gray[j]=p_gray[j]/255*255;
            }
        }
        //cv::Mat blur;
        cv::Mat output;
        //cv::GaussianBlur(gray_image,blur,cv::Size(3,3),0);
       gray_image.convertTo(output,CV_8UC1,1);
        try
        {
        sensor_msgs::ImagePtr gray_image_msg=cv_bridge::CvImage(std_msgs::Header(), sensor_msgs::image_encodings::MONO8, output).toImageMsg();

        gray_image_msg->header.seq = depth_image->header.seq;
        gray_image_msg->header.stamp = depth_image->header.stamp;
        gray_image_msg->header.frame_id = depth_image->header.frame_id;
        gray_pub.publish(gray_image_msg);
}catch(cv_bridge::Exception& e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }
}


int main(int argc, char **argv)
{
        init(argc, argv, "get_depth_data");
        ImageConverter ic;
        spin();
        return 0;
}
