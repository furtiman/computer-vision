#include "ros/ros.h"
#include <iostream>

#define PC_TOPIC_IN "/cloud_map"
#define SUB_QUEUE_SIZE 5
#define ADV_QUEUE_SIZE 1
#define PC_TOPIC_OUT "/processed_cloud_map"

using namespace pcl;

class SubscribeProcessPublish
{
public:
    SubscribeProcessPublish()
    {
        // Assign subscriber
        this->subscriber =
            this->nh.subscribe<PCLPointCloud2>(PC_TOPIC_IN, SUB_QUEUE_SIZE,
                                               &SubscribeProcessPublish::msg_callback,
                                               this);

        // Assign publisher
        this->publisher = this->nh.advertise<PCLPointCloud2>(PC_TOPIC_OUT, ADV_QUEUE_SIZE);
    }

    void msg_callback(const PCLPointCloud2ConstPtr &cloud_msg)
    {
        std::cout << "Received PC package with seq ID " << cloud_msg->header.seq << std::endl;
        // The height of the point cloud is 1, which indicates that it is unordered
        // http://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/PointCloud2.html
        // std::cout << "Height: " << cloud_msg->height << "Width: " << cloud_msg->width << std::endl;
    }

private:
    ros::NodeHandle nh;
    ros::Subscriber subscriber;
    ros::Publisher publisher;
};

int main(int argc, char **argv)
{
    // initialise the node
    ros::init(argc, argv, "listener_cpp");
    // Create an instance of the class, subscribe to topic
    SubscribeProcessPublish node;
    std::cout << "listener_cpp node initialised" << std::endl;

    ros::spin();
    return 0;
}
