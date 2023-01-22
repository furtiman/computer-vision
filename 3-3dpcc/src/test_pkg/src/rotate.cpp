#include "ros/ros.h"
#include <iostream>

#include "pcl/point_cloud.h"
#include <pcl_ros/point_cloud.h>
#include "pcl_conversions/pcl_conversions.h"
#include "pcl/console/parse.h"
#include <pcl/filters/voxel_grid.h>
#include "pcl/common/transforms.h"
#include "tf/transform_datatypes.h"
#include "tf/transform_listener.h"

#define PC_TOPIC_IN "/cloud_map"
#define SUB_QUEUE_SIZE 5
#define ADV_QUEUE_SIZE 1
#define PC_TOPIC_OUT "/rotated"

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
        // this->publisher = this->nh.advertise<PCLPointCloud2>(PC_TOPIC_OUT, ADV_QUEUE_SIZE);
        this->publisher = this->nh.advertise<PointCloud<PointXYZRGBA>>(PC_TOPIC_OUT, ADV_QUEUE_SIZE);
    }

    void msg_callback(const PCLPointCloud2ConstPtr &cloud_msg)
    {
        std::cout << "Received PC package with seq ID " << cloud_msg->header.seq << std::endl;

        PointCloud<PointXYZRGBA>::Ptr rotated(new PointCloud<PointXYZRGBA>());
        PointCloud<PointXYZRGBA> pclXYZ;
        PCLPointCloud2 out;

        fromPCLPointCloud2(*cloud_msg, pclXYZ);

        float theta = M_PI / 48; // The angle of rotation in radians (~ 7,5 deg)

        // Create a T/R matrix
        Eigen::Affine3f transform_2 = Eigen::Affine3f::Identity();
        // Add rotation to the transform matrix; theta radians around Y axis
        transform_2.rotate(Eigen::AngleAxisf(theta, Eigen::Vector3f::UnitY()));
        // Define a translation of 2.5 meters on the x axis.
        transform_2.translation() << 0.0, 5.0, 3.0;
        // Executing the transformation
        transformPointCloud(pclXYZ, *rotated, transform_2);

        toPCLPointCloud2(*rotated, out);
        this->publisher.publish(out);
        // Rate looprate(100);
        // looprate.sleep();
    }

private:
    ros::NodeHandle nh;
    ros::Subscriber subscriber;
    ros::Publisher publisher;
};

int main(int argc, char **argv)
{
    // initialise the node
    ros::init(argc, argv, "rotate");
    // Create an instance of the class, subscribe to topic
    SubscribeProcessPublish node;
    std::cout << "rotate node initialised" << std::endl;

    ros::spin();
    return 0;
}
