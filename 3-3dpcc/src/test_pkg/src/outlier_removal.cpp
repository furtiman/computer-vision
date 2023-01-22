#include "ros/ros.h"
#include <iostream>
#include "pcl/point_cloud.h"
#include "pcl_conversions/pcl_conversions.h"
#include <pcl/filters/statistical_outlier_removal.h>

#define PC_TOPIC_IN "/dim_crop"
#define PC_TOPIC_OUT "/outlier_removed"
#define MEAN_K 50
#define DEVIATION_THRESHOLD 0.1

using namespace pcl;

class SubscribeProcessPublish
{
public:
    SubscribeProcessPublish()
    {
        // Assign subscriber
        this->subscriber =
            this->nh.subscribe<PCLPointCloud2>(PC_TOPIC_IN, 5,
                                               &SubscribeProcessPublish::processPcMsg,
                                               this);

        // Assign publisher
        this->publisher = this->nh.advertise<PCLPointCloud2>(PC_TOPIC_OUT, 1);
    }

    void processPcMsg(const PCLPointCloud2ConstPtr &cloud_msg)
    {
        std::cout << "Received PC package with seq ID " << cloud_msg->header.seq << std::endl;
        PCLPointCloud2::Ptr noise_removed(new PCLPointCloud2());

        // Create the filtering object
        StatisticalOutlierRemoval<PCLPointCloud2> sor;
        sor.setInputCloud(cloud_msg);
        sor.setMeanK(MEAN_K);
        sor.setStddevMulThresh(DEVIATION_THRESHOLD);
        sor.filter(*noise_removed);

        // Publish the data for visualisation
        this->publisher.publish(*noise_removed);
    }

private:
    ros::NodeHandle nh;
    ros::Subscriber subscriber;
    ros::Publisher publisher;
};

int main(int argc, char **argv)
{
    // initialise the node
    ros::init(argc, argv, "outlier_removal");

    SubscribeProcessPublish node;
    std::cout << "outlier_removal node initialised" << std::endl;

    ros::spin();
    return 0;
}