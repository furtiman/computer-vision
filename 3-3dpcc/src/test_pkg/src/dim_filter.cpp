#include "ros/ros.h"
#include <iostream>
#include "pcl/point_cloud.h"
#include "pcl_conversions/pcl_conversions.h"
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>

#define PC_TOPIC_IN "/rotated"
#define PC_TOPIC_OUT "/dim_crop"

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

        // PCLPointCloud2::Ptr cloud_voxel_filtered(new PCLPointCloud2());
        PCLPointCloud2::Ptr floor_ceiling_removed(new pcl::PCLPointCloud2());
        PCLPointCloud2::Ptr x_removed(new PCLPointCloud2());
        PCLPointCloud2::Ptr y_removed(new PCLPointCloud2());
        PCLPointCloud2::Ptr cloud_filtered(new PCLPointCloud2());

        // // Define Voxel Grid filter for downsampling
        // VoxelGrid<PCLPointCloud2> voxel_filter;
        // voxel_filter.setInputCloud(cloud_msg);
        // voxel_filter.setLeafSize(float(0.05), float(0.05), float(0.05));
        // voxel_filter.filter(*cloud_voxel_filtered);

        // define a PassThrough filter
        PassThrough<PCLPointCloud2> pass_z;
        pass_z.setInputCloud(cloud_msg);
        pass_z.setFilterFieldName("z");
        pass_z.setFilterLimits(-1.4, 0.9);
        pass_z.filter(*floor_ceiling_removed);

        PassThrough<PCLPointCloud2> pass_x;
        pass_x.setInputCloud(floor_ceiling_removed);
        pass_x.setFilterFieldName("x");
        pass_x.setFilterLimits(1.1, 10);
        pass_x.filter(*x_removed);

        PassThrough<PCLPointCloud2> pass_y;
        pass_y.setInputCloud(x_removed);
        pass_y.setFilterFieldName("y");
        pass_y.setFilterLimits(-5, 5.45);
        pass_y.filter(*y_removed);

        // Publish the data for visualisation
        this->publisher.publish(*y_removed);
    }

private:
    ros::NodeHandle nh;
    ros::Subscriber subscriber;
    ros::Publisher publisher;
};

int main(int argc, char **argv)
{
    // initialise the node
    ros::init(argc, argv, "dim_filter");

    SubscribeProcessPublish node;
    std::cout << "dim_filter node initialised" << std::endl;

    // handle ROS communication events

    ros::spin();
    return 0;
}