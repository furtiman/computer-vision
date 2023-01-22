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

#include "pcl/octree/octree.h"
#include "pcl/octree/octree_impl.h"
#include "pcl/octree/octree_pointcloud_adjacency.h"

#include "pcl/outofcore/outofcore.h"

#include <math.h>

#define PC_TOPIC_IN "/rotated"
#define SUB_QUEUE_SIZE 10
#define ADV_QUEUE_SIZE 5
#define PC_TOPIC_OUT "/octree"

#define OCTREE_RES 0.03f
#define COLOR_MATCH_RADIUS 0.2f
#define COLUMN_POINTS_THRESHOLD 80

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
        this->publisher = this->nh.advertise<PointCloud<PointXYZ>>(PC_TOPIC_OUT, ADV_QUEUE_SIZE);
    }

    void msg_callback(const PCLPointCloud2ConstPtr &cloud_msg)
    {
        std::cout << "Received PC package with seq ID " << cloud_msg->header.seq << std::endl;
        std::cout << "IN header " << cloud_msg->header << std::endl;

        PointCloud<PointXYZ>::Ptr pclXYZ(new PointCloud<PointXYZ>());
        PointCloud<PointXYZ>::Ptr pclXYZ_out_columns(new PointCloud<PointXYZ>());

        PointCloud<PointXYZRGB>::Ptr pclXYZ_out_downsample(new PointCloud<PointXYZRGB>());
        PointCloud<PointXYZRGB>::Ptr pclXYZ_out_final(new PointCloud<PointXYZRGB>());

        PCLPointCloud2::Ptr cloud_voxel_filtered(new PCLPointCloud2());
        // Define Voxel Grid filter for downsampling the original cloud
        VoxelGrid<PCLPointCloud2> voxel_filter;
        voxel_filter.setInputCloud(cloud_msg);
        voxel_filter.setLeafSize(OCTREE_RES, OCTREE_RES, OCTREE_RES);
        voxel_filter.filter(*cloud_voxel_filtered);

        fromPCLPointCloud2(*cloud_voxel_filtered, *pclXYZ_out_downsample);

        fromPCLPointCloud2(*cloud_msg, *pclXYZ);

        pclXYZ_out_columns->width = pclXYZ->width;
        pclXYZ_out_columns->height = pclXYZ->height;
        pclXYZ_out_columns->header.frame_id = "map";
        int cloud_size = 0;
        int occupied_cols = 0;

        octree::OctreePointCloudDensity<PointXYZ> octreeA(OCTREE_RES);

        octreeA.defineBoundingBox(13.0, 13.0, 6.0);
        octreeA.setInputCloud(pclXYZ);
        octreeA.addPointsFromInputCloud();

        int downsample_cloud_index = 0;
        for (float x = 0.005f; x < 12.5f; x += OCTREE_RES)
        {
            for (float y = 0.005f; y < 12.5f; y += OCTREE_RES)
            {
                int column_points = 0;
                for (float z = 0.005f; z < 5.9f; z += OCTREE_RES)
                {
                    column_points += octreeA.getVoxelDensityAtPoint(PointXYZ(x, y, z));
                }

                // If column does not have enough points, delete voxels from
                // all the column
                if (column_points > COLUMN_POINTS_THRESHOLD)
                {
                    occupied_cols++;
                    for (float z = 0.005f; z < 5.9f; z += OCTREE_RES)
                    {
                        PointXYZ p = PointXYZ(x, y, z);

                        if (octreeA.isVoxelOccupiedAtPoint(p))
                        {
                            pclXYZ_out_columns->push_back(p);
                            cloud_size++;
                        }
                    }
                }
            }
        }

        pclXYZ_out_columns->width = cloud_size;

        PointXYZ p = pclXYZ_out_columns->points[0];

        pclXYZ_out_final->width = pclXYZ->width;
        pclXYZ_out_final->height = pclXYZ->height;
        pclXYZ_out_final->header.frame_id = "map";

        int unmatched_num = 0;
        cloud_size = 0;

        std::vector<pcl::PointXYZ, Eigen::aligned_allocator<pcl::PointXYZ>> columns = pclXYZ_out_columns->points;
        std::vector<pcl::PointXYZRGB, Eigen::aligned_allocator<pcl::PointXYZRGB>> ds = pclXYZ_out_downsample->points;

        // Color matching
        for (int i = 0; i < pclXYZ_out_columns->width; ++i)
        {
            bool matched = false;
            double c_x = columns[i].x;
            double c_y = columns[i].y;
            double c_z = columns[i].z;
            for (int j = 0; j < pclXYZ_out_downsample->width && !matched; ++j)
            {
                double ds_x = ds[j].x;
                double ds_y = ds[j].y;
                double ds_z = ds[j].z;
                double d_x = abs(c_x - ds_x);
                double d_y = abs(c_y - ds_y);
                double d_z = abs(c_z - ds_z);
                double sum = d_x + d_y + d_z;

                // If the point from the colored cloud is close enough, get its cloud
                if (sum < COLOR_MATCH_RADIUS * 3)
                {
                    cloud_size++;
                    PointXYZRGB p_rgb;
                    p_rgb.x = c_x;
                    p_rgb.y = c_y;
                    p_rgb.z = c_z;

                    p_rgb.r = pclXYZ_out_downsample->points[j].r;
                    p_rgb.g = pclXYZ_out_downsample->points[j].g;
                    p_rgb.b = pclXYZ_out_downsample->points[j].b;

                    pclXYZ_out_final->push_back(p_rgb);
                    matched = true;
                }
            }
            if (!matched)
            {
                unmatched_num++;
            }
            if (i % 1000 == 0)
                std::cout << "Point" << i << "out of" << pclXYZ_out_columns->width << std::endl;
        }

        pclXYZ_out_final->width = cloud_size;

        std::cout << "Occupied cols:" << occupied_cols << std::endl;
        std::cout << "Orig length:" << pclXYZ->width << std::endl;
        std::cout << "Downsample length:" << pclXYZ_out_downsample->width << std::endl;
        std::cout << "Columns length:" << pclXYZ_out_columns->width << std::endl;
        std::cout << "Total unmatched points:" << unmatched_num << std::endl;
        std::cout << "FINAL LENGTH:" << pclXYZ_out_final->width << std::endl;

        this->publisher.publish(*pclXYZ_out_final);
    }

private:
    ros::NodeHandle nh;
    ros::Subscriber subscriber;
    ros::Publisher publisher;
};

int main(int argc, char **argv)
{
    // initialise the node
    ros::init(argc, argv, "density");
    // Create an instance of the class, subscribe to topic
    SubscribeProcessPublish node;
    std::cout << "density node initialised" << std::endl;

    ros::spin();
    return 0;
}
