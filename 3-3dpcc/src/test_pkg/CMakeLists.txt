cmake_minimum_required(VERSION 3.0.2)
project(test_pkg)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  pcl_ros
  pcl_conversions
  sensor_msgs
)

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES test_pkg
#  CATKIN_DEPENDS roscpp rospy std_msgs
#  DEPENDS system_lib
)

###########
## Build ##
###########
include_directories(${PCL_INCLUDE_DIRS})

link_directories(${PCL_LIBRARY_DIRS})

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

# add_executable(talker src/talker.cpp)
# target_link_libraries(talker ${catkin_LIBRARIES})
# add_executable(listener src/listener.cpp)
# target_link_libraries(listener ${catkin_LIBRARIES})

# add_executable(knn src/knn.cpp)
# target_link_libraries(knn ${catkin_LIBRARIES})

add_executable(rotate src/rotate.cpp)
target_link_libraries(rotate ${catkin_LIBRARIES})

add_executable(dim_filter src/dim_filter.cpp)
target_link_libraries(dim_filter ${catkin_LIBRARIES})

add_executable(outlier_removal src/outlier_removal.cpp)
target_link_libraries(outlier_removal ${catkin_LIBRARIES})

add_executable(octree_density src/octree_density.cpp)
target_link_libraries(octree_density ${catkin_LIBRARIES})
