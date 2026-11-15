#pragma once
#include <mutex>
// ROG-Map's algorithm never uses ROS conversions. Supply the PCD IO include
// normally included transitively by the ROS header, without linking ROS.
#include <pcl/io/pcd_io.h>
