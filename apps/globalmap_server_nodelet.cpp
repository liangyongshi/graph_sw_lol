#include <mutex>
#include <memory>
#include <iostream>

#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <tf_conversions/tf_eigen.h>
#include <tf/transform_broadcaster.h>

#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>

#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>

#include <pcl/filters/voxel_grid.h>

#include <pclomp/ndt_omp.h>

#include <hdl_localization/pose_estimator.hpp>

#include <map_server/image_loader.h>
#include <yaml-cpp/yaml.h>
#ifdef HAVE_NEW_YAMLCPP
template<typename T>
void operator >> (const YAML::Node& node, T& i)
{
    i = node.as<T>();
}
#endif



namespace hdl_localization {

class GlobalmapServerNodelet : public nodelet::Nodelet {
public:
  using PointT = pcl::PointXYZI;

  GlobalmapServerNodelet() {
  }
  virtual ~GlobalmapServerNodelet() {
  }

  void onInit() override {
    nh = getNodeHandle();
    mt_nh = getMTNodeHandle();
    private_nh = getPrivateNodeHandle();
    initialize_params();
   // publish globalmap with "latched" publisher
    globalmap_pub = nh.advertise<sensor_msgs::PointCloud2>("/globalmap", 5, true);
    globalmap_pub_timer = nh.createWallTimer(ros::WallDuration(0.05), &GlobalmapServerNodelet::pub_once_cb, this, true, true);

  }

private:
  void initialize_params() {
    // read globalmap from a pcd file
    std::string globalmap_pcd = private_nh.param<std::string>("globalmap_pcd", "");
    globalmap.reset(new pcl::PointCloud<PointT>());
    pcl::io::loadPCDFile(globalmap_pcd, *globalmap);
    globalmap->header.frame_id = "map1";

    std::ifstream utm_file(globalmap_pcd + ".utm");
    if (utm_file.is_open() && private_nh.param<bool>("convert_utm_to_local", true)) {
      double utm_easting;
      double utm_northing;
      double altitude;
      utm_file >> utm_easting >> utm_northing >> altitude;
      for(auto& pt : globalmap->points) {
        pt.getVector3fMap() -= Eigen::Vector3f(utm_easting, utm_northing, altitude);
      }
      ROS_INFO_STREAM("Global map offset by UTM reference coordinates (x = " 
                      << utm_easting << ", y = " << utm_northing << ") and altitude (z = " << altitude << ")");
    }

    // downsample globalmap
    double downsample_resolution = private_nh.param<double>("downsample_resolution", 0.1);
    boost::shared_ptr<pcl::VoxelGrid<PointT>> voxelgrid(new pcl::VoxelGrid<PointT>());
    voxelgrid->setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);
    voxelgrid->setInputCloud(globalmap);

    pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
    voxelgrid->filter(*filtered);

    globalmap = filtered;

/*****************pub occupancy grid map****************************/
     pubmap();
  }

  void pub_once_cb(const ros::WallTimerEvent& event) {
    globalmap_pub.publish(globalmap);
  }

  void pubmap()
  {
      YAML::Node doc = YAML::LoadFile(yaml_path_);
      try
      {
          doc["resolution"] >> res_;
          doc["origin"][0] >> origin_[0];
          doc["origin"][1] >> origin_[1];
          doc["origin"][2] >> origin_[2];
          doc["negate"] >> negate_;
          doc["occupied_thresh"] >> occ_th_;
          doc["free_thresh"] >> free_th_;
      }
      catch(YAML::InvalidScalar)
      {
          ROS_ERROR("The .yaml does not contain tags required or they are invalid.");
          exit(-1);
      }
      mode_ = TRINARY;

      std::cout << "	       map name: " << map_name_
                << "\n	     resolution: " << res_
                << "\n	         origin: " << origin_[0] << ", " << origin_[1] << ", " << origin_[2]
                << "\n	         negate: " << negate_
                << "\n	occupied thresh: " << occ_th_
                << "\n	    free thresh: " << free_th_ << std::endl;
      map_server::loadMapFromFile(&map_resp_, pgm_path_, res_, negate_, occ_th_, free_th_, origin_, mode_);
      map_resp_.map.header.frame_id = "map1";
      map_publisher_ = nh.advertise<nav_msgs::OccupancyGrid> ("map", 1, true);
      map_publisher_.publish(map_resp_.map);
  }

private:
  // ROS
  ros::NodeHandle nh;
  ros::NodeHandle mt_nh;
  ros::NodeHandle private_nh;

  ros::Publisher globalmap_pub;

  ros::WallTimer globalmap_pub_timer;
  pcl::PointCloud<PointT>::Ptr globalmap;

  const char* yaml_path_="/home/xcy/graph_sw_localization/src/graph_localization/map/map.yaml";
  const char* pgm_path_="/home/xcy/graph_sw_localization/src/graph_localization/map/map.pgm";
  MapMode mode_;
  double origin_[3];///< 地图左下角相对地图原点的坐标
  double res_;
  int negate_;
  double occ_th_, free_th_;

  nav_msgs::GetMap::Response map_resp_;
    nav_msgs::MapMetaData meta_data_message_;
  std::string map_name_;
  ros::Publisher map_publisher_;
};

}


PLUGINLIB_EXPORT_CLASS(hdl_localization::GlobalmapServerNodelet, nodelet::Nodelet)
