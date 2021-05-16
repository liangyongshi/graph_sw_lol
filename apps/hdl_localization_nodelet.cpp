#include <mutex>
#include <memory>
#include <iostream>
#include <boost/circular_buffer.hpp>

#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
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
#include <pclomp/gicp_omp.h>

#include <hdl_localization/pose_estimator.hpp>

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>

#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>
using gtsam::symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
using gtsam::symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using gtsam::symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)
using namespace gtsam;


namespace hdl_localization {

class HdlLocalizationNodelet : public nodelet::Nodelet {
public:
  using PointT = pcl::PointXYZI;

  HdlLocalizationNodelet() {
  }
  virtual ~HdlLocalizationNodelet() {
  }


  void onInit() override {
    nh = getNodeHandle();
    mt_nh = getMTNodeHandle();
    private_nh = getPrivateNodeHandle();

    processing_time.resize(16);
    imuAccNoise = private_nh.param<double>("imuAccNoise", 0.01);
    imuGyrNoise = private_nh.param<double>("imuGyrNoise", 0.001);
    odomLinearNoise = private_nh.param<double>("odomLinearNoise", 0.01);
    odomAngularNoise = private_nh.param<double>("odomAngularNoise", 0.01);
    gravity= private_nh.param<double>("gravity", 10.04);
    opt_frames = private_nh.param<int>("opt_frames", 20);
    odom_child_frame_id = private_nh.param<std::string>("odom_child_frame_id", "base_link");
    use_imu = private_nh.param<bool>("use_imu", true);
    invert_imu = private_nh.param<bool>("invert_imu", false);
    key_interval = private_nh.param<float>("key_interval", 0.1);
    imu_sub = mt_nh.subscribe("/imu/data_raw", 256, &HdlLocalizationNodelet::imu_callback, this);
    points_sub = mt_nh.subscribe("/velodyne_points", 5, &HdlLocalizationNodelet::points_callback, this);
    globalmap_sub = nh.subscribe("/globalmap", 1, &HdlLocalizationNodelet::globalmap_callback, this);
    initialpose_sub = nh.subscribe("/initialpose", 8, &HdlLocalizationNodelet::initialpose_callback, this);
    pose_incre_pub = nh.advertise<nav_msgs::Odometry>("/ndt/incre_odom", 30, false);
    pose_pub = nh.advertise<nav_msgs::Odometry>("/ndt/global_odom", 20, false);
    initialize_params();

  }

private:
  void initialize_params() {
    // intialize scan matching method
    double downsample_resolution = private_nh.param<double>("downsample_resolution", 0.1);
    std::string ndt_neighbor_search_method = private_nh.param<std::string>("ndt_neighbor_search_method", "DIRECT7");

    double ndt_resolution = private_nh.param<double>("ndt_resolution", 1.0);
    boost::shared_ptr<pcl::VoxelGrid<PointT>> voxelgrid(new pcl::VoxelGrid<PointT>());
    voxelgrid->setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);
    downsample_filter = voxelgrid;

    pclomp::NormalDistributionsTransform<PointT, PointT>::Ptr ndt(new pclomp::NormalDistributionsTransform<PointT, PointT>());
    pclomp::GeneralizedIterativeClosestPoint<PointT, PointT>::Ptr gicp(new pclomp::GeneralizedIterativeClosestPoint<PointT, PointT>());

    ndt->setTransformationEpsilon(0.01);
    ndt->setResolution(ndt_resolution);
    if(ndt_neighbor_search_method == "DIRECT1") {
      NODELET_INFO("search_method DIRECT1 is selected");
      ndt->setNeighborhoodSearchMethod(pclomp::DIRECT1);
      registration = ndt;
//        ndt->setNeighborhoodSearchMethod(pclomp::DIRECT7);
//      localregistration=ndt;
    } else if(ndt_neighbor_search_method == "DIRECT7") {
      NODELET_INFO("search_method DIRECT7 is selected");
      ndt->setNeighborhoodSearchMethod(pclomp::DIRECT7);
      registration = ndt;
//      localregistration=ndt;

    } else if(ndt_neighbor_search_method == "GICP_OMP"){
      NODELET_INFO("search_method GICP_OMP is selected");
      registration = gicp;
    }
     else {
      if(ndt_neighbor_search_method == "KDTREE") {
        NODELET_INFO("search_method KDTREE is selected");
      } else {
        NODELET_WARN("invalid search method was given");
        NODELET_WARN("default method is selected (KDTREE)");
      }
      ndt->setNeighborhoodSearchMethod(pclomp::KDTREE);
      registration = ndt;

    }


    // initialize pose estimator
    if(private_nh.param<bool>("specify_init_pose", true))
    {
      NODELET_INFO("initialize pose estimator with specified parameters!!");
      pose_estimator.reset(new hdl_localization::PoseEstimator(registration,
        ros::Time::now(),
        Eigen::Vector3d(private_nh.param<double>("init_pos_x", 0.0), private_nh.param<double>("init_pos_y", 0.0), private_nh.param<double>("init_pos_z", 0.0)),
        Eigen::Quaterniond(private_nh.param<double>("init_ori_w", 1.0), private_nh.param<double>("init_ori_x", 0.0), private_nh.param<double>("init_ori_y", 0.0), private_nh.param<double>("init_ori_z", 0.0)),
        private_nh.param<double>("cool_time_duration", 0.5)
      ));
    }
//    poseFrom = gtsam::Pose3(gtsam::Rot3::Quaternion(1.0,0.0,0.0,0.0),gtsam::Point3(0.0,0.0,0.0));
      //*****************factor graph********************//
      boost::shared_ptr<gtsam::PreintegrationParams> p = gtsam::PreintegrationParams::MakeSharedU(gravity);
      p->accelerometerCovariance  = gtsam::Matrix33::Identity(3,3) * pow(imuAccNoise, 2); // acc white noise in continuous
      p->gyroscopeCovariance      = gtsam::Matrix33::Identity(3,3) * pow(imuGyrNoise, 2); // gyro white noise in continuous
      p->integrationCovariance    = gtsam::Matrix33::Identity(3,3) * pow(1e-4, 2); // error committed in integrating position from velocities
      gtsam::imuBias::ConstantBias prior_imu_bias((gtsam::Vector(6) << 0, 0, 0, 0, 0, 0).finished()); // assume zero initial bias
      priorPoseNoise  = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2).finished()); // rad,rad,rad,m, m, m
      priorVelNoise   = gtsam::noiseModel::Isotropic::Sigma(3, 1e2); // m/s
      priorBiasNoise  = gtsam::noiseModel::Isotropic::Sigma(6, 1e-3); // 1e-2 ~ 1e-3 seems to be good
      correctionNoise = gtsam::noiseModel::Isotropic::Sigma(6, 1); // meter
      noiseModelBetweenBias = (gtsam::Vector(6) << 3.99395e-03, 3.99395e-03, 3.99395e-03, 1.56363e-03, 1.56363e-03, 1.56363e-03).finished();
      imuIntegratorImu_ = new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias);

      key_count=0;
      systemInitialized= false;

//**************************************************************//
  }

private:
  /**
   * @brief callback for imu data
   * @param imu_msg
   */
  void imu_callback(const sensor_msgs::ImuConstPtr& imu_msg) {
      std::lock_guard<std::mutex> lock(imu_data_mutex);
      imu_data.push_back(imu_msg);
//      ROS_INFO("hahahahahahah");
  }
  /**
   * @brief callback for point cloud data
   * @param points_msg
   */
  void points_callback(const sensor_msgs::PointCloud2ConstPtr& points_msg) {


      points_curr_time=points_msg->header.stamp;

      if(firstkey)
      {
          points_pre_time=points_msg->header.stamp;
          firstkey= false;
      }

      auto time_interval= points_curr_time-points_pre_time;

      if(time_interval.toSec()<key_interval) return;// select key frames according to time interval

      std::lock_guard<std::mutex> estimator_lock(pose_estimator_mutex);

      if(!pose_estimator) {
          NODELET_ERROR("waiting for initial pose input!!");
          return;
      }

      if(!globalmap) {
          NODELET_ERROR("globalmap has not been received!!");
          return;
      }

      const auto& stamp = points_msg->header.stamp;
      pcl::PointCloud<PointT>::Ptr pcl_cloud(new pcl::PointCloud<PointT>());
      pcl::fromROSMsg(*points_msg, *pcl_cloud);



      if(pcl_cloud->empty()) {
          NODELET_ERROR("cloud is empty!!");
          return;
      }

      // transform pointcloud into odom_child_frame_id
      pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
      if(!pcl_ros::transformPointCloud(odom_child_frame_id, *pcl_cloud, *cloud, this->tf_listener)) {
          NODELET_ERROR("point cloud cannot be transformed into target frame!!");
          return;
      }


      auto filtered = downsample(cloud);

      auto t1 = ros::WallTime::now();

//      cur_cloud=filtered;
//      localregistration->setInputTarget(pre_cloud);
//      localregistration->setInputSource(cur_cloud);
//      Eigen::Matrix4f localOdom_temp;
//      Eigen::Matrix4d localOdom;
//      localOdom_temp = localregistration->getFinalTransformation();
//      localOdom=localOdom_temp.cast<double>();





      if(!systemInitialized){

          resetOptimization();

          newgraph.add(PriorFactor<Pose3>(X(0), prevPose_, priorPoseNoise));
          std::cout<<"initial value:"<<prevPose_.x()<<" "<<prevPose_.y()<<" "<<prevPose_.z()<<" "<<prevPose_.rotation().roll()<<" "
          <<prevPose_.rotation().pitch()<<" "<<prevPose_.rotation().yaw()<<std::endl;

          //initial velocity
          prevVel_ = gtsam::Vector3(0, 0, 0);
          gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_, priorVelNoise);
          newgraph.add(priorVel);

          //initial bias
          prevBias_ = gtsam::imuBias::ConstantBias();
          gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(B(0), prevBias_, priorBiasNoise);
          newgraph.add(priorBias);

          //add values
          initialEstimate.insert(X(0), prevPose_);
          initialEstimate.insert(V(0), prevVel_);
          initialEstimate.insert(B(0), prevBias_);

          optimizer->update(newgraph, initialEstimate);

          newgraph.resize(0);
          initialEstimate.clear();
          imuIntegratorImu_->resetIntegrationAndSetBias(prevBias_);
          systemInitialized=true;
          key_count=1;
          return;
      }
      else{



          if(key_count==opt_frames)
          {
              gtsam::noiseModel::Gaussian::shared_ptr updatedPoseNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer->marginalCovariance(X(key_count-1)));
              gtsam::noiseModel::Gaussian::shared_ptr updatedVelNoise  = gtsam::noiseModel::Gaussian::Covariance(optimizer->marginalCovariance(V(key_count-1)));
              gtsam::noiseModel::Gaussian::shared_ptr updatedBiasNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer->marginalCovariance(B(key_count-1)));
              // reset graph
              resetOptimization();
              // add pose
              gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, updatedPoseNoise);
              newgraph.add(priorPose);
              // add velocity
              gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_, updatedVelNoise);
              newgraph.add(priorVel);
              // add bias
              gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(B(0), prevBias_, updatedBiasNoise);
              newgraph.add(priorBias);
              // add values
              initialEstimate.insert(X(0), prevPose_);
              initialEstimate.insert(V(0), prevVel_);
              initialEstimate.insert(B(0), prevBias_);
              // optimize once
              optimizer->update(newgraph, initialEstimate);
              newgraph.resize(0);
              initialEstimate.clear();

              key_count=1;
          }
          gtsam::Pose3 poseFrom = gtsam::Pose3(gtsam::Rot3::Quaternion(pose_estimator->quat().w(),pose_estimator->quat().x(),pose_estimator->quat().y(),pose_estimator->quat().z()),
                                               gtsam::Point3(pose_estimator->pos()(0),pose_estimator->pos()(1),pose_estimator->pos()(2)));
          std::cout<<"prior value:"<<poseFrom.x()<<" "<<poseFrom.y()<<" "<<poseFrom.z()<<" "<<poseFrom.rotation().roll()<<" "
                   <<poseFrom.rotation().pitch()<<" "<<poseFrom.rotation().yaw()<<std::endl;

          auto aligned = pose_estimator->correct(filtered);

          gtsam::Pose3 poseTo = gtsam::Pose3(gtsam::Rot3::Quaternion(pose_estimator->quat().w(),pose_estimator->quat().x(),pose_estimator->quat().y(),pose_estimator->quat().z()),
                                             gtsam::Point3(pose_estimator->pos()(0),pose_estimator->pos()(1),pose_estimator->pos()(2)));
          std::cout<<"update value:"<<poseTo.x()<<" "<<poseTo.y()<<" "<<poseTo.z()<<" "<<poseTo.rotation().roll()<<" "
                   <<poseTo.rotation().pitch()<<" "<<poseTo.rotation().yaw()<<std::endl;




           std::lock_guard<std::mutex> lock(imu_data_mutex);
           auto imu_iter = imu_data.begin();
           for(imu_iter; imu_iter != imu_data.end(); imu_iter++) {
                  if(stamp < (*imu_iter)->header.stamp) {
                      break;
                  }
                  const auto& acc = (*imu_iter)->linear_acceleration;
                  const auto& gyro = (*imu_iter)->angular_velocity;
                  imuIntegratorImu_->integrateMeasurement(gtsam::Vector3(acc.x, acc.y, acc.z),gtsam::Vector3(gyro.x, gyro.y, gyro.z), 0.01);
              }
           imu_data.erase(imu_data.begin(),imu_iter);

          gtsam::PriorFactor<gtsam::Pose3> pose_factor(X(key_count), poseTo, correctionNoise);
          newgraph.add(pose_factor);

          const gtsam::PreintegratedImuMeasurements& preint_imu = dynamic_cast<const gtsam::PreintegratedImuMeasurements&>(*imuIntegratorImu_);
          gtsam::ImuFactor imu_factor(X(key_count - 1), V(key_count - 1), X(key_count), V(key_count), B(key_count - 1), preint_imu);

          newgraph.add(imu_factor);
          newgraph.add(gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>(B(key_count - 1), B(key_count), gtsam::imuBias::ConstantBias(),
                                                                              gtsam::noiseModel::Diagonal::Sigmas(sqrt(imuIntegratorImu_->deltaTij()) * noiseModelBetweenBias)));

          gtsam::NavState propState_ = imuIntegratorImu_->predict(prevState_, prevBias_);


          std::cout<<" imu integre propState:"<<propState_.pose().x()<<" "<<propState_.pose().y()<<" "<<propState_.pose().z()<<" "<<
                   propState_.pose().rotation().roll()<<" "<<propState_.pose().rotation().pitch()<<" "<<propState_.pose().rotation().yaw()<<std::endl;



          if(propState_.pose().z()<-0.1)
          {
              gtsam::Quaternion quat(propState_.quaternion());
              quat.normalized();
              poseTo = gtsam::Pose3(gtsam::Rot3::Quaternion(quat.w(),quat.x(),quat.y(),quat.z()),gtsam::Point3(propState_.pose().x(),propState_.pose().y(),-0.1));
          }
          else if (propState_.pose().z()>0.1)
          {

              gtsam::Quaternion quat(propState_.quaternion());
              quat.normalized();
              poseTo = gtsam::Pose3(gtsam::Rot3::Quaternion(quat.w(),quat.x(),quat.y(),quat.z()),gtsam::Point3(propState_.pose().x(),propState_.pose().y(),0.10));
          }

          std::cout<<"propState_.pose().z():"<<propState_.pose().z()<<std::endl;

          noiseModel::Diagonal::shared_ptr odometryNoise = noiseModel::Diagonal::Variances((Vector(6) << odomLinearNoise, odomLinearNoise, odomLinearNoise, odomAngularNoise, odomAngularNoise, odomAngularNoise).finished());

          newgraph.add(BetweenFactor<Pose3>(X(key_count-1), X(key_count), poseFrom.between(poseTo), odometryNoise));



          initialEstimate.insert(X(key_count), poseTo);
          initialEstimate.insert(V(key_count), propState_.v());
          initialEstimate.insert(B(key_count), prevBias_);
      }

      optimizer->update(newgraph, initialEstimate);
      newgraph.resize(0);
      initialEstimate.clear();
      gtsam::Pose3 latestEstimate;
      isamCurrentEstimate = optimizer->calculateEstimate();

      prevPose_ = isamCurrentEstimate.at<Pose3>(X(key_count));



      prevVel_ =isamCurrentEstimate.at<Vector3>(V(key_count));
      prevState_ = gtsam::NavState(prevPose_, prevVel_);
      prevBias_=isamCurrentEstimate.at<imuBias::ConstantBias>(B(key_count));
      imuIntegratorImu_->resetIntegrationAndSetBias(prevBias_);

      pose_estimator->update(prevState_,prevBias_);

      Eigen::Quaterniond quat(prevPose_.rotation().toQuaternion().w(),prevPose_.rotation().toQuaternion().x(),prevPose_.rotation().toQuaternion().y(),prevPose_.rotation().toQuaternion().z());
      quat.normalize();
      geometry_msgs::Quaternion odom_quat;
      odom_quat.w = quat.w();
      odom_quat.x = quat.x();
      odom_quat.y = quat.y();
      odom_quat.z = quat.z();

      geometry_msgs::TransformStamped odom_trans;
      odom_trans.header.stamp = stamp;
      odom_trans.header.frame_id = "map";
      odom_trans.child_frame_id = odom_child_frame_id;

      odom_trans.transform.translation.x = prevPose_.x();
      odom_trans.transform.translation.y = prevPose_.y();
      odom_trans.transform.translation.z = prevPose_.z();
      odom_trans.transform.rotation = odom_quat;

      pose_broadcaster.sendTransform(odom_trans);

      nav_msgs::Odometry laserOdometryROS;
      laserOdometryROS.header.stamp = points_msg->header.stamp;
      laserOdometryROS.header.frame_id = "map";
      laserOdometryROS.child_frame_id = odom_child_frame_id;
      laserOdometryROS.pose.pose.position.x = prevPose_.x();
      laserOdometryROS.pose.pose.position.y = prevPose_.y();
      laserOdometryROS.pose.pose.position.z = prevPose_.z();
      laserOdometryROS.pose.pose.orientation = odom_quat;
      pose_pub.publish(laserOdometryROS);

      points_pre_time = points_curr_time;

//      pre_cloud=filtered;
      ++key_count;

      auto t2 = ros::WallTime::now();
      NODELET_INFO_STREAM("processing_time: " <<(t2 - t1).toSec()* 1000.0 << "[msec]");

  }

  /**
   * @brief callback for globalmap input
   * @param points_msg
   */
  void globalmap_callback(const sensor_msgs::PointCloud2ConstPtr& points_msg) {
    NODELET_INFO("globalmap received!");
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
    pcl::fromROSMsg(*points_msg, *cloud);
    globalmap = cloud;
    registration->setInputTarget(globalmap);
  }

  /**
   * @brief callback for initial pose input ("2D Pose Estimate" on rviz)
   * @param pose_msg
   */
  void initialpose_callback(const geometry_msgs::PoseWithCovarianceStampedConstPtr& pose_msg) {
    NODELET_INFO("initial pose received!!");
    std::lock_guard<std::mutex> lock(pose_estimator_mutex);
    const auto& p = pose_msg->pose.pose.position;
    const auto& q = pose_msg->pose.pose.orientation;
    pose_estimator.reset(
          new hdl_localization::PoseEstimator(
            registration,
            ros::Time::now(),
            Eigen::Vector3d(p.x, p.y, p.z),
            Eigen::Quaterniond(q.w, q.x, q.y, q.z),
            private_nh.param<double>("cool_time_duration", 0.5))
    );
  }

  /**
   * @brief downsampling
   * @param cloud   input cloud
   * @return downsampled cloud
   */
  pcl::PointCloud<PointT>::ConstPtr downsample(const pcl::PointCloud<PointT>::ConstPtr& cloud) const {
    if(!downsample_filter) {
      return cloud;
    }
    pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
    downsample_filter->setInputCloud(cloud);
    downsample_filter->filter(*filtered);
    filtered->header = cloud->header;
    return filtered;
  }

  /**
   * @brief publish odometry
   * @param stamp  timestamp
   * @param pose   odometry pose to be published
   */

  void publish_odometry(const ros::Time& stamp, const Eigen::Matrix4d& pose) {
    // broadcast the transform over tf
    geometry_msgs::TransformStamped odom_trans = matrix2transform(stamp, pose, "map", odom_child_frame_id);
    pose_broadcaster.sendTransform(odom_trans);

    // publish the transform
    nav_msgs::Odometry odom;
    odom.header.stamp = stamp;
    odom.header.frame_id = "map";

    odom.pose.pose.position.x = pose(0, 3);
    odom.pose.pose.position.y = pose(1, 3);
    odom.pose.pose.position.z = pose(2, 3);
    odom.pose.pose.orientation = odom_trans.transform.rotation;

    odom.child_frame_id = odom_child_frame_id;
    odom.twist.twist.linear.x = 0.0;
    odom.twist.twist.linear.y = 0.0;
    odom.twist.twist.angular.z = 0.0;

    pose_pub.publish(odom);
  }

    void publish_incre_odometry(const ros::Time& stamp, const Eigen::Matrix4d& pose) {
        // broadcast the transform over tf
        geometry_msgs::TransformStamped odom_trans = matrix2transform(stamp, pose, odom_child_frame_id, "incre_baselink");
        pose_incre_broadcaster.sendTransform(odom_trans);

        // publish the transform
        nav_msgs::Odometry odom;
        odom.header.stamp = stamp;
        odom.header.frame_id = odom_child_frame_id;

        odom.pose.pose.position.x = pose(0, 3);
        odom.pose.pose.position.y = pose(1, 3);
        odom.pose.pose.position.z = pose(2, 3);
        odom.pose.pose.orientation = odom_trans.transform.rotation;

        odom.child_frame_id = "incre_baselink";
        odom.twist.twist.linear.x = 0.0;
        odom.twist.twist.linear.y = 0.0;
        odom.twist.twist.angular.z = 0.0;

        pose_incre_pub.publish(odom);
    }

  /**
   * @brief convert a Eigen::Matrix to TransformedStamped
   * @param stamp           timestamp
   * @param pose            pose matrix
   * @param frame_id        frame_id
   * @param child_frame_id  child_frame_id
   * @return transform
   */
  geometry_msgs::TransformStamped matrix2transform(const ros::Time& stamp, const Eigen::Matrix4d& pose, const std::string& frame_id, const std::string& child_frame_id) {
    Eigen::Quaterniond quat(pose.block<3, 3>(0, 0));
    quat.normalize();
    geometry_msgs::Quaternion odom_quat;
    odom_quat.w = quat.w();
    odom_quat.x = quat.x();
    odom_quat.y = quat.y();
    odom_quat.z = quat.z();

    geometry_msgs::TransformStamped odom_trans;
    odom_trans.header.stamp = stamp;
    odom_trans.header.frame_id = frame_id;
    odom_trans.child_frame_id = child_frame_id;

    odom_trans.transform.translation.x = pose(0, 3);
    odom_trans.transform.translation.y = pose(1, 3);
    odom_trans.transform.translation.z = pose(2, 3);
    odom_trans.transform.rotation = odom_quat;

    return odom_trans;
  }

    void resetOptimization()
    {

        std::cout<<"resetOptimization"<<std::endl;
        gtsam::ISAM2Params optParameters;
        optParameters.relinearizeThreshold = 0.1;
        optParameters.relinearizeSkip = 1;
        optimizer = new gtsam::ISAM2(optParameters);
        gtsam::NonlinearFactorGraph newGraphFactors;
        newgraph = newGraphFactors;
        gtsam::Values NewGraphValues;
        initialEstimate = NewGraphValues;

//        gtsam::LevenbergMarquardtParams optParameters;
//        optParameters.setVerbosity("ERROR");
//        optParameters.setOrderingType("METIS");
//        optParameters.setLinearSolverType("MULTIFRONTAL_CHOLESKY");
//        gtsam::NonlinearFactorGraph newGraphFactors;
//        newgraph = newGraphFactors;
//        gtsam::Values NewGraphValues;
//        initialEstimate = NewGraphValues;
//        optimizer= new gtsam::LevenbergMarquardtOptimizer optimizer(newgraph,initialEstimate,optParameters);


    }

private:
  // ROS
  ros::NodeHandle nh;
  ros::NodeHandle mt_nh;
  ros::NodeHandle private_nh;
  ros::Publisher pose_incre_pub;
  ros::Publisher pose_pub;
  
  std::string odom_child_frame_id;

  bool use_imu;
  bool invert_imu;

  float key_interval;
  int opt_frames;
  ros::Subscriber imu_sub;
  ros::Subscriber points_sub;
  ros::Subscriber globalmap_sub;
  ros::Subscriber initialpose_sub;
  ros::Subscriber updateGlobalPose_sub;

  ros::Time points_pre_time,points_curr_time;

  ros::Publisher aligned_pub;
  tf::TransformBroadcaster pose_broadcaster;
  tf::TransformBroadcaster pose_incre_broadcaster;
  tf::TransformListener tf_listener;

  // imu input buffer
  std::mutex imu_data_mutex;
  std::vector<sensor_msgs::ImuConstPtr> imu_data;

  // globalmap and registration method
  pcl::PointCloud<PointT>::Ptr globalmap;
  pcl::Filter<PointT>::Ptr downsample_filter;
  pcl::Registration<PointT, PointT>::Ptr registration;
  pcl::Registration<PointT, PointT>::Ptr localregistration;

  // pose estimator
  std::mutex pose_estimator_mutex;
  std::unique_ptr<hdl_localization::PoseEstimator> pose_estimator;


  // processing time buffer
  boost::circular_buffer<double> processing_time;

  // ********************the followings are for graph optimization*********************
public:
  bool firstkey=true;
  NonlinearFactorGraph newgraph;
  gtsam::ISAM2 *optimizer;
//  gtsam::LevenbergMarquardtOptimizer *optimizer;
  gtsam::Values initialEstimate;
  gtsam::Values isamCurrentEstimate;
  gtsam::PreintegratedImuMeasurements *imuIntegratorImu_;
  bool systemInitialized;
  int key_count;
  double imuAccNoise,imuGyrNoise,gravity;
  double odomLinearNoise,odomAngularNoise;

  gtsam::noiseModel::Diagonal::shared_ptr priorPoseNoise;
  gtsam::noiseModel::Diagonal::shared_ptr priorVelNoise;
  gtsam::noiseModel::Diagonal::shared_ptr priorBiasNoise;
  gtsam::noiseModel::Diagonal::shared_ptr correctionNoise;
  gtsam::Vector noiseModelBetweenBias;

//  pcl::PointCloud<PointT>::Ptr pre_cloud;  //registration for local odometry
//  pcl::PointCloud<PointT>::Ptr cur_cloud;  //registration for local odometry
  gtsam::Vector3 prevVel_;
  gtsam::imuBias::ConstantBias prevBias_;
  gtsam::NavState prevState_;
  gtsam::Pose3 prevPose_;
//  gtsam::Pose3 poseFrom;


};

}


PLUGINLIB_EXPORT_CLASS(hdl_localization::HdlLocalizationNodelet, nodelet::Nodelet)
