#include <mpvo_local_planner/mpvo_planner.h>
#include <base_local_planner/goal_functions.h>
#include <cmath>
#include <queue>
#include <angles/angles.h>
#include <ros/ros.h>
#include <tf2/utils.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>

#define Ro 0.15 //障害物半径
#define Rr 0.105 //ロボット半径
#define PI 3.1416

using namespace std;

namespace mpvo_local_planner {
  void MPVOPlanner::reconfigure(MPVOPlannerConfig &config)
  {

    boost::mutex::scoped_lock l(configuration_mutex_);

    generator_.setParameters(
        config.sim_time,
        config.sim_granularity,
        config.angular_sim_granularity,
        config.use_mpvo,
        sim_period_);

    double resolution = planner_util_->getCostmap()->getResolution();
    path_distance_bias_ = resolution * config.path_distance_bias;
    // pdistscale used for both path and alignment, set  forward_point_distance to zero to discard alignment
    path_costs_.setScale(path_distance_bias_);
    alignment_costs_.setScale(path_distance_bias_);

    goal_distance_bias_ = resolution * config.goal_distance_bias;
    goal_costs_.setScale(goal_distance_bias_);
    goal_front_costs_.setScale(goal_distance_bias_);

    occdist_scale_ = config.occdist_scale;
    obstacle_costs_.setScale(occdist_scale_);

    stop_time_buffer_ = config.stop_time_buffer;
    oscillation_costs_.setOscillationResetDist(config.oscillation_reset_dist, config.oscillation_reset_angle);
    forward_point_distance_ = config.forward_point_distance;
    goal_front_costs_.setXShift(forward_point_distance_);
    alignment_costs_.setXShift(forward_point_distance_);

    // obstacle costs can vary due to scaling footprint feature
    obstacle_costs_.setParams(config.max_vel_trans, config.max_scaling_factor, config.scaling_speed);

    twirling_costs_.setScale(config.twirling_scale);

    int vx_samp, vy_samp, vth_samp;
    vx_samp = config.vx_samples;
    vy_samp = config.vy_samples;
    vth_samp = config.vth_samples;

    if (vx_samp <= 0) {
      ROS_WARN("You've specified that you don't want any samples in the x dimension. We'll at least assume that you want to sample one value... so we're going to set vx_samples to 1 instead");
      vx_samp = 1;
      config.vx_samples = vx_samp;
    }

    if (vy_samp <= 0) {
      ROS_WARN("You've specified that you don't want any samples in the y dimension. We'll at least assume that you want to sample one value... so we're going to set vy_samples to 1 instead");
      vy_samp = 1;
      config.vy_samples = vy_samp;
    }

    if (vth_samp <= 0) {
      ROS_WARN("You've specified that you don't want any samples in the th dimension. We'll at least assume that you want to sample one value... so we're going to set vth_samples to 1 instead");
      vth_samp = 1;
      config.vth_samples = vth_samp;
    }

    vsamples_[0] = vx_samp;
    vsamples_[1] = vy_samp;
    vsamples_[2] = vth_samp;


  }

  MPVOPlanner::MPVOPlanner(std::string name, base_local_planner::LocalPlannerUtil *planner_util) :
      planner_util_(planner_util),
      obstacle_costs_(planner_util->getCostmap()),
      path_costs_(planner_util->getCostmap()),
      goal_costs_(planner_util->getCostmap(), 0.0, 0.0, true),
      goal_front_costs_(planner_util->getCostmap(), 0.0, 0.0, true),
      alignment_costs_(planner_util->getCostmap())
  {
    ros::NodeHandle private_nh("~/" + name);

    goal_front_costs_.setStopOnFailure( false );
    alignment_costs_.setStopOnFailure( false );

    //Assuming this planner is being run within the navigation stack, we can
    //just do an upward search for the frequency at which its being run. This
    //also allows the frequency to be overwritten locally.
    std::string controller_frequency_param_name;
    if(!private_nh.searchParam("controller_frequency", controller_frequency_param_name)) {
      sim_period_ = 0.05;
    } else {
      double controller_frequency = 0;
      private_nh.param(controller_frequency_param_name, controller_frequency, 20.0);
      if(controller_frequency > 0) {
        sim_period_ = 1.0 / controller_frequency;
      } else {
        ROS_WARN("A controller_frequency less than 0 has been set. Ignoring the parameter, assuming a rate of 20Hz");
        sim_period_ = 0.05;
      }
    }
    ROS_INFO("Sim period is set to %.2f", sim_period_);

    oscillation_costs_.resetOscillationFlags();

    bool sum_scores;
    private_nh.param("sum_scores", sum_scores, false);
    obstacle_costs_.setSumScores(sum_scores);


    private_nh.param("publish_cost_grid_pc", publish_cost_grid_pc_, false);
    map_viz_.initialize(name, planner_util->getGlobalFrame(), boost::bind(&MPVOPlanner::getCellCosts, this, _1, _2, _3, _4, _5, _6));

    private_nh.param("global_frame_id", frame_id_, std::string("odom"));

    traj_cloud_pub_ = private_nh.advertise<sensor_msgs::PointCloud2>("trajectory_cloud", 1);
    private_nh.param("publish_traj_pc", publish_traj_pc_, false);

    // set up all the cost functions that will be applied in order
    // (any function returning negative values will abort scoring, so the order can improve performance)
    std::vector<base_local_planner::TrajectoryCostFunction*> critics;
    critics.push_back(&oscillation_costs_); // discards oscillating motions (assisgns cost -1)
    critics.push_back(&obstacle_costs_); // discards trajectories that move into obstacles
    critics.push_back(&goal_front_costs_); // prefers trajectories that make the nose go towards (local) nose goal
    critics.push_back(&alignment_costs_); // prefers trajectories that keep the robot nose on nose path
    critics.push_back(&path_costs_); // prefers trajectories on global path
    critics.push_back(&goal_costs_); // prefers trajectories that go towards (local) goal, based on wave propagation
    critics.push_back(&twirling_costs_); // optionally prefer trajectories that don't spin

    // trajectory generators
    std::vector<base_local_planner::TrajectorySampleGenerator*> generator_list;
    generator_list.push_back(&generator_);

    scored_sampling_planner_ = base_local_planner::SimpleScoredSamplingPlanner(generator_list, critics);

    private_nh.param("cheat_factor", cheat_factor_, 1.0);
  }

  // used for visualization only, total_costs are not really total costs
  bool MPVOPlanner::getCellCosts(int cx, int cy, float &path_cost, float &goal_cost, float &occ_cost, float &total_cost) {

    path_cost = path_costs_.getCellCosts(cx, cy);
    goal_cost = goal_costs_.getCellCosts(cx, cy);
    occ_cost = planner_util_->getCostmap()->getCost(cx, cy);
    if (path_cost == path_costs_.obstacleCosts() ||
        path_cost == path_costs_.unreachableCellCosts() ||
        occ_cost >= costmap_2d::INSCRIBED_INFLATED_OBSTACLE) {
      return false;
    }

    total_cost =
        path_distance_bias_ * path_cost +
        goal_distance_bias_ * goal_cost +
        occdist_scale_ * occ_cost;
    return true;
  }

  bool MPVOPlanner::setPlan(const std::vector<geometry_msgs::PoseStamped>& orig_global_plan) {
    oscillation_costs_.resetOscillationFlags();
    return planner_util_->setPlan(orig_global_plan);
  }

  /**
   * This function is used when other strategies are to be applied,
   * but the cost functions for obstacles are to be reused.
   */
  bool MPVOPlanner::checkTrajectory(
      Eigen::Vector3f pos,
      Eigen::Vector3f vel,
      Eigen::Vector3f vel_samples){
    oscillation_costs_.resetOscillationFlags();
    base_local_planner::Trajectory traj;
    geometry_msgs::PoseStamped goal_pose = global_plan_.back();
    Eigen::Vector3f goal(goal_pose.pose.position.x, goal_pose.pose.position.y, tf2::getYaw(goal_pose.pose.orientation));
    base_local_planner::LocalPlannerLimits limits = planner_util_->getCurrentLimits();
    generator_.initialise(pos,
        vel,
        goal,
        &limits,
        vsamples_);
    generator_.generateTrajectory(pos, vel, vel_samples, traj);
    double cost = scored_sampling_planner_.scoreTrajectory(traj, -1);
    //if the trajectory is a legal one... the check passes
    if(cost >= 0) {
      return true;
    }
    ROS_WARN("Invalid Trajectory %f, %f, %f, cost: %f", vel_samples[0], vel_samples[1], vel_samples[2], cost);

    //otherwise the check fails
    return false;
  }


  void MPVOPlanner::updatePlanAndLocalCosts(
      const geometry_msgs::PoseStamped& global_pose,
      const std::vector<geometry_msgs::PoseStamped>& new_plan,
      const std::vector<geometry_msgs::Point>& footprint_spec) {
    global_plan_.resize(new_plan.size());
    for (unsigned int i = 0; i < new_plan.size(); ++i) {
      global_plan_[i] = new_plan[i];
    }

    obstacle_costs_.setFootprint(footprint_spec);

    // costs for going away from path
    path_costs_.setTargetPoses(global_plan_);

    // costs for not going towards the local goal as much as possible
    goal_costs_.setTargetPoses(global_plan_);

    // alignment costs
    geometry_msgs::PoseStamped goal_pose = global_plan_.back();

    Eigen::Vector3f pos(global_pose.pose.position.x, global_pose.pose.position.y, tf2::getYaw(global_pose.pose.orientation));
    double sq_dist =
        (pos[0] - goal_pose.pose.position.x) * (pos[0] - goal_pose.pose.position.x) +
        (pos[1] - goal_pose.pose.position.y) * (pos[1] - goal_pose.pose.position.y);

    // we want the robot nose to be drawn to its final position
    // (before robot turns towards goal orientation), not the end of the
    // path for the robot center. Choosing the final position after
    // turning towards goal orientation causes instability when the
    // robot needs to make a 180 degree turn at the end
    std::vector<geometry_msgs::PoseStamped> front_global_plan = global_plan_;
    double angle_to_goal = atan2(goal_pose.pose.position.y - pos[1], goal_pose.pose.position.x - pos[0]);
    front_global_plan.back().pose.position.x = front_global_plan.back().pose.position.x +
      forward_point_distance_ * cos(angle_to_goal);
    front_global_plan.back().pose.position.y = front_global_plan.back().pose.position.y + forward_point_distance_ *
      sin(angle_to_goal);

    goal_front_costs_.setTargetPoses(front_global_plan);

    // keeping the nose on the path
    if (sq_dist > forward_point_distance_ * forward_point_distance_ * cheat_factor_) {
      alignment_costs_.setScale(path_distance_bias_);
      // costs for robot being aligned with path (nose on path, not ju
      alignment_costs_.setTargetPoses(global_plan_);
    } else {
      // once we are close to goal, trying to keep the nose close to anything destabilizes behavior.
      alignment_costs_.setScale(0.0);
    }
  }


  /*
   * given the current state of the robot, find a good trajectory
   */
  base_local_planner::Trajectory MPVOPlanner::findBestPath(
      const geometry_msgs::PoseStamped& global_pose,
      const geometry_msgs::PoseStamped& global_vel,
      geometry_msgs::PoseStamped& drive_velocities) {

    //make sure that our configuration doesn't change mid-run
    boost::mutex::scoped_lock l(configuration_mutex_);

    Eigen::Vector3f pos(global_pose.pose.position.x, global_pose.pose.position.y, tf2::getYaw(global_pose.pose.orientation));
    Eigen::Vector3f vel(global_vel.pose.position.x, global_vel.pose.position.y, tf2::getYaw(global_vel.pose.orientation));
    geometry_msgs::PoseStamped goal_pose = global_plan_.back();
    Eigen::Vector3f goal(goal_pose.pose.position.x, goal_pose.pose.position.y, tf2::getYaw(goal_pose.pose.orientation));
    base_local_planner::LocalPlannerLimits limits = planner_util_->getCurrentLimits();

    // prepare cost functions and generators for this run
    generator_.initialise(pos,
        vel,
        goal,
        &limits,
        vsamples_);

    result_traj_.cost_ = -7;
    // find best trajectory by sampling and scoring the samples
    std::vector<base_local_planner::Trajectory> all_explored;
    scored_sampling_planner_.findBestTrajectory(result_traj_, &all_explored);

    if(publish_traj_pc_)
    {
        sensor_msgs::PointCloud2 traj_cloud;
        traj_cloud.header.frame_id = frame_id_;
        traj_cloud.header.stamp = ros::Time::now();

        sensor_msgs::PointCloud2Modifier cloud_mod(traj_cloud);
        cloud_mod.setPointCloud2Fields(5, "x", 1, sensor_msgs::PointField::FLOAT32,
                                          "y", 1, sensor_msgs::PointField::FLOAT32,
                                          "z", 1, sensor_msgs::PointField::FLOAT32,
                                          "theta", 1, sensor_msgs::PointField::FLOAT32,
                                          "cost", 1, sensor_msgs::PointField::FLOAT32);

        unsigned int num_points = 0;
        for(std::vector<base_local_planner::Trajectory>::iterator t=all_explored.begin(); t != all_explored.end(); ++t)
        {
            if (t->cost_<0)
              continue;
            num_points += t->getPointsSize();
        }

        cloud_mod.resize(num_points);
        sensor_msgs::PointCloud2Iterator<float> iter_x(traj_cloud, "x");
        for(std::vector<base_local_planner::Trajectory>::iterator t=all_explored.begin(); t != all_explored.end(); ++t)
        {
            if(t->cost_<0)
                continue;
            // Fill out the plan
            for(unsigned int i = 0; i < t->getPointsSize(); ++i) {
                double p_x, p_y, p_th;
                t->getPoint(i, p_x, p_y, p_th);
                iter_x[0] = p_x;
                iter_x[1] = p_y;
                iter_x[2] = 0.0;
                iter_x[3] = p_th;
                iter_x[4] = t->cost_;
                ++iter_x;
            }
        }
        traj_cloud_pub_.publish(traj_cloud);
    }

    // verbose publishing of point clouds
    if (publish_cost_grid_pc_) {
      //we'll publish the visualization of the costs to rviz before returning our best trajectory
      map_viz_.publishCostCloud(planner_util_->getCostmap());
    }

    // debrief stateful scoring functions
    oscillation_costs_.updateOscillationFlags(pos, &result_traj_, planner_util_->getCurrentLimits().min_vel_trans);

    //if we don't have a legal trajectory, we'll just command zero
    if (result_traj_.cost_ < 0) {
      drive_velocities.pose.position.x = 0;
      drive_velocities.pose.position.y = 0;
      drive_velocities.pose.position.z = 0;
      drive_velocities.pose.orientation.w = 1;
      drive_velocities.pose.orientation.x = 0;
      drive_velocities.pose.orientation.y = 0;
      drive_velocities.pose.orientation.z = 0;
    } else {
      drive_velocities.pose.position.x = result_traj_.xv_;
      drive_velocities.pose.position.y = result_traj_.yv_;
      drive_velocities.pose.position.z = 0;
      tf2::Quaternion q;
      q.setRPY(0, 0, result_traj_.thetav_);
      tf2::convert(q, drive_velocities.pose.orientation);
    }

    return result_traj_;
  }


  /***********************
	***** 速度指令値生成 *****
	***********************/
/******************************************************************************/
	void MPVOPlanner::cmdVelocity(
			double pr_x,
	    double pr_y,
      double theta_ro,
	    double v_ro,
	    double w_ro,
	    double &cmd_v,
	    double &cmd_w)
	{
    /************************
    ** Vs：ロボットの速度制限 **
    ***********************/
    double v_max, v_min, w_max, w_min;
    v_max = 0.15; //最大速度[m/s]
    v_min = 0.0; //最小速度[m/s]
    w_max = 2.75; //最大角速度[rad/s]
    w_min = -2.75; //最小角速度[rad/s]

    /***************************
    ** Vd：ダイナミックウィンドウ **
    ***************************/
    double dv, dw; //最大加速度
    dv = 2.5;
    dw = 3.2;
    double Ts = 0.1; //反復周期[s]
    double vd_max, vd_min, wd_max, wd_min;
    vd_max = v_ro + dv * Ts;
    if(vd_max > v_max){
      vd_max = v_max;
    }

    vd_min = v_ro - dv * Ts;
    if(vd_min < v_min){
      vd_min = v_min;
    }

    wd_max = w_ro + dw * Ts;
    if(wd_max > w_max){
      wd_max = w_max;
    }

    wd_min = w_ro - dw * Ts;
    if(wd_min < w_min){
      wd_min = w_min;
    }

    /************************
    ** m*n個の組(v[m],w[n]) **
    ***********************/
    double width_v, width_w; //各速度の間隔
    width_v = (vd_max - vd_min) / 4;
    width_w = (wd_max - wd_min) / 4;
    double v[] = {vd_min, vd_min + width_v, vd_min + width_v * 2, vd_min + width_v * 3, vd_max};
    double w[] = {wd_min, wd_min + width_w, wd_min + width_w * 2, wd_min + width_w * 3, wd_max};
    int m, n; //m,n番目

    /****************
    ** 各パラメータ **
    ***************/
    //ゴール
    double pg_x = 2.0; //位置x[m]
    double pg_y = 0.0; //位置y[m]
    double zeta = atan2((pg_y - pr_y), (pg_x - pr_x)); //方向[rad]

    //経過時間
    double t = ros::Time::now().toSec();
    cout << "t=" << t << endl;

    //障害物
    int k; //k番目の障害物
/*
    //障害物1個
    double po_x[] = {2.0 - 0.1 * t}; //位置x[m]
    double po_y[] = {0.0}; //位置y[m]
    double v_ob[] = {-0.1}; //速度[m/s]
    double a_o[] = {0.0}; //ax+by=c
    double b_o[] = {1.0};
    double c1[] = {Rr + Ro};
    double c2[] = {-(Rr + Ro)};
    double c;

    //障害物2個
    double po_x[] = {2.0 - 0.1 * t, 0.0}; //位置x[m]
    double po_y[] = {0.0, 2.0 - 0.15 * t}; //位置y[m]
    double v_ob[] = {-0.1, -0.15}; //速度[m/s]
    double a_o[] = {0.0, 1.0}; //ax+by=c
    double b_o[] = {1.0, 0.0};
    double c1[] = {Rr + Ro, Rr + Ro};
    double c2[] = {-(Rr + Ro), -(Rr + Ro)};
    double c;
*/
    //障害物3個
    double po_x[] = {-1.0, 0.0, 1.0}; //位置x[m]
    double po_y[] = {2.0 - 0.15 * t, -2.0 + 0.15 * t, 2.0 - 0.1 * t}; //位置y[m]
    double v_ob[] = {-0.15, 0.15, -0.1}; //速度[m/s]
    double a_o[] = {1.0, 1.0, 1.0}; //ax+by=c
    double b_o[] = {0.0, 0.0, 0.0};
    double c1[] = {-1 + Rr + Ro, Rr + Ro, 1 + Rr + Ro};
    double c2[] = {-1 -(Rr + Ro), -(Rr + Ro), 1 -(Rr + Ro)};
    double c;

    //ロボット
    double a_r, b_r; //直線軌道のときの傾きと切片
    double Rtrj; //円軌道の半径
    double cr_x, cr_y; //ロボット座標系における円軌道の中心位置
    double cw_x, cw_y; //世界座標系における円軌道の中心位置
    double X_r, Y_r; //ロボット座標系における有限時間先のロボットの予測位置
    double X_w, Y_w, theta_f; //世界座標系における有限時間先のロボットの予測位置、向き


    double dis; //ロボットと交点との間の距離
    double t_c; //交点までの到達時間
    double h_co; //円軌道の中心と障害物軌道の距離
    double h1_ro, h2_ro; //ロボットと障害物軌道の距離
    double A, B, C;
    double p1_x, p1_y, p2_x, p2_y; //世界座標系における障害物軌道との交点
    double rp1_x, rp1_y; //ロボット座標系における障害物軌道との交点
    double dX, dY;
    double delta; //交点までのロボットの回転角度
    bool judge; //trueのとき、その(v,w)の組は有効
    int flag = 0;
    double min; //ゴールまでの距離の最小値

/*
    //障害物1個
    double T_lon = 6.5; //制限時間
    double T_lout = 5.0; //制限時間

    //障害物2個
    double T_lon = 6.0; //制限時間
    double T_lout = 5.0; //制限時間
*/
    //障害物3個
    double T_lon = 6.0; //制限時間
    double T_lout = 9.0; //制限時間

    double T_lg = 0.5; //ゴール直前の制限時間

    /*****************
    ** 速度指令値決定 **
    *****************/
    for(m = 0; m < 5; m++){
      for(n = 0; n < 5; n++){
        for(k = 0; k < 3; k++){
        /********************ロボットの現在位置が障害物の軌道上にあるとき********************/
          if((c2[k] <= a_o[k]*pr_x + b_o[k]*pr_y) && (a_o[k]*pr_x + b_o[k]*pr_y <= c1[k]) && ((a_o[k]==0.0 && pr_x <= po_x[k]) || (b_o[k]==0.0 && v_ob[k] < 0.0 && pr_y <= po_y[k]) || (b_o[k]==0.0 && v_ob[k] > 0.0 && pr_y >= po_y[k]))){
          /***************直線軌道のとき***************/
            if(w[n] == 0.0){
              //ロボット軌道y=ax+b
              a_r = tan(theta_ro); //ロボット軌道の傾き
              b_r = pr_y - tan(theta_ro) * pr_x; //ロボット軌道の切片
            /**********交点なし**********/
              if((a_o[k]==0.0 && a_r==0.0) || (b_o[k]==0.0 && abs(theta_ro)==PI/2)){
                judge = false;
              }
            /**********交点あり**********/
              else{
                //交点の座標
                if(a_o[k]==0.0){
                  p1_x = (c1[k] - b_r) / a_r;
                  p1_y = c1[k];
                  p2_x = (c2[k] - b_r) / a_r;
                  p2_y = c2[k];
                }else if(b_o[k]==0.0){
                  p1_x = c1[k];
                  p1_y = a_r * c1[k] + b_r;
                  p2_x = c2[k];
                  p2_y = a_r * c2[k] + b_r;
                }

                if(p1_x < p2_x){
                  p1_x = p2_x;
                  p1_y = p2_y;
                }
                //交点までの距離
                dis = sqrt(pow((p1_x - pr_x), 2) + pow((p1_y - pr_y), 2));
                //交点までの到達時間
                t_c = dis / v[m];
                //有限時間より小さいかつ交点が障害物位置より手前にあるときtrue
                if((t_c < T_lon) && ((a_o[k]==0.0 && p1_x < po_x[k]) || (b_o[k]==0.0 && v_ob[k] < 0.0 && p1_y < po_y[k]) || (b_o[k]==0.0 && v_ob[k] > 0.0 && p1_y > po_y[k]))){
                  judge = true;
                }else{
                  judge = false;
                }
              }
            }
          /***************円軌道のとき***************/
            else{
              //c1,c2を選択
              if(a_o[k]==0.0){
                if(w[n] > 0.0){
                  c = c1[k];
                }else{
                  c = c2[k];
                }
              }else if(b_o[k]==0.0){
                c = c1[k];
              }
              //ロボット座標系における円軌道の中心位置
              if(w[n] > 0.0){
                Rtrj = v[m] / w[n];
                cr_x = 0.0;
                cr_y = Rtrj;
              }else{
                Rtrj = v[m] / abs(w[n]);
                cr_x = 0.0;
                cr_y = -1 * Rtrj;
              }
              //中心位置をロボット座標系から世界座標系に変換
              cw_x = pr_x + cr_x * cos(theta_ro) - cr_y * sin(theta_ro);
              cw_y = pr_y + cr_x * sin(theta_ro) + cr_y * cos(theta_ro);
              //円軌道の中心と障害物軌道の距離
              h_co = abs(a_o[k] * cw_x + b_o[k] * cw_y - c) / sqrt(pow(a_o[k], 2) + pow(b_o[k], 2));
            /**********交点が2個**********/
              if(h_co < Rtrj){
                //交点の座標
                if(a_o[k] == 0.0){
                  A = 1.0;
                  B = -2 * cw_x;
                  C = pow(cw_x, 2) + pow((c - cw_y), 2) - pow(Rtrj, 2);
                  p1_x = (-B + sqrt(pow(B, 2) - 4 * A * C)) / (2 * A);
                  p1_y = c;
                  p2_x = (-B - sqrt(pow(B, 2) - 4 * A * C)) / (2 * A);
                  p2_y = c;
                  if(p1_x < p2_x){
                    p1_x = p2_x;
                    p1_y = p2_y;
                  }
                }else if(b_o[k] == 0.0){
                  A = 1.0;
                  B = -2 * cw_y;
                  C = pow(cw_y, 2) + pow((c - cw_x), 2) - pow(Rtrj, 2);
                  p1_x = c;
                  p1_y = (-B + sqrt(pow(B, 2) - 4 * A * C)) / (2 * A);
                  p2_x = c;
                  p2_y = (-B - sqrt(pow(B, 2) - 4 * A * C)) / (2 * A);
                  if(sqrt(pow((p1_x - pr_x), 2) + pow((p1_y - pr_y), 2)) > sqrt(pow((p2_x - pr_x), 2) + pow((p2_y - pr_y), 2))){
                    p1_x = p2_x;
                    p1_y = p2_y;
                  }
                }
                //交点をロボット座標系へ変換
                rp1_x = -pr_x + p1_x * cos(theta_ro) + p1_y * sin(theta_ro);
                rp1_y = -pr_y - p1_x * sin(theta_ro) + p1_y * cos(theta_ro);
                //なす角度を求める
                dY = rp1_x;
                dX = Rtrj - abs(rp1_y);
                delta = atan2(dY, dX);
                if(delta < 0.0){
                  delta += 2 * PI;
                }
                //交点までの時間
                t_c = delta / abs(w[n]);
                //有限時間より小さいかつ交点が障害物位置より手前にあるときtrue
                if((t_c < T_lon) && ((a_o[k]==0.0 && p1_x < po_x[k]) || (b_o[k]==0.0 && v_ob[k] < 0.0 && p1_y < po_y[k]) || (b_o[k]==0.0 && v_ob[k] > 0.0 && p1_y > po_y[k]))){
                  judge = true;
                }else{
                  judge = false;
                }
              }
            /**********交点が0または1個**********/
              else{
                judge = false;
              }

            }
          }
        /********************ロボットの現在位置が障害物の軌道上にないとき********************/
          else{
            //障害物軌道
            //ロボットと障害物軌道の距離の比較
            h1_ro = abs(a_o[k] * pr_x + b_o[k] * pr_y - c1[k]) / sqrt(pow(a_o[k], 2) + pow(b_o[k], 2));
            h2_ro = abs(a_o[k] * pr_x + b_o[k] * pr_y - c2[k]) / sqrt(pow(a_o[k], 2) + pow(b_o[k], 2));
            if(h1_ro <= h2_ro){
              c = c1[k];
            }else{
              c = c2[k];
            }

          /***************直線軌道のとき***************/
            if(w[n] == 0.0){
              //ロボット軌道y=ax+b
              a_r = tan(theta_ro); //ロボット軌道の傾き
              b_r = pr_y - tan(theta_ro) * pr_x; //ロボット軌道の切片

            /**********交点なし**********/
              if((a_o[k]==0.0 && a_r==0.0) || (b_o[k]==0.0 && abs(theta_ro)==PI/2)){
                judge = true;
              }
            /*****交点あり*****/
              else{
                //交点の座標
                if(a_o[k]==0.0){
                  p1_x = (c - b_r) / a_r;
                  p1_y = c;
                }else if(b_o[k]==0.0){
                  p1_x = c;
                  p1_y = a_r * c + b_r;
                }
                //交点の位置がロボットより前方
                if(p1_x >= pr_x){
                  //交点までの距離、到達時間
                  dis = sqrt(pow((p1_x - pr_x), 2) + pow((p1_y - pr_y), 2));
                  t_c = dis / v[m];
                  //有限時間より大きいまたは交点が障害物位置より先にあるときtrue
                  if((t_c > T_lout) || ((a_o[k]==0.0 && p1_x > po_x[k]) || (b_o[k]==0.0 && v_ob[k] < 0.0 && p1_y > po_y[k]) || (b_o[k]==0.0 && v_ob[k] > 0.0 && p1_y < po_y[k]))){
                    judge = true;
                  }else{
                    judge = false;
                  }
                }
                //交点の位置がロボットより後方
                else{
                  judge = true;
                }
              }
            }
          /**********円軌道のとき**********/
            else{
              if(w[n] > 0.0){
                Rtrj = v[m] / w[n];
                cr_x = 0.0;
                cr_y = Rtrj;
              }else{
                Rtrj = v[m] / abs(w[n]);
                cr_x = 0.0;
                cr_y = -1 * Rtrj;
              }
              //中心位置をロボット座標系から世界座標系に変換
              cw_x = pr_x + cr_x * cos(theta_ro) - cr_y * sin(theta_ro);
              cw_y = pr_y + cr_x * sin(theta_ro) + cr_y * cos(theta_ro);

              //円軌道の中心と障害物軌道の距離
              h_co = abs(a_o[k] * cw_x + b_o[k] * cw_y - c) / sqrt(pow(a_o[k], 2) + pow(b_o[k], 2));

            /*****交点あり*****/
              if(h_co <= Rtrj){
                //交点の座標
                //交点が1つ
                if(h_co == Rtrj){
                  if(a_o[k] == 0.0){
                    A = 1.0;
                    B = -2 * cw_x;
                    C = pow(cw_x, 2) + pow((c - cw_y), 2) - pow(Rtrj, 2);
                    p1_x = -B / (2 * A);
                    p1_y = c;
                  }else if(b_o[k] == 0.0){
                    A = 1.0;
                    B = -2 * cw_y;
                    C = pow(cw_y, 2) + pow((c - cw_x), 2) - pow(Rtrj, 2);
                    p1_x = c;
                    p1_y = -B / (2 * A);
                  }
                }
                //交点が2つ
                else{
                  if(a_o[k] == 0.0){
                    A = 1.0;
                    B = -2 * cw_x;
                    C = pow(cw_x, 2) + pow((c - cw_y), 2) - pow(Rtrj, 2);
                    p1_x = (-B + sqrt(pow(B, 2) - 4 * A * C)) / (2 * A);
                    p1_y = c;
                    p2_x = (-B - sqrt(pow(B, 2) - 4 * A * C)) / (2 * A);
                    p2_y = c;
                    if(p1_x < p2_x){
                      p1_x = p2_x;
                      p1_y = p2_y;
                    }
                  }else if(b_o[k] == 0.0){
                    A = 1.0;
                    B = -2 * cw_y;
                    C = pow(cw_y, 2) + pow((c - cw_x), 2) - pow(Rtrj, 2);
                    p1_x = c;
                    p1_y = (-B + sqrt(pow(B, 2) - 4 * A * C)) / (2 * A);
                    p2_x = c;
                    p2_y = (-B - sqrt(pow(B, 2) - 4 * A * C)) / (2 * A);
                    if(sqrt(pow((p1_x - pr_x), 2) + pow((p1_y - pr_y), 2)) > sqrt(pow((p2_x - pr_x), 2) + pow((p2_y - pr_y), 2))){
                      p1_x = p2_x;
                      p1_y = p2_y;
                    }
                  }
                }

                //交点の位置がロボットより前方
                if(p1_x >= pr_x){

                  //交点をロボット座標系へ変換
                  rp1_x = -pr_x + p1_x * cos(theta_ro) + p1_y * sin(theta_ro);
                  rp1_y = -pr_y - p1_x * sin(theta_ro) + p1_y * cos(theta_ro);
                  //なす角度を求める
                  dY = rp1_x;
                  dX = Rtrj - abs(rp1_y);
                  delta = atan2(dY, dX);
                  if(delta < 0.0){
                    delta += 2 * PI;
                  }
                  //交点までの時間
                  t_c = delta / abs(w[n]);
                  //有限時間より大きいまたは交点が障害物位置より先にあるときtrue
                  if((t_c > T_lout) || ((a_o[k]==0.0 && p1_x > po_x[k]) || (b_o[k]==0.0 && v_ob[k] < 0.0 && p1_y > po_y[k]) || (b_o[k]==0.0 && v_ob[k] > 0.0 && p1_y < po_y[k]))){
                    judge = true;
                  }else{
                    judge = false;
                  }
                }
                //交点の位置がロボットより後方
                else{
                  judge = true;
                }
              }
            /*****交点なし*****/
              else{
                judge = true;
              }

            }
          }

          //judgeがfalseのときループ抜ける
          if(judge){
          }else{
            break;
          }

        } //for k

      /***************trueのとき、有効な(v,w)の組の中から最適なものを選択***************/
        if(judge){
        /*****有限時間先のロボットの予測位置*****/
          //ゴール直前のとき制限時間を短くする
          if(sqrt(pow((pr_x - pg_x), 2) + pow((pr_y - pg_y), 2)) <= 1.0){
            if(w[n] == 0.0){
              X_w = pr_x + v[m] * cos(theta_ro) * T_lg;
              Y_w = pr_y + v[m] * sin(theta_ro) * T_lg;
            }else{
              if(w[n] > 0.0){
                X_r = Rtrj * sin(w[n] * T_lg);
                Y_r = Rtrj * (1 - cos(w[n] * T_lg));
              }else{
                X_r = Rtrj * sin(abs(w[n]) * T_lg);
                Y_r = -1 * Rtrj * (1 - cos(abs(w[n]) * T_lg));
              }
              X_w = pr_x + X_r * cos(theta_ro) - Y_r * sin(theta_ro);
              Y_w = pr_y + X_r * sin(theta_ro) + Y_r * cos(theta_ro);
            }
          }else{
            if(w[n] == 0.0){
              X_w = pr_x + v[m] * cos(theta_ro) * T_lout;
              Y_w = pr_y + v[m] * sin(theta_ro) * T_lout;
            }else{
              if(w[n] > 0.0){
                X_r = Rtrj * sin(w[n] * T_lout);
                Y_r = Rtrj * (1 - cos(w[n] * T_lout));
              }else{
                X_r = Rtrj * sin(abs(w[n]) * T_lout);
                Y_r = -1 * Rtrj * (1 - cos(abs(w[n]) * T_lout));
              }
              X_w = pr_x + X_r * cos(theta_ro) - Y_r * sin(theta_ro);
              Y_w = pr_y + X_r * sin(theta_ro) + Y_r * cos(theta_ro);
            }
          }

        /*****ゴールまでの距離が最も短くなるものを選択*****/
          if((min > sqrt(pow((pg_x - X_w), 2) + pow((pg_y - Y_w), 2))) && (flag == 1)){
						min = sqrt(pow((pg_x - X_w), 2) + pow((pg_y - Y_w), 2));
						cmd_v = v[m];
            cmd_w = w[n];
					}else if((min == sqrt(pow((pg_x - X_w), 2) + pow((pg_y - Y_w), 2))) && (flag == 1) && (v[m] > cmd_v)){
            min = sqrt(pow((pg_x - X_w), 2) + pow((pg_y - Y_w), 2));
						cmd_v = v[m];
            cmd_w = w[n];
          }

					if(flag == 0){
            min = sqrt(pow((pg_x - X_w), 2) + pow((pg_y - Y_w), 2));
						cmd_v = v[m];
            cmd_w = w[n];
						flag = 1; //初めて衝突なしの場合に１を与える
					}

        }

      }
    }

  /**************flag=0のときロボット停止**********/
    if(flag == 0){
      cmd_v = 0.0;
			cmd_w = 0.0;
			cout << "robot cannot move!" << endl;
    }


	}
/******************************************************************************/




};
