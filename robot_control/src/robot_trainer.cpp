#include "robot_trainer.h"
#include <stdlib.h>
#include <time.h>
#include <nav_msgs/GetMap.h>
#include <nav_msgs/OccupancyGrid.h>

RobotTrainer::RobotTrainer()
    : m_nh("~")
    , m_client_move("move_base", 1)
{
    srand(time(NULL));
    while(!m_client_move.waitForServer(ros::Duration(5.0)))
    {
        ROS_INFO_STREAM("Waiting for move_base action server...");
    }

    m_sub_costmap = m_nh.subscribe("/move_base/global_costmap/costmap", 1, &RobotTrainer::handle_costmap, this);
    m_goal_move.target_pose.header.frame_id = "map";
    m_timer_traj = m_nh.createTimer(ros::Duration(10), &RobotTrainer::handle_timer_traj, this, true, false);

    ROS_INFO_STREAM("Robot Trainer ready.");
}

RobotTrainer::~RobotTrainer()
{

}

void RobotTrainer::handle_costmap(const nav_msgs::OccupancyGridConstPtr &msg)
{
    m_costmap = *msg;
    ROS_INFO_STREAM("Costmap received.");
    if(m_client_move.getState().toString() != "ACTIVE")
    {
        m_client_move.cancelAllGoals();
        new_goal();
    }
}

void RobotTrainer::new_goal()
{
    if(!m_client_move.isServerConnected())
    {
        ROS_INFO_STREAM("Server disconnected, waiting...");
        if(!m_client_move.waitForServer(ros::Duration(5)))
        {
            ROS_INFO_STREAM("No move_base action server...");
            return;
        }
    }

    tf::Quaternion q;
    q.setRPY(0,0, (rand()%360)*M_PI/180.0);
    m_goal_move.target_pose.pose.orientation.x = q.getX();
    m_goal_move.target_pose.pose.orientation.y = q.getY();
    m_goal_move.target_pose.pose.orientation.z = q.getZ();
    m_goal_move.target_pose.pose.orientation.w = q.getW();

    int x,y;
    while(1)
    {
        x = rand()%m_costmap.info.width;
        y = rand()%m_costmap.info.height;
//        ROS_INFO_STREAM("target:" << x*m_costmap.info.resolution << ", " << y*m_costmap.info.resolution);
        if(m_costmap.data[x+y*m_costmap.info.width] < 50)
            break;
    }
    m_goal_move.target_pose.pose.position.x = x*m_costmap.info.resolution;
    m_goal_move.target_pose.pose.position.y = y*m_costmap.info.resolution;
    m_goal_move.target_pose.header.stamp = ros::Time::now();
    m_client_move.sendGoal(m_goal_move,
                           boost::bind(&RobotTrainer::handle_move_done, this, _1, _2),
                           boost::bind(&RobotTrainer::handle_move_active, this),
                           movebase_act_client::SimpleFeedbackCallback());
    m_timer_traj.setPeriod(ros::Duration(20));
    m_timer_traj.start();
}

void RobotTrainer::handle_move_done(const actionlib::SimpleClientGoalState& state,
                                    const move_base_msgs::MoveBaseResultConstPtr& result)
{
    ROS_INFO_STREAM("Finished as " << state.toString());
    m_timer_traj.stop();
    new_goal();
}

void RobotTrainer::handle_move_active()
{
    ROS_INFO_STREAM("New goal went active. (" <<
                    m_goal_move.target_pose.pose.position.x << ", " <<
                    m_goal_move.target_pose.pose.position.y << ")");
}

void RobotTrainer::handle_timer_traj(const ros::TimerEvent &event)
{
    m_timer_traj.stop();
    if((m_client_move.getState().toString() == "ACTIVE") || (m_client_move.getState().toString() == "LOSS"))
    {
        m_client_move.cancelAllGoals();
        ROS_INFO_STREAM("goal expired, restarting.");
        new_goal();
    }
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "robot_trainer");
    RobotTrainer rt;
    ros::spin();

    return 0;
}
