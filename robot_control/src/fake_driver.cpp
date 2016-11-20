#include "fake_driver.h"
#include <math.h>

FakeDriver::FakeDriver()
    : m_nh("~")
{
    control_freq = m_nh.param("controller_freq", 15.0);

    std::string grid_topic = m_nh.param("occupancy_grid", std::string(""));
    if(grid_topic.empty())
    {
        ROS_ERROR_STREAM("no occupancy_grid provided. Check launch file.");
        std::abort();
    }
    m_sub_grid = m_nh.subscribe(grid_topic, 1, &FakeDriver::handle_grid, this);

    std::string vel_topic = m_nh.param("vel_topic", std::string(""));
    if(vel_topic.empty())
    {
        ROS_ERROR_STREAM("no cmd_vel provided. Check launch file.");
        std::abort();
    }
//    m_sub_twist = m_nh.subscribe(vel_topic, 1, &FakeDriver::handle_twist, this);
    ros::NodeHandle nh;
    m_sub_twist = nh.subscribe("cmd_vel", 1, &FakeDriver::handle_twist, this);

    m_odom_trans.header.frame_id = grid_topic;
    m_odom_trans.child_frame_id = "bot";
    m_odom_trans.header.stamp = ros::Time::now();
    m_odom_trans.transform.translation.x = 2.5;
    m_odom_trans.transform.translation.y = 1.5;
    m_odom_trans.transform.translation.z = 0;
    m_odom_trans.transform.rotation = tf::createQuaternionMsgFromYaw(0);
    m_tf_boardcaster.sendTransform(m_odom_trans);

    ROS_INFO_STREAM("robot controller running at " << control_freq << "hz");
    ROS_INFO_STREAM("No movement until map is received.");
}

FakeDriver::~FakeDriver()
{

}

void FakeDriver::handle_twist(const geometry_msgs::TwistConstPtr &msg)
{
    m_twist_cmd = *msg;
}

void FakeDriver::handle_grid(const nav_msgs::OccupancyGridConstPtr &msg)
{
    m_grid = *msg;
    ROS_INFO_STREAM("Map received.");
}

void FakeDriver::step()
{
    m_odom_trans.header.stamp = ros::Time::now();
    double timelapsed = 1.0/control_freq;

    tf::Quaternion q;
    q.setZ(m_odom_trans.transform.rotation.z);
    q.setW(m_odom_trans.transform.rotation.w);
    double heading = q.getAngle() * (m_odom_trans.transform.rotation.z<0 ? -1 : 1);

    double next_x = m_odom_trans.transform.translation.x +
                    m_twist_cmd.linear.x*cos(heading)*timelapsed -
                    m_twist_cmd.linear.y*sin(heading)*timelapsed;
    double next_y = m_odom_trans.transform.translation.y +
                    m_twist_cmd.linear.x*sin(heading)*timelapsed +
                    m_twist_cmd.linear.y*cos(heading)*timelapsed;
    double next_heading = heading + m_twist_cmd.angular.z*timelapsed;
    next_heading = fmod(fabs(next_heading ), 2*M_PI) * ((next_heading <0) ? -1 : 1);

    if(m_grid.data.size() > 0)
    {
        double inv_res = 1.0 / m_grid.info.resolution;
        double a = 0.3*cos(heading)*inv_res;
        double b = 0.2*sin(heading)*inv_res;
        double c = 0.3*sin(heading)*inv_res;
        double d = 0.2*cos(heading)*inv_res;
        double x = next_x*inv_res;
        double y = next_y*inv_res;

        int x1 = x + a - b;
        int y1 = y + c + d;
        int x2 = x + a + b;
        int y2 = y + c - d;
        int x3 = x - a - b;
        int y3 = y - c + d;
        int x4 = x - a + b;
        int y4 = y - c - d;
        int x5 = x     - b;
        int y5 = y     + d;
        int x6 = x     + b;
        int y6 = y     - d;

//        tf::Transform transform;
//        transform.setOrigin( tf::Vector3(next_x+b, next_y-d, 0.0) );
//        transform.setRotation( tf::Quaternion(0, 0, 0, 1) );
//        m_tf_boardcaster.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "map", "asd"));

        if((m_grid.data[y1*m_grid.info.width + x1] != 0) ||
           (m_grid.data[y2*m_grid.info.width + x2] != 0) ||
           (m_grid.data[y3*m_grid.info.width + x3] != 0) ||
           (m_grid.data[y4*m_grid.info.width + x4] != 0) ||
           (m_grid.data[y5*m_grid.info.width + x5] != 0) ||
           (m_grid.data[y6*m_grid.info.width + x6] != 0))
        {
            ROS_WARN_STREAM("Collision! Motors disabled.");
            // publish collision msg
        }
        else
        {
            m_odom_trans.transform.translation.x = next_x;
            m_odom_trans.transform.translation.y = next_y;
            m_odom_trans.transform.translation.z = 0;
            m_odom_trans.transform.rotation = tf::createQuaternionMsgFromYaw(next_heading);
        }
        m_tf_boardcaster.sendTransform(m_odom_trans);
    }
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "fake_driver");
    FakeDriver fake_driver;
    ros::Rate controll_loop(fake_driver.control_freq);

    while(ros::ok())
    {
        ros::spinOnce();
        fake_driver.step();
        controll_loop.sleep();
    }

    return 0;
}
