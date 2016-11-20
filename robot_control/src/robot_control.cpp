#include "robot_control.h"

RobotControl::RobotControl()
{

}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "robot_control");
    RobotControl rc;
    ros::spin();

    return 0;
}
