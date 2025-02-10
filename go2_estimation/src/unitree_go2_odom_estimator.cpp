#include "unitree_go2_odom_estimator.h"
#include "obelisk_ros_utils.h"

int main(int argc, char* argv[]) {
    obelisk::utils::SpinObelisk<obelisk::UnitreeGo2OdomEstimator, rclcpp::executors::SingleThreadedExecutor>(
        argc, argv, "go2_estimator");
}