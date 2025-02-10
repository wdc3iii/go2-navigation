#include "rclcpp/rclcpp.hpp"

#include "obelisk_estimator.h"
#include "obelisk_ros_utils.h"

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/exceptions.h>

namespace obelisk {
    class UnitreeGo2OdomEstimator
        : public obelisk::ObeliskEstimator<obelisk_estimator_msgs::msg::EstimatedState> {
    public:
        UnitreeGo2OdomEstimator(const std::string& name)
            : obelisk::ObeliskEstimator<obelisk_estimator_msgs::msg::EstimatedState>(name) {

            this->RegisterObkSubscription<obelisk_sensor_msgs::msg::ObkJointEncoders>(
                "sub_sensor_setting", "sub_sensor",
                std::bind(&UnitreeGo2OdomEstimator::JointEncoderCallback, this, std::placeholders::_1));

            this->RegisterObkSubscription<obelisk_sensor_msgs::msg::ObkImu>(
                "sub_imu_setting", "imu_sensor",
                std::bind(&UnitreeGo2OdomEstimator::TorsoIMUCallback, this, std::placeholders::_1));

            tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
            tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
        }

    protected:
        void JointEncoderCallback(const obelisk_sensor_msgs::msg::ObkJointEncoders& msg) {
            joint_encoders_ = msg.joint_pos;
            joint_vels_ = msg.joint_vel;
            joint_names_ = msg.joint_names;
        }

        void TorsoIMUCallback(const obelisk_sensor_msgs::msg::ObkImu& msg) {
            pose_ = {0., 0., 0., msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w};
            omega_ = {0., 0., 0., msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z};
        }

        obelisk_estimator_msgs::msg::EstimatedState ComputeStateEstimate() override {
            obelisk_estimator_msgs::msg::EstimatedState msg;

            msg.header.stamp = this->now();
            msg.base_link_name = "torso";
            
            try {
                geometry_msgs::msg::TransformStamped transform = 
                    tf_buffer_->lookupTransform("target_frame", "source_frame", tf2::TimePointZero);

                pose_[0] = transform.transform.translation.x;
                pose_[1] = transform.transform.translation.y;
                pose_[2] = transform.transform.translation.z;

                msg.q_joints       = joint_encoders_;   // Joint Positions
                msg.v_joints       = joint_vels_;        // Joint Velocities
                msg.joint_names    = joint_names_;      // Joint Names
                msg.q_base         = pose_;             // Quaternion
                msg.v_base         = omega_;            // Angular Velocity
                // msg.base_link_name = "link0";

                this->GetPublisher<obelisk_estimator_msgs::msg::EstimatedState>(this->est_pub_key_)->publish(msg);
            } catch (const tf2::TransformException &ex) {
                RCLCPP_WARN(this->get_logger(), "Could not transform: %s", ex.what());
            }

            return msg;
        };

    private:
        std::vector<double> joint_encoders_;
        std::vector<double> joint_vels_;
        std::vector<double> pose_;
        std::vector<double> omega_;
        std::vector<std::string> joint_names_;
        std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
        std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    };
}   // namespace obelisk