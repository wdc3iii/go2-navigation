# go2-navigation
Repository aimed at running SLAM-based navigation on the Go2

```export ROS_DOMAIN_ID=5```
```export OBELISK_BUILD_UNITREE=true```

## Useful commands
Setup:
```
bash setup.sh
```

Enter the docker container: 
```
docker compose -f docker/docker-compose.yml run --build sample-walking
```

Build and activate Obelisk:
```
obk
```

Build all the packages:
```
colcon build --symlink-install --parallel-workers $(nproc)
```

Build packages with verbose output:
```
colcon build --symlink-install --parallel-workers $(nproc) --event-handlers console_direct+
```

Build packages in debug mode:
```
colcon build --symlink-install --parallel-workers $(nproc) --cmake-args -DCMAKE_BUILD_TYPE=Debug
```

Source the package:
```
source install/setup.bash
```

Set logging dir:
```
export ROS_HOME=~/sample-contact-walking
```
## Launch the Go2-Nav stack:
### Launch the RL velocity controller, with joystick commands

```obk-launch config_file_path=${GO2_NAVIGATION_ROOT}/install/go2_rl_vel_tracking/share/go2_rl_vel_tracking/config/rl_vel_tracking.yml device_name=onboard```

### Launch a high-level test file (sim only) where MPC trajectory is followed exactly.

```obk-launch config_file_path=${GO2_NAVIGATION_ROOT}/install/go2_dyn_tube_mpc/share/go2_dyn_tube_mpc/config/high_level_test.yml device_name=onboard```

### Launch Dynamic Tube MPC

```obk-launch config_file_path=${GO2_NAVIGATION_ROOT}/install/go2_dyn_tube_mpc/share/go2_dyn_tube_mpc/config/dynamic_tube_mpc.yml device_name=onboard```

To spoof data provided by SLAM, first run (in a different terminal, `use_robot_sim=True` to follow MuJoCo simulation, `False` to follow MPC exactly.)

```ros2 run testing mimic_slam_robot --ros-args -p use_robot_sim:=True```

The relevant commands are aliased by 

```launch_go2_rl_vel```

```launch_go2_high_level_test```

```launch_go2_dtmpc```


## Connecting the joystick
### USB
Can verify that the the controller connects via
```
sudo apt-get update
sudo apt-get install evtest
sudo evtest
```
where you replace eventX with the correct number. You can see these by looking at `/dev/input/`.

Then you may need to change the permissions for the joystick:
```
sudo chmod 666 /dev/input/eventX
```
event21 seems to be consistent for the xbox remote for my machine.

Can run `ros2 run joy joy_enumerate_devices` to see what devices are found.
