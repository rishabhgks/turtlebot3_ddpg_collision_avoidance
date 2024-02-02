# Project Title: Mapless Collision Avoidance of Turtlebot3 Mobile Robot Using DDPG and Prioritized Experience Replay with Multiple Robots and Multiple Targets
A preliminary version of this work is implemented in paper [Accelerated Sim-to-Real Deep Reinforcement Learning: Learning Collision Avoidance from Human Player](https://arxiv.org/abs/2102.10711) published in 2021 IEEE/SICE International Symposium on System Integration (SII) and [Voronoi-Based Multi-Robot Autonomous Exploration in Unknown Environments via Deep Reinforcement Learning](https://ieeexplore.ieee.org/abstract/document/9244647) published in IEEE Transactions on Vehicular Technology.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Prerequisites

What things you need to install the software

```
Ubuntu 16.04
ROS Kinetic
Tensorflow-gpu == 1.13.1 or 1.14.0
Keras == 2.3.1
```


### Virtual Environment

You need to make a virtual environment called 'ddpg_env' and install the following library for it.

```
Tensorflow-gpu == 1.13.1 or 1.14.0
Keras == 2.3.1
```

### Installing

The next step is to install dependent packages for TurtleBot3 control on Remote PC. For more details, please refer to [turtlebot3](http://emanual.robotis.com/docs/en/platform/turtlebot3/setup/#setup).

```
$ sudo apt-get update
$ sudo apt-get upgrade
$ wget https://raw.githubusercontent.com/ROBOTIS-GIT/robotis_tools/master/install_ros_kinetic.sh && chmod 755 ./install_ros_kinetic.sh && bash ./install_ros_kinetic.sh

$ sudo apt-get install ros-kinetic-joy ros-kinetic-teleop-twist-joy ros-kinetic-teleop-twist-keyboard ros-kinetic-laser-proc ros-kinetic-rgbd-launch ros-kinetic-depthimage-to-laserscan ros-kinetic-rosserial-arduino ros-kinetic-rosserial-python ros-kinetic-rosserial-server ros-kinetic-rosserial-client ros-kinetic-rosserial-msgs ros-kinetic-amcl ros-kinetic-map-server ros-kinetic-move-base ros-kinetic-urdf ros-kinetic-xacro ros-kinetic-compressed-image-transport ros-kinetic-rqt-image-view ros-kinetic-gmapping ros-kinetic-navigation ros-kinetic-interactive-markers

$ cd ~/catkin_ws/src/
$ git clone https://github.com/ROBOTIS-GIT/turtlebot3_msgs.git
$ git clone https://github.com/ROBOTIS-GIT/turtlebot3.git
$ cd ~/catkin_ws && catkin_make
```


## Setting up the network between PC and turtlebot

Please refer to [Turtlebot3 Setup](http://emanual.robotis.com/docs/en/platform/turtlebot3/pc_setup/#install-ubuntu-on-remote-pc)

## Git clone ddpg scripts
For multi-robot and multi-target integration:
```
checkout branch: <multi-robot-v3>
```
```
$ cd ~/catkin_ws/src/
$ git clone https://github.com/subhramoy/turtlebot3_ddpg_collision_avoidance.git
$ cd ~/catkin_ws && catkin_make
```

### Start gazebo world (Change your world file location based on your setting)

To launch the maze world
```
$ roslaunch turtlebot_ddpg turtlebot3_empty_world.launch world_file:='/home/.../catkin_ws/src/turtlebot3_ddpg_collision_avoidance/turtlebot_ddpg/worlds/turtlebot3_modified_maze.world'
```
To launch the corridor world
```
$ roslaunch turtlebot_ddpg turtlebot3_empty_world.launch world_file:='/home/.../catkin_ws/src/turtlebot3_ddpg_collision_avoidance/turtlebot_ddpg/worlds/turtlebot3_modified_corridor2.world'
```

### Start a new terminals for executing RL for robot navigation training and testing
```
$ source ~/ddpg_env/bin/activate
```
Terminal 1
```
$ cd ~/catkin_ws 
$ source devel/setup.bash
$ export TURTLEBOT3_MODEL=waffle_pi
$ catkin_make 
$ roscore 
```

Terminal 2
```
$ export TURTLEBOT3_MODEL=waffle_pi
$ rosrun turtlebot_ddpg ddpg_network_turtlebot3_amcl_fd_replay_human_dynamic.py
```

For train and play with ddpg with human data
```
$ cd ~/catkin_ws/src/.../turtlebot_ddpg/scripts/fd_replay/play_human_data
$ rosrun turtlebot_ddpg ddpg_network_turtlebot3_amcl_fd_replay_human.py
```
For train and play with original ddpg

```
$ cd ~/catkin_ws/.../turtlebot_ddpg/scripts/original_ddpg
$ rosrun turtlebot_ddpg ddpg_network_turtlebot3_original_ddpg.py
```

For training
```
please change train_indicator=1 under ddpg_network_turtlebot3_original_ddpg.py (DDPG) or ddpg_network_turtlebot3_amcl_fd_replay_human.py (DDPG-PER)
```

For playing trained weights
```
 please change train_indicator=0 under ddpg_network_turtlebot3_original_ddpg.py (DDPG) or ddpg_network_turtlebot3_amcl_fd_replay_human.py (DDPG-PER)
```




# Paper
If you use this code in your research, please cite our paper:


```
@ARTICLE{10100908,
  author={Mohanti, Subhramoy and Roy, Debashri and Eisen, Mark and Cavalcanti, Dave and Chowdhury, Kaushik},
  journal={IEEE Transactions on Mobile Computing}, 
  title={L-NORM: Learning and Network Orchestration at the Edge for Robot Connectivity and Mobility in Factory Floor Environments}, 
  year={2023},
  volume={},
  number={},
  pages={1-16},
  keywords={Robots;Robot sensing systems;Robot kinematics;Collision avoidance;Navigation;IEEE 802.11ax Standard;Uplink;Edge network;orchestration;robot navigation;reinforcement learning;multi-modal data},
  doi={10.1109/TMC.2023.3266643}}
```

and also refer to the papers from:
https://github.com/hanlinniu/turtlebot3_ddpg_collision_avoidance
