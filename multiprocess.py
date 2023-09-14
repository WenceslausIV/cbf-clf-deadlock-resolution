import os
from multiprocessing import Pool


def task(param):
    # os.system(f"echo {param}")
    os.system(
        f"ssh {param} 'export ROSBOT_VER=ROSBOT_2.0_PRO; export ROS_DISTRO=noetic; export ROS_MASTER_URI=http://master:11311; export ROS_IPV6=on; . /opt/ros/noetic/setup.bash; . ~/husarion_ws/devel/setup.sh; export SERIAL_PORT=/dev/ttyS4; source rosbot_ros/devel/setup.bash; python3 /opt/ros/noetic/lib/teleop_twist_keyboard/teleop_twist_keyboardres.py'")


with Pool(8) as p:
    p.map(task,
          ["husarion@192.168.1.117", "husarion@192.168.1.137", "husarion@192.168.1.138", "husarion@192.168.1.188"])
