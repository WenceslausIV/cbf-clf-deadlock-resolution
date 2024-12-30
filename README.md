# Husarion-ROSbot2Pro-cbf-clf-deadlock-resolution

1. Replace the teleop_twist_keyboard.py at /opt/ros/noetic/lib/teleop_twist_keyboard/ for deadlock detection using cbf-clf for each robot


2. Put teleop_twist_keyboardres.py at /opt/ros/noetic/lib/teleop_twist_keyboard/ for deadlock resolution using cbf-clf for each robot

3. Run multiprocess.py on your PC after ssh into each robot having rosbots docker & vrpn system on
