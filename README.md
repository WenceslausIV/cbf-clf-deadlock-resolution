Yanze Zhang, Yiwei Lyu, Siwon Jo, Yupeng Yang, and Wenhao Luo, “Adaptive Deadlock Avoidance
for Decentralized Multi-Agent Systems via CBF-inspired Risk Measurement,” in IEEE International
Conference on Robotics and Automation (ICRA 2025)

#cbf-clf-deadlock-resolution

1. Replace the teleop_twist_keyboard.py at /opt/ros/noetic/lib/teleop_twist_keyboard/ for deadlock detection using cbf-clf for each robot


2. Put teleop_twist_keyboardres.py at /opt/ros/noetic/lib/teleop_twist_keyboard/ for deadlock resolution using cbf-clf for each robot

3. Run multiprocess.py on your PC after ssh into each robot having rosbots docker & vrpn system on

![-3688943094785373252my_movie-ezgif com-video-to-gif-converter (1)](https://github.com/user-attachments/assets/2fc4040d-2aff-4781-bd7a-26a557f44866)
