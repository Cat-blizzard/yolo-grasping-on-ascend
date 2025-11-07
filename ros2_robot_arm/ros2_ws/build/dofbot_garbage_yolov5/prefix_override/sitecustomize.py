import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/root/ros2_robot_arm/ros2_ws/install/dofbot_garbage_yolov5'
