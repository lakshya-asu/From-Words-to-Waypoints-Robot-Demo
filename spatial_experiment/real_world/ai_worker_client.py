import time
import math
import threading
from typing import Optional, Tuple, Dict, Any
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor

from std_msgs.msg import Header
from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped
from sensor_msgs.msg import Image, CameraInfo, JointState
from nav_msgs.msg import Odometry

# Nav2
from nav2_msgs.action import NavigateToPose

# CV / TF
from cv_bridge import CvBridge
import tf2_ros
from tf2_ros import TransformException
import tf_transformations

from stretch.core.robot import AbstractRobotClient
from stretch.core.interfaces import Observations

class AIWorkerRobotClient(AbstractRobotClient, Node):
    """
    ROS 2 Robot Client for the ROBOTIS AI Worker.
    Fulfills the AbstractRobotClient interface required by spatial_experiment.
    Runs an internal rclpy spin loop in a background thread.
    """
    def __init__(self, node_name="ai_worker_client_node"):
        # Initialize rclpy if not already initialized
        if not rclpy.ok():
            rclpy.init()
            
        Node.__init__(self, node_name)
        
        self.running = False
        self._last_fail = False
        self.bridge = CvBridge()

        # Observation state
        self._rgb_img = None
        self._depth_img = None
        self._camera_info = None
        self._camera_K = np.eye(3)
        self._last_lidar_timestamp = None
        
        self._base_pose = np.array([0.0, 0.0, 0.0]) # x, y, theta
        self._camera_pose = np.eye(4) # 4x4 homogenous matrix in map frame
        
        self._head_pan = 0.0
        self._head_tilt = 0.0
        
        self._obs_lock = threading.Lock()

        # TF Listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Subscribers
        self.rgb_sub = self.create_subscription(
            Image,
            '/zed/zed_node/rgb/image_rect_color',
            self._rgb_callback,
            10
        )
        self.depth_sub = self.create_subscription(
            Image,
            '/zed/zed_node/depth/depth_registered',
            self._depth_callback,
            10
        )
        self.cam_info_sub = self.create_subscription(
            CameraInfo,
            '/zed/zed_node/rgb/camera_info',
            self._cam_info_callback,
            10
        )
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self._joint_state_callback,
            10
        )

        # Action Clients
        self.nav_to_pose_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        self.get_logger().info("AIWorkerRobotClient initialized.")

    # --- Lifecycle ---
    def start(self):
        """Start the background spinner thread."""
        if self.running:
            return True
        
        self.running = True
        self.executor = MultiThreadedExecutor()
        self.executor.add_node(self)
        self.spin_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.spin_thread.start()
        
        self.get_logger().info("AIWorkerRobotClient spinner started. Waiting for sensors...")
        
        # Wait for initial data
        timeout = 10.0
        start_t = time.time()
        while time.time() - start_t < timeout:
            with self._obs_lock:
                if self._rgb_img is not None and self._depth_img is not None and self._camera_info is not None:
                    self.get_logger().info("Sensor streams established.")
                    # Force a tf update to populate poses
                    self._update_poses()
                    return True
            time.sleep(0.1)
            
        self.get_logger().warn("Timeout waiting for ZED camera data. Ensure the bringup is running.")
        return False

    def stop(self):
        """Stop the background spinner."""
        self.running = False
        if rclpy.ok():
            self.executor.shutdown()
            rclpy.shutdown()
        if hasattr(self, 'spin_thread'):
            self.spin_thread.join(timeout=2.0)
        self.get_logger().info("AIWorkerRobotClient stopped.")

    # --- Callbacks ---
    def _rgb_callback(self, msg: Image):
        with self._obs_lock:
            # ZED image is likely BGRA or RGBA, convert to RGB
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            self._rgb_img = cv_img
            self._last_lidar_timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def _depth_callback(self, msg: Image):
        with self._obs_lock:
            # Convert to 32FC1 meters
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
            # Replace NaNs or infinite values with 0
            cv_img = np.nan_to_num(cv_img, nan=0.0, posinf=0.0, neginf=0.0)
            self._depth_img = cv_img

    def _cam_info_callback(self, msg: CameraInfo):
        with self._obs_lock:
            self._camera_info = msg
            self._camera_K = np.array(msg.k).reshape(3, 3)

    def _joint_state_callback(self, msg: JointState):
        with self._obs_lock:
            # Assuming typical pan/tilt names. Update these if ai_worker differs.
            if 'head_pan_joint' in msg.name:
                idx = msg.name.index('head_pan_joint')
                self._head_pan = msg.position[idx]
            if 'head_tilt_joint' in msg.name:
                idx = msg.name.index('head_tilt_joint')
                self._head_tilt = msg.position[idx]

    def _update_poses(self):
        """Update base_link and camera spatial transforms."""
        # 1. Base Pose (map -> base_link)
        try:
            t_base = self.tf_buffer.lookup_transform(
                'map',
                'base_link',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            trans = t_base.transform.translation
            rot = t_base.transform.rotation
            _, _, yaw = tf_transformations.euler_from_quaternion([rot.x, rot.y, rot.z, rot.w])
            self._base_pose = np.array([trans.x, trans.y, yaw])
        except TransformException as ex:
            pass # Keep old pose if lookup fails occasionally

        # 2. Camera Pose (map -> zed_camera_center or equivalent)
        # Match this frame to where the camera optical frame is in ai_worker URDF
        # If 'zed_left_camera_optical_frame' is standard for stereolabs config:
        target_camera_frame = 'zed_left_camera_optical_frame'
        try:
            t_cam = self.tf_buffer.lookup_transform(
                'map',
                target_camera_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            
            # Construct 4x4 homogeneous matrix
            mat = tf_transformations.quaternion_matrix([
                t_cam.transform.rotation.x,
                t_cam.transform.rotation.y,
                t_cam.transform.rotation.z,
                t_cam.transform.rotation.w
            ])
            mat[0, 3] = t_cam.transform.translation.x
            mat[1, 3] = t_cam.transform.translation.y
            mat[2, 3] = t_cam.transform.translation.z
            
            self._camera_pose = mat
            
        except TransformException as ex:
            pass 

    # --- AbstractRobotClient Interface ---
    
    def get_observation(self) -> Observations:
        """Returns the unified observation struct expected by spatial_experiment."""
        self._update_poses()
        
        with self._obs_lock:
            if self._rgb_img is None or self._depth_img is None:
                # Return empty/dummy if not ready yet, though start() should prevent this
                return Observations(
                    rgb=np.zeros((360, 640, 3), dtype=np.uint8),
                    depth=np.zeros((360, 640), dtype=np.float32),
                    camera_K=np.eye(3),
                    camera_pose=np.eye(4),
                    lidar_timestamp=time.time()
                )
                
            obs = Observations(
                rgb=self._rgb_img.copy(),
                depth=self._depth_img.copy(),
                camera_K=self._camera_K.copy(),
                camera_pose=self._camera_pose.copy(),
                gps=self._base_pose[:2].copy(),
                compass=np.array([self._base_pose[2]]),
                lidar_timestamp=self._last_lidar_timestamp or time.time(),
                joint=np.array([self._head_pan, self._head_tilt]) # Expose head joints if needed
            )
            return obs

    def get_base_pose(self) -> np.ndarray:
        """Returns [x, y, theta] in the map frame."""
        self._update_poses()
        return self._base_pose

    def move_base_to(self, target: np.ndarray, relative=False, blocking=True, timeout=10.0, verbose=False):
        """
        Move the robot base.
        If `relative` is True, `target` is [dx, dy, dtheta] relative to current pose.
        If `relative` is False, `target` is [x, y, theta] in map frame.
        """
        self._last_fail = False
        
        # If the target is basically just a rotation in place, we can optionally use cmd_vel 
        # for faster execution instead of spinning up Nav2.
        dx = 0.0
        dy = 0.0
        dtheta = 0.0
        
        if relative:
            dx, dy, dtheta = target
        else:
            curr_pose = self.get_base_pose()
            dx = target[0] - curr_pose[0]
            dy = target[1] - curr_pose[1]
            dtheta = target[2] - curr_pose[2]
            
        # Normalize angular diff
        dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
            
        dist = np.hypot(dx, dy)
        
        if dist < 0.05 and abs(dtheta) > 0.01:
            # Basically a pure rotation -> Use cmd_vel
            if verbose:
                self.get_logger().info(f"Using cmd_vel for in-place rotation of {dtheta} rad")
            return self._rotate_in_place_cmd_vel(dtheta, timeout=timeout)
        else:
            # Use Nav2 for global navigation
            if relative:
                curr_pose = self.get_base_pose()
                target_x = curr_pose[0] + dx
                target_y = curr_pose[1] + dy
                target_theta = curr_pose[2] + dtheta
            else:
                target_x, target_y, target_theta = target
                
            if verbose:
                self.get_logger().info(f"Using Nav2 to reach map pose: x={target_x:.2f}, y={target_y:.2f}, th={target_theta:.2f}")
            return self._navigate_to_pose_action(target_x, target_y, target_theta, blocking=blocking, timeout=timeout)

    def _rotate_in_place_cmd_vel(self, dtheta_rad, max_speed=0.5, timeout=10.0):
        """Blocks by default until rotation is roughly complete."""
        start_pose = self.get_base_pose()
        target_yaw = (start_pose[2] + dtheta_rad + np.pi) % (2 * np.pi) - np.pi
        
        msg = Twist()
        direction = 1.0 if dtheta_rad > 0 else -1.0
        msg.angular.z = max_speed * direction
        
        start_t = time.time()
        r = self.create_rate(10)
        
        while time.time() - start_t < timeout and self.running:
            curr_yaw = self.get_base_pose()[2]
            diff = (target_yaw - curr_yaw + np.pi) % (2 * np.pi) - np.pi
            
            if abs(diff) < 0.05: # within ~3 degrees
                break
                
            # Proportional slowdown near target
            msg.angular.z = np.clip(diff * 1.5, -max_speed, max_speed)
            
            # Ensure minimum velocity to break static friction
            if abs(msg.angular.z) < 0.1:
                msg.angular.z = 0.1 * np.sign(msg.angular.z)
                
            self.cmd_vel_pub.publish(msg)
            time.sleep(0.1)
            
        # Stop
        msg.angular.z = 0.0
        self.cmd_vel_pub.publish(msg)
        
        if time.time() - start_t >= timeout:
            self._last_fail = True
            return False
            
        return True

    def _navigate_to_pose_action(self, x, y, theta, blocking=True, timeout=60.0):
        if not self.nav_to_pose_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error("NavigateToPose action server not available!")
            self._last_fail = True
            return False
            
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = float(x)
        goal_msg.pose.pose.position.y = float(y)
        goal_msg.pose.pose.position.z = 0.0
        
        qx, qy, qz, qw = tf_transformations.quaternion_from_euler(0, 0, theta)
        goal_msg.pose.pose.orientation.x = qx
        goal_msg.pose.pose.orientation.y = qy
        goal_msg.pose.pose.orientation.z = qz
        goal_msg.pose.pose.orientation.w = qw
        
        send_goal_future = self.nav_to_pose_client.send_goal_async(goal_msg)
        
        if not blocking:
            return True
            
        # Block until goal is sent and accepted
        while not send_goal_future.done():
            time.sleep(0.1)
            
        goal_handle = send_goal_future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Nav2 goal rejected")
            self._last_fail = True
            return False
            
        # Block until result
        result_future = goal_handle.get_result_async()
        start_t = time.time()
        
        while not result_future.done() and self.running:
            if time.time() - start_t > timeout:
                self.get_logger().warn("Nav2 Goal timed out. Canceling.")
                goal_handle.cancel_goal_async()
                self._last_fail = True
                return False
            time.sleep(0.1)
            
        if not result_future.done():
            return False
            
        self._last_fail = False
        return True

    def last_motion_failed(self):
        return self._last_fail

    # Empty stubs to fulfill AbstractRobotClient requirements for unsupported/stretch-specific logic
    def in_manipulation_mode(self): return False
    def switch_to_manipulation_mode(self): pass
    def switch_to_navigation_mode(self): pass
    def move_to_nav_posture(self): pass
    
    def get_robot_model(self):
        class AIWorkerModel:
            def get_footprint(self): 
                # Roughly based on AI worker base dimensions: 0.530 x 0.443 meters
                return [[-0.265, -0.22], [0.265, -0.22], [0.265, 0.22], [-0.265, 0.22]]
        return AIWorkerModel()
        
    def get_pose_graph(self): 
        # Return empty list, spatial_experiment handles map building directly via SparseVoxelMap
        return []
        
    def execute_trajectory(self, pts, **kwargs): return True
    def wait_for_waypoint(self, pt, **kwargs): return True
