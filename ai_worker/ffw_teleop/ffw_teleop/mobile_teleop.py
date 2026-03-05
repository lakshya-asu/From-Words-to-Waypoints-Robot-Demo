#!/usr/bin/env python3
# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import termios
import tty

from geometry_msgs.msg import Twist
import rclpy
from rclpy.node import Node


def get_key():
    """Read a single keypress from terminal."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        key = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return key


class KeyboardTeleop(Node):

    def __init__(self):
        super().__init__('keyboard_teleop')
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 5)

        self.linear_speed = 0.4    # m/s
        self.angular_speed = 0.8   # rad/s

        self.get_logger().info(
            '\nKeyboard Teleop\n'
            '---------------------------\n'
            'W/S : Forward / Backward\n'
            'A/D : Turn Left / Right\n'
            'Space: Stop\n'
            'Ctrl+C: Quit\n'
        )

    def run(self):
        twist = Twist()

        while rclpy.ok():
            key = get_key()

            twist.linear.x = 0.0
            twist.angular.z = 0.0

            if key == 'w':
                twist.linear.x = self.linear_speed
            elif key == 's':
                twist.linear.x = -self.linear_speed
            elif key == 'a':
                twist.angular.z = self.angular_speed
            elif key == 'd':
                twist.angular.z = -self.angular_speed
            elif key == ' ':
                pass  # stop
            elif key == '\x03':  # Ctrl+C
                break

            self.publisher.publish(twist)

        # stop robot on exit
        self.publisher.publish(Twist())
        self.get_logger().info('Teleop stopped.')


def main():
    rclpy.init()
    node = KeyboardTeleop()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
