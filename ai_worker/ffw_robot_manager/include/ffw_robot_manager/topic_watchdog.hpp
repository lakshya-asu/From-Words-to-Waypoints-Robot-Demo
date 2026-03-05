// Copyright 2025 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Author: Woojin Wie

#pragma once

#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>

namespace ffw_robot_manager
{

template<typename MsgT>
class TopicWatchdog {
public:
  using TimeoutCallback = std::function<void()>;

  TopicWatchdog(
    rclcpp_lifecycle::LifecycleNode * node,
    const std::string & topic_name,
    std::chrono::milliseconds timeout,
    TimeoutCallback on_timeout_cb)
  : node_(node),
    topic_name_(topic_name),
    timeout_(timeout),
    on_timeout_cb_(on_timeout_cb)
  {
    sub_ = node_->create_subscription<MsgT>(
      topic_name_, 10,
      [this](const typename MsgT::SharedPtr /*msg*/) {
        last_msg_time_ = node_->now();
        timed_out_ = false;
      });
    timer_ = node_->create_wall_timer(
      std::chrono::milliseconds(50),
      std::bind(&TopicWatchdog::check_timeout, this));
    last_msg_time_ = node_->now();
  }

  bool is_timed_out() const {return timed_out_;}

private:
  void check_timeout()
  {
    auto now = node_->now();
    if ((now - last_msg_time_).nanoseconds() > timeout_.count() * 1'000'000) {
      if (!timed_out_) {
        timed_out_ = true;
        if (on_timeout_cb_) {on_timeout_cb_();}
      }
    }
  }

  rclcpp_lifecycle::LifecycleNode * node_;
  std::string topic_name_;
  std::chrono::milliseconds timeout_;
  TimeoutCallback on_timeout_cb_;
  typename rclcpp::Subscription<MsgT>::SharedPtr sub_;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Time last_msg_time_;
  bool timed_out_ = false;
};

}  // namespace ffw_robot_manager
