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

#include "ffw_robot_manager/ubetter_battery_model.hpp"
#include "sensor_msgs/msg/battery_state.hpp"

namespace ffw_robot_manager
{

// Battery voltage to SOC mapping for ubetter battery
// Voltage values are in 0.01V units (2940 = 29.40V)
// Percentage values are in 0.1% units (1000 = 100.0%)
const uint16_t UbetterBatteryModel::battery_percent_data[BATTERY_DATA_NUMBER][2] =
{  /* {voltage_0.01V , %_0.1% }*/
  {2940, 1000}, {2841, 908}, {2833, 877},
  {2823, 846}, {2805, 815}, {2784, 784}, {2761, 753},
  {2739, 722}, {2719, 691}, {2700, 660}, {2682, 629},
  {2664, 598}, {2645, 567}, {2626, 536}, {2610, 505},
  {2583, 474}, {2565, 443}, {2549, 412}, {2534, 381},
  {2521, 350}, {2507, 319}, {2493, 288}, {2477, 257},
  {2458, 226}, {2437, 195}, {2420, 164}, {2404, 133},
  {2372, 102}, {2329, 71}, {2272, 40}, {2200, 0},
};

UbetterBatteryModel::UbetterBatteryModel()
{
}

double UbetterBatteryModel::voltage_to_soc(double voltage_v) const
{
  // Convert Volts to 0.01V units for lookup table (e.g., 29.40V -> 2940)
  const double voltage_units_d = voltage_v * 100.0;

  // Negative voltages are invalid, return 0%
  if (voltage_units_d < 0.0) {
    return 0.0;
  }

  // Handle edge cases first using double to avoid premature truncation/casting
  if (voltage_units_d >= battery_percent_data[0][0]) {
    return 1.0;  // Above highest voltage (29.40V = 100%)
  } else if (voltage_units_d <= battery_percent_data[BATTERY_DATA_NUMBER - 1][0]) {
    return 0.0;  // Below lowest voltage (22.00V = 0%)
  }

  // Safe to cast after clamping checks
  const uint16_t voltage_units = static_cast<uint16_t>(voltage_units_d);

  // Find the correct range for interpolation
  // Data is in descending order: 2940 -> 2200 (29.40V -> 22.00V)
  for (size_t i = 0; i < BATTERY_DATA_NUMBER - 1; ++i) {
    if (voltage_units <= battery_percent_data[i][0] &&
      voltage_units >= battery_percent_data[i + 1][0])
    {
      // Linear interpolation between two points
      double voltage_diff = battery_percent_data[i][0] - battery_percent_data[i + 1][0];
      double soc_diff = battery_percent_data[i][1] - battery_percent_data[i + 1][1];
      double ratio = (voltage_units - battery_percent_data[i + 1][0]) / voltage_diff;

      // Convert from 0.1% units to fraction [0.0, 1.0]
      return (battery_percent_data[i + 1][1] + ratio * soc_diff) / 1000.0;
    }
  }

  return 0.0;  // Default to 0% if no match
}

std::string UbetterBatteryModel::get_model_name() const
{
  return "ubetter";
}

std::pair<double, double> UbetterBatteryModel::get_voltage_range() const
{
  // {min, max} in Volts (22.00V to 29.40V)
  return {battery_percent_data[BATTERY_DATA_NUMBER - 1][0] / 100.0,
    battery_percent_data[0][0] / 100.0};
}

bool UbetterBatteryModel::is_voltage_valid(double voltage_v) const
{
  auto range = get_voltage_range();
  return voltage_v >= range.first && voltage_v <= range.second;
}

uint8_t UbetterBatteryModel::get_power_supply_technology() const
{
  return sensor_msgs::msg::BatteryState::POWER_SUPPLY_TECHNOLOGY_LIPO;
}

}  // namespace ffw_robot_manager
