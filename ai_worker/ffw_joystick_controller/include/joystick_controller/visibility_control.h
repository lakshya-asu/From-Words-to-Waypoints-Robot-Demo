// Copyright 2024 ROBOTIS CO., LTD.
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

#ifndef JOYSTICK_CONTROLLER__VISIBILITY_CONTROL_H_
#define JOYSTICK_CONTROLLER__VISIBILITY_CONTROL_H_

// This logic was borrowed (then namespaced) from the examples on the gcc wiki:
//     https://gcc.gnu.org/wiki/Visibility

#if defined _WIN32 || defined __CYGWIN__
  #ifdef __GNUC__
    #define JOYSTICK_CONTROLLER_EXPORT __attribute__ ((dllexport))
    #define JOYSTICK_CONTROLLER_IMPORT __attribute__ ((dllimport))
  #else
    #define JOYSTICK_CONTROLLER_EXPORT __declspec(dllexport)
    #define JOYSTICK_CONTROLLER_IMPORT __declspec(dllimport)
  #endif
  #ifdef JOYSTICK_CONTROLLER_BUILDING_LIBRARY
    #define JOYSTICK_CONTROLLER_PUBLIC JOYSTICK_CONTROLLER_EXPORT
  #else
    #define JOYSTICK_CONTROLLER_PUBLIC JOYSTICK_CONTROLLER_IMPORT
  #endif
  #define JOYSTICK_CONTROLLER_PUBLIC_TYPE JOYSTICK_CONTROLLER_PUBLIC
  #define JOYSTICK_CONTROLLER_LOCAL
#else
  #define JOYSTICK_CONTROLLER_EXPORT __attribute__ ((visibility("default")))
  #define JOYSTICK_CONTROLLER_IMPORT
  #if __GNUC__ >= 4
    #define JOYSTICK_CONTROLLER_PUBLIC __attribute__ ((visibility("default")))
    #define JOYSTICK_CONTROLLER_LOCAL  __attribute__ ((visibility("hidden")))
  #else
    #define JOYSTICK_CONTROLLER_PUBLIC
    #define JOYSTICK_CONTROLLER_LOCAL
  #endif
  #define JOYSTICK_CONTROLLER_PUBLIC_TYPE
#endif

#endif  // JOYSTICK_CONTROLLER__VISIBILITY_CONTROL_H_
