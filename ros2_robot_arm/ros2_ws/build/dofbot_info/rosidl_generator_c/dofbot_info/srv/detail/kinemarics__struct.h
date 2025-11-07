// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from dofbot_info:srv/Kinemarics.idl
// generated code does not contain a copyright notice

#ifndef DOFBOT_INFO__SRV__DETAIL__KINEMARICS__STRUCT_H_
#define DOFBOT_INFO__SRV__DETAIL__KINEMARICS__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'kin_name'
#include "rosidl_runtime_c/string.h"

/// Struct defined in srv/Kinemarics in the package dofbot_info.
typedef struct dofbot_info__srv__Kinemarics_Request
{
  double tar_x;
  double tar_y;
  double tar_z;
  double roll;
  double pitch;
  double yaw;
  double cur_joint1;
  double cur_joint2;
  double cur_joint3;
  double cur_joint4;
  double cur_joint5;
  double cur_joint6;
  rosidl_runtime_c__String kin_name;
} dofbot_info__srv__Kinemarics_Request;

// Struct for a sequence of dofbot_info__srv__Kinemarics_Request.
typedef struct dofbot_info__srv__Kinemarics_Request__Sequence
{
  dofbot_info__srv__Kinemarics_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} dofbot_info__srv__Kinemarics_Request__Sequence;


// Constants defined in the message

/// Struct defined in srv/Kinemarics in the package dofbot_info.
typedef struct dofbot_info__srv__Kinemarics_Response
{
  double joint1;
  double joint2;
  double joint3;
  double joint4;
  double joint5;
  double joint6;
  double x;
  double y;
  double z;
  double roll;
  double pitch;
  double yaw;
} dofbot_info__srv__Kinemarics_Response;

// Struct for a sequence of dofbot_info__srv__Kinemarics_Response.
typedef struct dofbot_info__srv__Kinemarics_Response__Sequence
{
  dofbot_info__srv__Kinemarics_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} dofbot_info__srv__Kinemarics_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // DOFBOT_INFO__SRV__DETAIL__KINEMARICS__STRUCT_H_
