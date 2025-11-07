// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from dofbot_info:srv/Kinemarics.idl
// generated code does not contain a copyright notice

#ifndef DOFBOT_INFO__SRV__DETAIL__KINEMARICS__TRAITS_HPP_
#define DOFBOT_INFO__SRV__DETAIL__KINEMARICS__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "dofbot_info/srv/detail/kinemarics__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace dofbot_info
{

namespace srv
{

inline void to_flow_style_yaml(
  const Kinemarics_Request & msg,
  std::ostream & out)
{
  out << "{";
  // member: tar_x
  {
    out << "tar_x: ";
    rosidl_generator_traits::value_to_yaml(msg.tar_x, out);
    out << ", ";
  }

  // member: tar_y
  {
    out << "tar_y: ";
    rosidl_generator_traits::value_to_yaml(msg.tar_y, out);
    out << ", ";
  }

  // member: tar_z
  {
    out << "tar_z: ";
    rosidl_generator_traits::value_to_yaml(msg.tar_z, out);
    out << ", ";
  }

  // member: roll
  {
    out << "roll: ";
    rosidl_generator_traits::value_to_yaml(msg.roll, out);
    out << ", ";
  }

  // member: pitch
  {
    out << "pitch: ";
    rosidl_generator_traits::value_to_yaml(msg.pitch, out);
    out << ", ";
  }

  // member: yaw
  {
    out << "yaw: ";
    rosidl_generator_traits::value_to_yaml(msg.yaw, out);
    out << ", ";
  }

  // member: cur_joint1
  {
    out << "cur_joint1: ";
    rosidl_generator_traits::value_to_yaml(msg.cur_joint1, out);
    out << ", ";
  }

  // member: cur_joint2
  {
    out << "cur_joint2: ";
    rosidl_generator_traits::value_to_yaml(msg.cur_joint2, out);
    out << ", ";
  }

  // member: cur_joint3
  {
    out << "cur_joint3: ";
    rosidl_generator_traits::value_to_yaml(msg.cur_joint3, out);
    out << ", ";
  }

  // member: cur_joint4
  {
    out << "cur_joint4: ";
    rosidl_generator_traits::value_to_yaml(msg.cur_joint4, out);
    out << ", ";
  }

  // member: cur_joint5
  {
    out << "cur_joint5: ";
    rosidl_generator_traits::value_to_yaml(msg.cur_joint5, out);
    out << ", ";
  }

  // member: cur_joint6
  {
    out << "cur_joint6: ";
    rosidl_generator_traits::value_to_yaml(msg.cur_joint6, out);
    out << ", ";
  }

  // member: kin_name
  {
    out << "kin_name: ";
    rosidl_generator_traits::value_to_yaml(msg.kin_name, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const Kinemarics_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: tar_x
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "tar_x: ";
    rosidl_generator_traits::value_to_yaml(msg.tar_x, out);
    out << "\n";
  }

  // member: tar_y
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "tar_y: ";
    rosidl_generator_traits::value_to_yaml(msg.tar_y, out);
    out << "\n";
  }

  // member: tar_z
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "tar_z: ";
    rosidl_generator_traits::value_to_yaml(msg.tar_z, out);
    out << "\n";
  }

  // member: roll
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "roll: ";
    rosidl_generator_traits::value_to_yaml(msg.roll, out);
    out << "\n";
  }

  // member: pitch
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "pitch: ";
    rosidl_generator_traits::value_to_yaml(msg.pitch, out);
    out << "\n";
  }

  // member: yaw
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "yaw: ";
    rosidl_generator_traits::value_to_yaml(msg.yaw, out);
    out << "\n";
  }

  // member: cur_joint1
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "cur_joint1: ";
    rosidl_generator_traits::value_to_yaml(msg.cur_joint1, out);
    out << "\n";
  }

  // member: cur_joint2
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "cur_joint2: ";
    rosidl_generator_traits::value_to_yaml(msg.cur_joint2, out);
    out << "\n";
  }

  // member: cur_joint3
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "cur_joint3: ";
    rosidl_generator_traits::value_to_yaml(msg.cur_joint3, out);
    out << "\n";
  }

  // member: cur_joint4
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "cur_joint4: ";
    rosidl_generator_traits::value_to_yaml(msg.cur_joint4, out);
    out << "\n";
  }

  // member: cur_joint5
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "cur_joint5: ";
    rosidl_generator_traits::value_to_yaml(msg.cur_joint5, out);
    out << "\n";
  }

  // member: cur_joint6
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "cur_joint6: ";
    rosidl_generator_traits::value_to_yaml(msg.cur_joint6, out);
    out << "\n";
  }

  // member: kin_name
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "kin_name: ";
    rosidl_generator_traits::value_to_yaml(msg.kin_name, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const Kinemarics_Request & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace srv

}  // namespace dofbot_info

namespace rosidl_generator_traits
{

[[deprecated("use dofbot_info::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const dofbot_info::srv::Kinemarics_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  dofbot_info::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use dofbot_info::srv::to_yaml() instead")]]
inline std::string to_yaml(const dofbot_info::srv::Kinemarics_Request & msg)
{
  return dofbot_info::srv::to_yaml(msg);
}

template<>
inline const char * data_type<dofbot_info::srv::Kinemarics_Request>()
{
  return "dofbot_info::srv::Kinemarics_Request";
}

template<>
inline const char * name<dofbot_info::srv::Kinemarics_Request>()
{
  return "dofbot_info/srv/Kinemarics_Request";
}

template<>
struct has_fixed_size<dofbot_info::srv::Kinemarics_Request>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<dofbot_info::srv::Kinemarics_Request>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<dofbot_info::srv::Kinemarics_Request>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace dofbot_info
{

namespace srv
{

inline void to_flow_style_yaml(
  const Kinemarics_Response & msg,
  std::ostream & out)
{
  out << "{";
  // member: joint1
  {
    out << "joint1: ";
    rosidl_generator_traits::value_to_yaml(msg.joint1, out);
    out << ", ";
  }

  // member: joint2
  {
    out << "joint2: ";
    rosidl_generator_traits::value_to_yaml(msg.joint2, out);
    out << ", ";
  }

  // member: joint3
  {
    out << "joint3: ";
    rosidl_generator_traits::value_to_yaml(msg.joint3, out);
    out << ", ";
  }

  // member: joint4
  {
    out << "joint4: ";
    rosidl_generator_traits::value_to_yaml(msg.joint4, out);
    out << ", ";
  }

  // member: joint5
  {
    out << "joint5: ";
    rosidl_generator_traits::value_to_yaml(msg.joint5, out);
    out << ", ";
  }

  // member: joint6
  {
    out << "joint6: ";
    rosidl_generator_traits::value_to_yaml(msg.joint6, out);
    out << ", ";
  }

  // member: x
  {
    out << "x: ";
    rosidl_generator_traits::value_to_yaml(msg.x, out);
    out << ", ";
  }

  // member: y
  {
    out << "y: ";
    rosidl_generator_traits::value_to_yaml(msg.y, out);
    out << ", ";
  }

  // member: z
  {
    out << "z: ";
    rosidl_generator_traits::value_to_yaml(msg.z, out);
    out << ", ";
  }

  // member: roll
  {
    out << "roll: ";
    rosidl_generator_traits::value_to_yaml(msg.roll, out);
    out << ", ";
  }

  // member: pitch
  {
    out << "pitch: ";
    rosidl_generator_traits::value_to_yaml(msg.pitch, out);
    out << ", ";
  }

  // member: yaw
  {
    out << "yaw: ";
    rosidl_generator_traits::value_to_yaml(msg.yaw, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const Kinemarics_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: joint1
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "joint1: ";
    rosidl_generator_traits::value_to_yaml(msg.joint1, out);
    out << "\n";
  }

  // member: joint2
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "joint2: ";
    rosidl_generator_traits::value_to_yaml(msg.joint2, out);
    out << "\n";
  }

  // member: joint3
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "joint3: ";
    rosidl_generator_traits::value_to_yaml(msg.joint3, out);
    out << "\n";
  }

  // member: joint4
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "joint4: ";
    rosidl_generator_traits::value_to_yaml(msg.joint4, out);
    out << "\n";
  }

  // member: joint5
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "joint5: ";
    rosidl_generator_traits::value_to_yaml(msg.joint5, out);
    out << "\n";
  }

  // member: joint6
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "joint6: ";
    rosidl_generator_traits::value_to_yaml(msg.joint6, out);
    out << "\n";
  }

  // member: x
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "x: ";
    rosidl_generator_traits::value_to_yaml(msg.x, out);
    out << "\n";
  }

  // member: y
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "y: ";
    rosidl_generator_traits::value_to_yaml(msg.y, out);
    out << "\n";
  }

  // member: z
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "z: ";
    rosidl_generator_traits::value_to_yaml(msg.z, out);
    out << "\n";
  }

  // member: roll
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "roll: ";
    rosidl_generator_traits::value_to_yaml(msg.roll, out);
    out << "\n";
  }

  // member: pitch
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "pitch: ";
    rosidl_generator_traits::value_to_yaml(msg.pitch, out);
    out << "\n";
  }

  // member: yaw
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "yaw: ";
    rosidl_generator_traits::value_to_yaml(msg.yaw, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const Kinemarics_Response & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace srv

}  // namespace dofbot_info

namespace rosidl_generator_traits
{

[[deprecated("use dofbot_info::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const dofbot_info::srv::Kinemarics_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  dofbot_info::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use dofbot_info::srv::to_yaml() instead")]]
inline std::string to_yaml(const dofbot_info::srv::Kinemarics_Response & msg)
{
  return dofbot_info::srv::to_yaml(msg);
}

template<>
inline const char * data_type<dofbot_info::srv::Kinemarics_Response>()
{
  return "dofbot_info::srv::Kinemarics_Response";
}

template<>
inline const char * name<dofbot_info::srv::Kinemarics_Response>()
{
  return "dofbot_info/srv/Kinemarics_Response";
}

template<>
struct has_fixed_size<dofbot_info::srv::Kinemarics_Response>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<dofbot_info::srv::Kinemarics_Response>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<dofbot_info::srv::Kinemarics_Response>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<dofbot_info::srv::Kinemarics>()
{
  return "dofbot_info::srv::Kinemarics";
}

template<>
inline const char * name<dofbot_info::srv::Kinemarics>()
{
  return "dofbot_info/srv/Kinemarics";
}

template<>
struct has_fixed_size<dofbot_info::srv::Kinemarics>
  : std::integral_constant<
    bool,
    has_fixed_size<dofbot_info::srv::Kinemarics_Request>::value &&
    has_fixed_size<dofbot_info::srv::Kinemarics_Response>::value
  >
{
};

template<>
struct has_bounded_size<dofbot_info::srv::Kinemarics>
  : std::integral_constant<
    bool,
    has_bounded_size<dofbot_info::srv::Kinemarics_Request>::value &&
    has_bounded_size<dofbot_info::srv::Kinemarics_Response>::value
  >
{
};

template<>
struct is_service<dofbot_info::srv::Kinemarics>
  : std::true_type
{
};

template<>
struct is_service_request<dofbot_info::srv::Kinemarics_Request>
  : std::true_type
{
};

template<>
struct is_service_response<dofbot_info::srv::Kinemarics_Response>
  : std::true_type
{
};

}  // namespace rosidl_generator_traits

#endif  // DOFBOT_INFO__SRV__DETAIL__KINEMARICS__TRAITS_HPP_
