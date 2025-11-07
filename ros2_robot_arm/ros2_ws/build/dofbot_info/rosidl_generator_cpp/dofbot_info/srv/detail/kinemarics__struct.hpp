// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from dofbot_info:srv/Kinemarics.idl
// generated code does not contain a copyright notice

#ifndef DOFBOT_INFO__SRV__DETAIL__KINEMARICS__STRUCT_HPP_
#define DOFBOT_INFO__SRV__DETAIL__KINEMARICS__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__dofbot_info__srv__Kinemarics_Request __attribute__((deprecated))
#else
# define DEPRECATED__dofbot_info__srv__Kinemarics_Request __declspec(deprecated)
#endif

namespace dofbot_info
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct Kinemarics_Request_
{
  using Type = Kinemarics_Request_<ContainerAllocator>;

  explicit Kinemarics_Request_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->tar_x = 0.0;
      this->tar_y = 0.0;
      this->tar_z = 0.0;
      this->roll = 0.0;
      this->pitch = 0.0;
      this->yaw = 0.0;
      this->cur_joint1 = 0.0;
      this->cur_joint2 = 0.0;
      this->cur_joint3 = 0.0;
      this->cur_joint4 = 0.0;
      this->cur_joint5 = 0.0;
      this->cur_joint6 = 0.0;
      this->kin_name = "";
    }
  }

  explicit Kinemarics_Request_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : kin_name(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->tar_x = 0.0;
      this->tar_y = 0.0;
      this->tar_z = 0.0;
      this->roll = 0.0;
      this->pitch = 0.0;
      this->yaw = 0.0;
      this->cur_joint1 = 0.0;
      this->cur_joint2 = 0.0;
      this->cur_joint3 = 0.0;
      this->cur_joint4 = 0.0;
      this->cur_joint5 = 0.0;
      this->cur_joint6 = 0.0;
      this->kin_name = "";
    }
  }

  // field types and members
  using _tar_x_type =
    double;
  _tar_x_type tar_x;
  using _tar_y_type =
    double;
  _tar_y_type tar_y;
  using _tar_z_type =
    double;
  _tar_z_type tar_z;
  using _roll_type =
    double;
  _roll_type roll;
  using _pitch_type =
    double;
  _pitch_type pitch;
  using _yaw_type =
    double;
  _yaw_type yaw;
  using _cur_joint1_type =
    double;
  _cur_joint1_type cur_joint1;
  using _cur_joint2_type =
    double;
  _cur_joint2_type cur_joint2;
  using _cur_joint3_type =
    double;
  _cur_joint3_type cur_joint3;
  using _cur_joint4_type =
    double;
  _cur_joint4_type cur_joint4;
  using _cur_joint5_type =
    double;
  _cur_joint5_type cur_joint5;
  using _cur_joint6_type =
    double;
  _cur_joint6_type cur_joint6;
  using _kin_name_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _kin_name_type kin_name;

  // setters for named parameter idiom
  Type & set__tar_x(
    const double & _arg)
  {
    this->tar_x = _arg;
    return *this;
  }
  Type & set__tar_y(
    const double & _arg)
  {
    this->tar_y = _arg;
    return *this;
  }
  Type & set__tar_z(
    const double & _arg)
  {
    this->tar_z = _arg;
    return *this;
  }
  Type & set__roll(
    const double & _arg)
  {
    this->roll = _arg;
    return *this;
  }
  Type & set__pitch(
    const double & _arg)
  {
    this->pitch = _arg;
    return *this;
  }
  Type & set__yaw(
    const double & _arg)
  {
    this->yaw = _arg;
    return *this;
  }
  Type & set__cur_joint1(
    const double & _arg)
  {
    this->cur_joint1 = _arg;
    return *this;
  }
  Type & set__cur_joint2(
    const double & _arg)
  {
    this->cur_joint2 = _arg;
    return *this;
  }
  Type & set__cur_joint3(
    const double & _arg)
  {
    this->cur_joint3 = _arg;
    return *this;
  }
  Type & set__cur_joint4(
    const double & _arg)
  {
    this->cur_joint4 = _arg;
    return *this;
  }
  Type & set__cur_joint5(
    const double & _arg)
  {
    this->cur_joint5 = _arg;
    return *this;
  }
  Type & set__cur_joint6(
    const double & _arg)
  {
    this->cur_joint6 = _arg;
    return *this;
  }
  Type & set__kin_name(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->kin_name = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    dofbot_info::srv::Kinemarics_Request_<ContainerAllocator> *;
  using ConstRawPtr =
    const dofbot_info::srv::Kinemarics_Request_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<dofbot_info::srv::Kinemarics_Request_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<dofbot_info::srv::Kinemarics_Request_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      dofbot_info::srv::Kinemarics_Request_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<dofbot_info::srv::Kinemarics_Request_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      dofbot_info::srv::Kinemarics_Request_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<dofbot_info::srv::Kinemarics_Request_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<dofbot_info::srv::Kinemarics_Request_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<dofbot_info::srv::Kinemarics_Request_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__dofbot_info__srv__Kinemarics_Request
    std::shared_ptr<dofbot_info::srv::Kinemarics_Request_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__dofbot_info__srv__Kinemarics_Request
    std::shared_ptr<dofbot_info::srv::Kinemarics_Request_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const Kinemarics_Request_ & other) const
  {
    if (this->tar_x != other.tar_x) {
      return false;
    }
    if (this->tar_y != other.tar_y) {
      return false;
    }
    if (this->tar_z != other.tar_z) {
      return false;
    }
    if (this->roll != other.roll) {
      return false;
    }
    if (this->pitch != other.pitch) {
      return false;
    }
    if (this->yaw != other.yaw) {
      return false;
    }
    if (this->cur_joint1 != other.cur_joint1) {
      return false;
    }
    if (this->cur_joint2 != other.cur_joint2) {
      return false;
    }
    if (this->cur_joint3 != other.cur_joint3) {
      return false;
    }
    if (this->cur_joint4 != other.cur_joint4) {
      return false;
    }
    if (this->cur_joint5 != other.cur_joint5) {
      return false;
    }
    if (this->cur_joint6 != other.cur_joint6) {
      return false;
    }
    if (this->kin_name != other.kin_name) {
      return false;
    }
    return true;
  }
  bool operator!=(const Kinemarics_Request_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct Kinemarics_Request_

// alias to use template instance with default allocator
using Kinemarics_Request =
  dofbot_info::srv::Kinemarics_Request_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace dofbot_info


#ifndef _WIN32
# define DEPRECATED__dofbot_info__srv__Kinemarics_Response __attribute__((deprecated))
#else
# define DEPRECATED__dofbot_info__srv__Kinemarics_Response __declspec(deprecated)
#endif

namespace dofbot_info
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct Kinemarics_Response_
{
  using Type = Kinemarics_Response_<ContainerAllocator>;

  explicit Kinemarics_Response_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->joint1 = 0.0;
      this->joint2 = 0.0;
      this->joint3 = 0.0;
      this->joint4 = 0.0;
      this->joint5 = 0.0;
      this->joint6 = 0.0;
      this->x = 0.0;
      this->y = 0.0;
      this->z = 0.0;
      this->roll = 0.0;
      this->pitch = 0.0;
      this->yaw = 0.0;
    }
  }

  explicit Kinemarics_Response_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->joint1 = 0.0;
      this->joint2 = 0.0;
      this->joint3 = 0.0;
      this->joint4 = 0.0;
      this->joint5 = 0.0;
      this->joint6 = 0.0;
      this->x = 0.0;
      this->y = 0.0;
      this->z = 0.0;
      this->roll = 0.0;
      this->pitch = 0.0;
      this->yaw = 0.0;
    }
  }

  // field types and members
  using _joint1_type =
    double;
  _joint1_type joint1;
  using _joint2_type =
    double;
  _joint2_type joint2;
  using _joint3_type =
    double;
  _joint3_type joint3;
  using _joint4_type =
    double;
  _joint4_type joint4;
  using _joint5_type =
    double;
  _joint5_type joint5;
  using _joint6_type =
    double;
  _joint6_type joint6;
  using _x_type =
    double;
  _x_type x;
  using _y_type =
    double;
  _y_type y;
  using _z_type =
    double;
  _z_type z;
  using _roll_type =
    double;
  _roll_type roll;
  using _pitch_type =
    double;
  _pitch_type pitch;
  using _yaw_type =
    double;
  _yaw_type yaw;

  // setters for named parameter idiom
  Type & set__joint1(
    const double & _arg)
  {
    this->joint1 = _arg;
    return *this;
  }
  Type & set__joint2(
    const double & _arg)
  {
    this->joint2 = _arg;
    return *this;
  }
  Type & set__joint3(
    const double & _arg)
  {
    this->joint3 = _arg;
    return *this;
  }
  Type & set__joint4(
    const double & _arg)
  {
    this->joint4 = _arg;
    return *this;
  }
  Type & set__joint5(
    const double & _arg)
  {
    this->joint5 = _arg;
    return *this;
  }
  Type & set__joint6(
    const double & _arg)
  {
    this->joint6 = _arg;
    return *this;
  }
  Type & set__x(
    const double & _arg)
  {
    this->x = _arg;
    return *this;
  }
  Type & set__y(
    const double & _arg)
  {
    this->y = _arg;
    return *this;
  }
  Type & set__z(
    const double & _arg)
  {
    this->z = _arg;
    return *this;
  }
  Type & set__roll(
    const double & _arg)
  {
    this->roll = _arg;
    return *this;
  }
  Type & set__pitch(
    const double & _arg)
  {
    this->pitch = _arg;
    return *this;
  }
  Type & set__yaw(
    const double & _arg)
  {
    this->yaw = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    dofbot_info::srv::Kinemarics_Response_<ContainerAllocator> *;
  using ConstRawPtr =
    const dofbot_info::srv::Kinemarics_Response_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<dofbot_info::srv::Kinemarics_Response_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<dofbot_info::srv::Kinemarics_Response_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      dofbot_info::srv::Kinemarics_Response_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<dofbot_info::srv::Kinemarics_Response_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      dofbot_info::srv::Kinemarics_Response_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<dofbot_info::srv::Kinemarics_Response_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<dofbot_info::srv::Kinemarics_Response_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<dofbot_info::srv::Kinemarics_Response_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__dofbot_info__srv__Kinemarics_Response
    std::shared_ptr<dofbot_info::srv::Kinemarics_Response_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__dofbot_info__srv__Kinemarics_Response
    std::shared_ptr<dofbot_info::srv::Kinemarics_Response_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const Kinemarics_Response_ & other) const
  {
    if (this->joint1 != other.joint1) {
      return false;
    }
    if (this->joint2 != other.joint2) {
      return false;
    }
    if (this->joint3 != other.joint3) {
      return false;
    }
    if (this->joint4 != other.joint4) {
      return false;
    }
    if (this->joint5 != other.joint5) {
      return false;
    }
    if (this->joint6 != other.joint6) {
      return false;
    }
    if (this->x != other.x) {
      return false;
    }
    if (this->y != other.y) {
      return false;
    }
    if (this->z != other.z) {
      return false;
    }
    if (this->roll != other.roll) {
      return false;
    }
    if (this->pitch != other.pitch) {
      return false;
    }
    if (this->yaw != other.yaw) {
      return false;
    }
    return true;
  }
  bool operator!=(const Kinemarics_Response_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct Kinemarics_Response_

// alias to use template instance with default allocator
using Kinemarics_Response =
  dofbot_info::srv::Kinemarics_Response_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace dofbot_info

namespace dofbot_info
{

namespace srv
{

struct Kinemarics
{
  using Request = dofbot_info::srv::Kinemarics_Request;
  using Response = dofbot_info::srv::Kinemarics_Response;
};

}  // namespace srv

}  // namespace dofbot_info

#endif  // DOFBOT_INFO__SRV__DETAIL__KINEMARICS__STRUCT_HPP_
