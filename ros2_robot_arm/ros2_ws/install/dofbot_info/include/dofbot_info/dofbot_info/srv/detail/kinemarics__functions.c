// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from dofbot_info:srv/Kinemarics.idl
// generated code does not contain a copyright notice
#include "dofbot_info/srv/detail/kinemarics__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"

// Include directives for member types
// Member `kin_name`
#include "rosidl_runtime_c/string_functions.h"

bool
dofbot_info__srv__Kinemarics_Request__init(dofbot_info__srv__Kinemarics_Request * msg)
{
  if (!msg) {
    return false;
  }
  // tar_x
  // tar_y
  // tar_z
  // roll
  // pitch
  // yaw
  // cur_joint1
  // cur_joint2
  // cur_joint3
  // cur_joint4
  // cur_joint5
  // cur_joint6
  // kin_name
  if (!rosidl_runtime_c__String__init(&msg->kin_name)) {
    dofbot_info__srv__Kinemarics_Request__fini(msg);
    return false;
  }
  return true;
}

void
dofbot_info__srv__Kinemarics_Request__fini(dofbot_info__srv__Kinemarics_Request * msg)
{
  if (!msg) {
    return;
  }
  // tar_x
  // tar_y
  // tar_z
  // roll
  // pitch
  // yaw
  // cur_joint1
  // cur_joint2
  // cur_joint3
  // cur_joint4
  // cur_joint5
  // cur_joint6
  // kin_name
  rosidl_runtime_c__String__fini(&msg->kin_name);
}

bool
dofbot_info__srv__Kinemarics_Request__are_equal(const dofbot_info__srv__Kinemarics_Request * lhs, const dofbot_info__srv__Kinemarics_Request * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // tar_x
  if (lhs->tar_x != rhs->tar_x) {
    return false;
  }
  // tar_y
  if (lhs->tar_y != rhs->tar_y) {
    return false;
  }
  // tar_z
  if (lhs->tar_z != rhs->tar_z) {
    return false;
  }
  // roll
  if (lhs->roll != rhs->roll) {
    return false;
  }
  // pitch
  if (lhs->pitch != rhs->pitch) {
    return false;
  }
  // yaw
  if (lhs->yaw != rhs->yaw) {
    return false;
  }
  // cur_joint1
  if (lhs->cur_joint1 != rhs->cur_joint1) {
    return false;
  }
  // cur_joint2
  if (lhs->cur_joint2 != rhs->cur_joint2) {
    return false;
  }
  // cur_joint3
  if (lhs->cur_joint3 != rhs->cur_joint3) {
    return false;
  }
  // cur_joint4
  if (lhs->cur_joint4 != rhs->cur_joint4) {
    return false;
  }
  // cur_joint5
  if (lhs->cur_joint5 != rhs->cur_joint5) {
    return false;
  }
  // cur_joint6
  if (lhs->cur_joint6 != rhs->cur_joint6) {
    return false;
  }
  // kin_name
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->kin_name), &(rhs->kin_name)))
  {
    return false;
  }
  return true;
}

bool
dofbot_info__srv__Kinemarics_Request__copy(
  const dofbot_info__srv__Kinemarics_Request * input,
  dofbot_info__srv__Kinemarics_Request * output)
{
  if (!input || !output) {
    return false;
  }
  // tar_x
  output->tar_x = input->tar_x;
  // tar_y
  output->tar_y = input->tar_y;
  // tar_z
  output->tar_z = input->tar_z;
  // roll
  output->roll = input->roll;
  // pitch
  output->pitch = input->pitch;
  // yaw
  output->yaw = input->yaw;
  // cur_joint1
  output->cur_joint1 = input->cur_joint1;
  // cur_joint2
  output->cur_joint2 = input->cur_joint2;
  // cur_joint3
  output->cur_joint3 = input->cur_joint3;
  // cur_joint4
  output->cur_joint4 = input->cur_joint4;
  // cur_joint5
  output->cur_joint5 = input->cur_joint5;
  // cur_joint6
  output->cur_joint6 = input->cur_joint6;
  // kin_name
  if (!rosidl_runtime_c__String__copy(
      &(input->kin_name), &(output->kin_name)))
  {
    return false;
  }
  return true;
}

dofbot_info__srv__Kinemarics_Request *
dofbot_info__srv__Kinemarics_Request__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  dofbot_info__srv__Kinemarics_Request * msg = (dofbot_info__srv__Kinemarics_Request *)allocator.allocate(sizeof(dofbot_info__srv__Kinemarics_Request), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(dofbot_info__srv__Kinemarics_Request));
  bool success = dofbot_info__srv__Kinemarics_Request__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
dofbot_info__srv__Kinemarics_Request__destroy(dofbot_info__srv__Kinemarics_Request * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    dofbot_info__srv__Kinemarics_Request__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
dofbot_info__srv__Kinemarics_Request__Sequence__init(dofbot_info__srv__Kinemarics_Request__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  dofbot_info__srv__Kinemarics_Request * data = NULL;

  if (size) {
    data = (dofbot_info__srv__Kinemarics_Request *)allocator.zero_allocate(size, sizeof(dofbot_info__srv__Kinemarics_Request), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = dofbot_info__srv__Kinemarics_Request__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        dofbot_info__srv__Kinemarics_Request__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
dofbot_info__srv__Kinemarics_Request__Sequence__fini(dofbot_info__srv__Kinemarics_Request__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      dofbot_info__srv__Kinemarics_Request__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

dofbot_info__srv__Kinemarics_Request__Sequence *
dofbot_info__srv__Kinemarics_Request__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  dofbot_info__srv__Kinemarics_Request__Sequence * array = (dofbot_info__srv__Kinemarics_Request__Sequence *)allocator.allocate(sizeof(dofbot_info__srv__Kinemarics_Request__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = dofbot_info__srv__Kinemarics_Request__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
dofbot_info__srv__Kinemarics_Request__Sequence__destroy(dofbot_info__srv__Kinemarics_Request__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    dofbot_info__srv__Kinemarics_Request__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
dofbot_info__srv__Kinemarics_Request__Sequence__are_equal(const dofbot_info__srv__Kinemarics_Request__Sequence * lhs, const dofbot_info__srv__Kinemarics_Request__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!dofbot_info__srv__Kinemarics_Request__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
dofbot_info__srv__Kinemarics_Request__Sequence__copy(
  const dofbot_info__srv__Kinemarics_Request__Sequence * input,
  dofbot_info__srv__Kinemarics_Request__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(dofbot_info__srv__Kinemarics_Request);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    dofbot_info__srv__Kinemarics_Request * data =
      (dofbot_info__srv__Kinemarics_Request *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!dofbot_info__srv__Kinemarics_Request__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          dofbot_info__srv__Kinemarics_Request__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!dofbot_info__srv__Kinemarics_Request__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}


bool
dofbot_info__srv__Kinemarics_Response__init(dofbot_info__srv__Kinemarics_Response * msg)
{
  if (!msg) {
    return false;
  }
  // joint1
  // joint2
  // joint3
  // joint4
  // joint5
  // joint6
  // x
  // y
  // z
  // roll
  // pitch
  // yaw
  return true;
}

void
dofbot_info__srv__Kinemarics_Response__fini(dofbot_info__srv__Kinemarics_Response * msg)
{
  if (!msg) {
    return;
  }
  // joint1
  // joint2
  // joint3
  // joint4
  // joint5
  // joint6
  // x
  // y
  // z
  // roll
  // pitch
  // yaw
}

bool
dofbot_info__srv__Kinemarics_Response__are_equal(const dofbot_info__srv__Kinemarics_Response * lhs, const dofbot_info__srv__Kinemarics_Response * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // joint1
  if (lhs->joint1 != rhs->joint1) {
    return false;
  }
  // joint2
  if (lhs->joint2 != rhs->joint2) {
    return false;
  }
  // joint3
  if (lhs->joint3 != rhs->joint3) {
    return false;
  }
  // joint4
  if (lhs->joint4 != rhs->joint4) {
    return false;
  }
  // joint5
  if (lhs->joint5 != rhs->joint5) {
    return false;
  }
  // joint6
  if (lhs->joint6 != rhs->joint6) {
    return false;
  }
  // x
  if (lhs->x != rhs->x) {
    return false;
  }
  // y
  if (lhs->y != rhs->y) {
    return false;
  }
  // z
  if (lhs->z != rhs->z) {
    return false;
  }
  // roll
  if (lhs->roll != rhs->roll) {
    return false;
  }
  // pitch
  if (lhs->pitch != rhs->pitch) {
    return false;
  }
  // yaw
  if (lhs->yaw != rhs->yaw) {
    return false;
  }
  return true;
}

bool
dofbot_info__srv__Kinemarics_Response__copy(
  const dofbot_info__srv__Kinemarics_Response * input,
  dofbot_info__srv__Kinemarics_Response * output)
{
  if (!input || !output) {
    return false;
  }
  // joint1
  output->joint1 = input->joint1;
  // joint2
  output->joint2 = input->joint2;
  // joint3
  output->joint3 = input->joint3;
  // joint4
  output->joint4 = input->joint4;
  // joint5
  output->joint5 = input->joint5;
  // joint6
  output->joint6 = input->joint6;
  // x
  output->x = input->x;
  // y
  output->y = input->y;
  // z
  output->z = input->z;
  // roll
  output->roll = input->roll;
  // pitch
  output->pitch = input->pitch;
  // yaw
  output->yaw = input->yaw;
  return true;
}

dofbot_info__srv__Kinemarics_Response *
dofbot_info__srv__Kinemarics_Response__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  dofbot_info__srv__Kinemarics_Response * msg = (dofbot_info__srv__Kinemarics_Response *)allocator.allocate(sizeof(dofbot_info__srv__Kinemarics_Response), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(dofbot_info__srv__Kinemarics_Response));
  bool success = dofbot_info__srv__Kinemarics_Response__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
dofbot_info__srv__Kinemarics_Response__destroy(dofbot_info__srv__Kinemarics_Response * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    dofbot_info__srv__Kinemarics_Response__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
dofbot_info__srv__Kinemarics_Response__Sequence__init(dofbot_info__srv__Kinemarics_Response__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  dofbot_info__srv__Kinemarics_Response * data = NULL;

  if (size) {
    data = (dofbot_info__srv__Kinemarics_Response *)allocator.zero_allocate(size, sizeof(dofbot_info__srv__Kinemarics_Response), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = dofbot_info__srv__Kinemarics_Response__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        dofbot_info__srv__Kinemarics_Response__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
dofbot_info__srv__Kinemarics_Response__Sequence__fini(dofbot_info__srv__Kinemarics_Response__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      dofbot_info__srv__Kinemarics_Response__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

dofbot_info__srv__Kinemarics_Response__Sequence *
dofbot_info__srv__Kinemarics_Response__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  dofbot_info__srv__Kinemarics_Response__Sequence * array = (dofbot_info__srv__Kinemarics_Response__Sequence *)allocator.allocate(sizeof(dofbot_info__srv__Kinemarics_Response__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = dofbot_info__srv__Kinemarics_Response__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
dofbot_info__srv__Kinemarics_Response__Sequence__destroy(dofbot_info__srv__Kinemarics_Response__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    dofbot_info__srv__Kinemarics_Response__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
dofbot_info__srv__Kinemarics_Response__Sequence__are_equal(const dofbot_info__srv__Kinemarics_Response__Sequence * lhs, const dofbot_info__srv__Kinemarics_Response__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!dofbot_info__srv__Kinemarics_Response__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
dofbot_info__srv__Kinemarics_Response__Sequence__copy(
  const dofbot_info__srv__Kinemarics_Response__Sequence * input,
  dofbot_info__srv__Kinemarics_Response__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(dofbot_info__srv__Kinemarics_Response);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    dofbot_info__srv__Kinemarics_Response * data =
      (dofbot_info__srv__Kinemarics_Response *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!dofbot_info__srv__Kinemarics_Response__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          dofbot_info__srv__Kinemarics_Response__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!dofbot_info__srv__Kinemarics_Response__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
