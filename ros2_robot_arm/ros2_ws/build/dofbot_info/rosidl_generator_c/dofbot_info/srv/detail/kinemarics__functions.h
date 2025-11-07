// generated from rosidl_generator_c/resource/idl__functions.h.em
// with input from dofbot_info:srv/Kinemarics.idl
// generated code does not contain a copyright notice

#ifndef DOFBOT_INFO__SRV__DETAIL__KINEMARICS__FUNCTIONS_H_
#define DOFBOT_INFO__SRV__DETAIL__KINEMARICS__FUNCTIONS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdlib.h>

#include "rosidl_runtime_c/visibility_control.h"
#include "dofbot_info/msg/rosidl_generator_c__visibility_control.h"

#include "dofbot_info/srv/detail/kinemarics__struct.h"

/// Initialize srv/Kinemarics message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * dofbot_info__srv__Kinemarics_Request
 * )) before or use
 * dofbot_info__srv__Kinemarics_Request__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_dofbot_info
bool
dofbot_info__srv__Kinemarics_Request__init(dofbot_info__srv__Kinemarics_Request * msg);

/// Finalize srv/Kinemarics message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_dofbot_info
void
dofbot_info__srv__Kinemarics_Request__fini(dofbot_info__srv__Kinemarics_Request * msg);

/// Create srv/Kinemarics message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * dofbot_info__srv__Kinemarics_Request__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_dofbot_info
dofbot_info__srv__Kinemarics_Request *
dofbot_info__srv__Kinemarics_Request__create();

/// Destroy srv/Kinemarics message.
/**
 * It calls
 * dofbot_info__srv__Kinemarics_Request__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_dofbot_info
void
dofbot_info__srv__Kinemarics_Request__destroy(dofbot_info__srv__Kinemarics_Request * msg);

/// Check for srv/Kinemarics message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_dofbot_info
bool
dofbot_info__srv__Kinemarics_Request__are_equal(const dofbot_info__srv__Kinemarics_Request * lhs, const dofbot_info__srv__Kinemarics_Request * rhs);

/// Copy a srv/Kinemarics message.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source message pointer.
 * \param[out] output The target message pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer is null
 *   or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_dofbot_info
bool
dofbot_info__srv__Kinemarics_Request__copy(
  const dofbot_info__srv__Kinemarics_Request * input,
  dofbot_info__srv__Kinemarics_Request * output);

/// Initialize array of srv/Kinemarics messages.
/**
 * It allocates the memory for the number of elements and calls
 * dofbot_info__srv__Kinemarics_Request__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_dofbot_info
bool
dofbot_info__srv__Kinemarics_Request__Sequence__init(dofbot_info__srv__Kinemarics_Request__Sequence * array, size_t size);

/// Finalize array of srv/Kinemarics messages.
/**
 * It calls
 * dofbot_info__srv__Kinemarics_Request__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_dofbot_info
void
dofbot_info__srv__Kinemarics_Request__Sequence__fini(dofbot_info__srv__Kinemarics_Request__Sequence * array);

/// Create array of srv/Kinemarics messages.
/**
 * It allocates the memory for the array and calls
 * dofbot_info__srv__Kinemarics_Request__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_dofbot_info
dofbot_info__srv__Kinemarics_Request__Sequence *
dofbot_info__srv__Kinemarics_Request__Sequence__create(size_t size);

/// Destroy array of srv/Kinemarics messages.
/**
 * It calls
 * dofbot_info__srv__Kinemarics_Request__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_dofbot_info
void
dofbot_info__srv__Kinemarics_Request__Sequence__destroy(dofbot_info__srv__Kinemarics_Request__Sequence * array);

/// Check for srv/Kinemarics message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_dofbot_info
bool
dofbot_info__srv__Kinemarics_Request__Sequence__are_equal(const dofbot_info__srv__Kinemarics_Request__Sequence * lhs, const dofbot_info__srv__Kinemarics_Request__Sequence * rhs);

/// Copy an array of srv/Kinemarics messages.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source array pointer.
 * \param[out] output The target array pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer
 *   is null or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_dofbot_info
bool
dofbot_info__srv__Kinemarics_Request__Sequence__copy(
  const dofbot_info__srv__Kinemarics_Request__Sequence * input,
  dofbot_info__srv__Kinemarics_Request__Sequence * output);

/// Initialize srv/Kinemarics message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * dofbot_info__srv__Kinemarics_Response
 * )) before or use
 * dofbot_info__srv__Kinemarics_Response__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_dofbot_info
bool
dofbot_info__srv__Kinemarics_Response__init(dofbot_info__srv__Kinemarics_Response * msg);

/// Finalize srv/Kinemarics message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_dofbot_info
void
dofbot_info__srv__Kinemarics_Response__fini(dofbot_info__srv__Kinemarics_Response * msg);

/// Create srv/Kinemarics message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * dofbot_info__srv__Kinemarics_Response__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_dofbot_info
dofbot_info__srv__Kinemarics_Response *
dofbot_info__srv__Kinemarics_Response__create();

/// Destroy srv/Kinemarics message.
/**
 * It calls
 * dofbot_info__srv__Kinemarics_Response__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_dofbot_info
void
dofbot_info__srv__Kinemarics_Response__destroy(dofbot_info__srv__Kinemarics_Response * msg);

/// Check for srv/Kinemarics message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_dofbot_info
bool
dofbot_info__srv__Kinemarics_Response__are_equal(const dofbot_info__srv__Kinemarics_Response * lhs, const dofbot_info__srv__Kinemarics_Response * rhs);

/// Copy a srv/Kinemarics message.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source message pointer.
 * \param[out] output The target message pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer is null
 *   or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_dofbot_info
bool
dofbot_info__srv__Kinemarics_Response__copy(
  const dofbot_info__srv__Kinemarics_Response * input,
  dofbot_info__srv__Kinemarics_Response * output);

/// Initialize array of srv/Kinemarics messages.
/**
 * It allocates the memory for the number of elements and calls
 * dofbot_info__srv__Kinemarics_Response__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_dofbot_info
bool
dofbot_info__srv__Kinemarics_Response__Sequence__init(dofbot_info__srv__Kinemarics_Response__Sequence * array, size_t size);

/// Finalize array of srv/Kinemarics messages.
/**
 * It calls
 * dofbot_info__srv__Kinemarics_Response__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_dofbot_info
void
dofbot_info__srv__Kinemarics_Response__Sequence__fini(dofbot_info__srv__Kinemarics_Response__Sequence * array);

/// Create array of srv/Kinemarics messages.
/**
 * It allocates the memory for the array and calls
 * dofbot_info__srv__Kinemarics_Response__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_dofbot_info
dofbot_info__srv__Kinemarics_Response__Sequence *
dofbot_info__srv__Kinemarics_Response__Sequence__create(size_t size);

/// Destroy array of srv/Kinemarics messages.
/**
 * It calls
 * dofbot_info__srv__Kinemarics_Response__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_dofbot_info
void
dofbot_info__srv__Kinemarics_Response__Sequence__destroy(dofbot_info__srv__Kinemarics_Response__Sequence * array);

/// Check for srv/Kinemarics message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_dofbot_info
bool
dofbot_info__srv__Kinemarics_Response__Sequence__are_equal(const dofbot_info__srv__Kinemarics_Response__Sequence * lhs, const dofbot_info__srv__Kinemarics_Response__Sequence * rhs);

/// Copy an array of srv/Kinemarics messages.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source array pointer.
 * \param[out] output The target array pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer
 *   is null or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_dofbot_info
bool
dofbot_info__srv__Kinemarics_Response__Sequence__copy(
  const dofbot_info__srv__Kinemarics_Response__Sequence * input,
  dofbot_info__srv__Kinemarics_Response__Sequence * output);

#ifdef __cplusplus
}
#endif

#endif  // DOFBOT_INFO__SRV__DETAIL__KINEMARICS__FUNCTIONS_H_
