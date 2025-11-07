// generated from rosidl_generator_py/resource/_idl_support.c.em
// with input from dofbot_info:srv/Kinemarics.idl
// generated code does not contain a copyright notice
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <stdbool.h>
#ifndef _WIN32
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wunused-function"
#endif
#include "numpy/ndarrayobject.h"
#ifndef _WIN32
# pragma GCC diagnostic pop
#endif
#include "rosidl_runtime_c/visibility_control.h"
#include "dofbot_info/srv/detail/kinemarics__struct.h"
#include "dofbot_info/srv/detail/kinemarics__functions.h"

#include "rosidl_runtime_c/string.h"
#include "rosidl_runtime_c/string_functions.h"


ROSIDL_GENERATOR_C_EXPORT
bool dofbot_info__srv__kinemarics__request__convert_from_py(PyObject * _pymsg, void * _ros_message)
{
  // check that the passed message is of the expected Python class
  {
    char full_classname_dest[47];
    {
      char * class_name = NULL;
      char * module_name = NULL;
      {
        PyObject * class_attr = PyObject_GetAttrString(_pymsg, "__class__");
        if (class_attr) {
          PyObject * name_attr = PyObject_GetAttrString(class_attr, "__name__");
          if (name_attr) {
            class_name = (char *)PyUnicode_1BYTE_DATA(name_attr);
            Py_DECREF(name_attr);
          }
          PyObject * module_attr = PyObject_GetAttrString(class_attr, "__module__");
          if (module_attr) {
            module_name = (char *)PyUnicode_1BYTE_DATA(module_attr);
            Py_DECREF(module_attr);
          }
          Py_DECREF(class_attr);
        }
      }
      if (!class_name || !module_name) {
        return false;
      }
      snprintf(full_classname_dest, sizeof(full_classname_dest), "%s.%s", module_name, class_name);
    }
    assert(strncmp("dofbot_info.srv._kinemarics.Kinemarics_Request", full_classname_dest, 46) == 0);
  }
  dofbot_info__srv__Kinemarics_Request * ros_message = _ros_message;
  {  // tar_x
    PyObject * field = PyObject_GetAttrString(_pymsg, "tar_x");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->tar_x = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // tar_y
    PyObject * field = PyObject_GetAttrString(_pymsg, "tar_y");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->tar_y = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // tar_z
    PyObject * field = PyObject_GetAttrString(_pymsg, "tar_z");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->tar_z = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // roll
    PyObject * field = PyObject_GetAttrString(_pymsg, "roll");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->roll = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // pitch
    PyObject * field = PyObject_GetAttrString(_pymsg, "pitch");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->pitch = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // yaw
    PyObject * field = PyObject_GetAttrString(_pymsg, "yaw");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->yaw = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // cur_joint1
    PyObject * field = PyObject_GetAttrString(_pymsg, "cur_joint1");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->cur_joint1 = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // cur_joint2
    PyObject * field = PyObject_GetAttrString(_pymsg, "cur_joint2");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->cur_joint2 = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // cur_joint3
    PyObject * field = PyObject_GetAttrString(_pymsg, "cur_joint3");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->cur_joint3 = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // cur_joint4
    PyObject * field = PyObject_GetAttrString(_pymsg, "cur_joint4");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->cur_joint4 = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // cur_joint5
    PyObject * field = PyObject_GetAttrString(_pymsg, "cur_joint5");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->cur_joint5 = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // cur_joint6
    PyObject * field = PyObject_GetAttrString(_pymsg, "cur_joint6");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->cur_joint6 = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // kin_name
    PyObject * field = PyObject_GetAttrString(_pymsg, "kin_name");
    if (!field) {
      return false;
    }
    assert(PyUnicode_Check(field));
    PyObject * encoded_field = PyUnicode_AsUTF8String(field);
    if (!encoded_field) {
      Py_DECREF(field);
      return false;
    }
    rosidl_runtime_c__String__assign(&ros_message->kin_name, PyBytes_AS_STRING(encoded_field));
    Py_DECREF(encoded_field);
    Py_DECREF(field);
  }

  return true;
}

ROSIDL_GENERATOR_C_EXPORT
PyObject * dofbot_info__srv__kinemarics__request__convert_to_py(void * raw_ros_message)
{
  /* NOTE(esteve): Call constructor of Kinemarics_Request */
  PyObject * _pymessage = NULL;
  {
    PyObject * pymessage_module = PyImport_ImportModule("dofbot_info.srv._kinemarics");
    assert(pymessage_module);
    PyObject * pymessage_class = PyObject_GetAttrString(pymessage_module, "Kinemarics_Request");
    assert(pymessage_class);
    Py_DECREF(pymessage_module);
    _pymessage = PyObject_CallObject(pymessage_class, NULL);
    Py_DECREF(pymessage_class);
    if (!_pymessage) {
      return NULL;
    }
  }
  dofbot_info__srv__Kinemarics_Request * ros_message = (dofbot_info__srv__Kinemarics_Request *)raw_ros_message;
  {  // tar_x
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->tar_x);
    {
      int rc = PyObject_SetAttrString(_pymessage, "tar_x", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // tar_y
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->tar_y);
    {
      int rc = PyObject_SetAttrString(_pymessage, "tar_y", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // tar_z
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->tar_z);
    {
      int rc = PyObject_SetAttrString(_pymessage, "tar_z", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // roll
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->roll);
    {
      int rc = PyObject_SetAttrString(_pymessage, "roll", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // pitch
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->pitch);
    {
      int rc = PyObject_SetAttrString(_pymessage, "pitch", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // yaw
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->yaw);
    {
      int rc = PyObject_SetAttrString(_pymessage, "yaw", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // cur_joint1
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->cur_joint1);
    {
      int rc = PyObject_SetAttrString(_pymessage, "cur_joint1", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // cur_joint2
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->cur_joint2);
    {
      int rc = PyObject_SetAttrString(_pymessage, "cur_joint2", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // cur_joint3
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->cur_joint3);
    {
      int rc = PyObject_SetAttrString(_pymessage, "cur_joint3", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // cur_joint4
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->cur_joint4);
    {
      int rc = PyObject_SetAttrString(_pymessage, "cur_joint4", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // cur_joint5
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->cur_joint5);
    {
      int rc = PyObject_SetAttrString(_pymessage, "cur_joint5", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // cur_joint6
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->cur_joint6);
    {
      int rc = PyObject_SetAttrString(_pymessage, "cur_joint6", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // kin_name
    PyObject * field = NULL;
    field = PyUnicode_DecodeUTF8(
      ros_message->kin_name.data,
      strlen(ros_message->kin_name.data),
      "replace");
    if (!field) {
      return NULL;
    }
    {
      int rc = PyObject_SetAttrString(_pymessage, "kin_name", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }

  // ownership of _pymessage is transferred to the caller
  return _pymessage;
}

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
// already included above
// #include <Python.h>
// already included above
// #include <stdbool.h>
// already included above
// #include "numpy/ndarrayobject.h"
// already included above
// #include "rosidl_runtime_c/visibility_control.h"
// already included above
// #include "dofbot_info/srv/detail/kinemarics__struct.h"
// already included above
// #include "dofbot_info/srv/detail/kinemarics__functions.h"


ROSIDL_GENERATOR_C_EXPORT
bool dofbot_info__srv__kinemarics__response__convert_from_py(PyObject * _pymsg, void * _ros_message)
{
  // check that the passed message is of the expected Python class
  {
    char full_classname_dest[48];
    {
      char * class_name = NULL;
      char * module_name = NULL;
      {
        PyObject * class_attr = PyObject_GetAttrString(_pymsg, "__class__");
        if (class_attr) {
          PyObject * name_attr = PyObject_GetAttrString(class_attr, "__name__");
          if (name_attr) {
            class_name = (char *)PyUnicode_1BYTE_DATA(name_attr);
            Py_DECREF(name_attr);
          }
          PyObject * module_attr = PyObject_GetAttrString(class_attr, "__module__");
          if (module_attr) {
            module_name = (char *)PyUnicode_1BYTE_DATA(module_attr);
            Py_DECREF(module_attr);
          }
          Py_DECREF(class_attr);
        }
      }
      if (!class_name || !module_name) {
        return false;
      }
      snprintf(full_classname_dest, sizeof(full_classname_dest), "%s.%s", module_name, class_name);
    }
    assert(strncmp("dofbot_info.srv._kinemarics.Kinemarics_Response", full_classname_dest, 47) == 0);
  }
  dofbot_info__srv__Kinemarics_Response * ros_message = _ros_message;
  {  // joint1
    PyObject * field = PyObject_GetAttrString(_pymsg, "joint1");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->joint1 = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // joint2
    PyObject * field = PyObject_GetAttrString(_pymsg, "joint2");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->joint2 = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // joint3
    PyObject * field = PyObject_GetAttrString(_pymsg, "joint3");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->joint3 = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // joint4
    PyObject * field = PyObject_GetAttrString(_pymsg, "joint4");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->joint4 = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // joint5
    PyObject * field = PyObject_GetAttrString(_pymsg, "joint5");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->joint5 = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // joint6
    PyObject * field = PyObject_GetAttrString(_pymsg, "joint6");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->joint6 = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // x
    PyObject * field = PyObject_GetAttrString(_pymsg, "x");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->x = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // y
    PyObject * field = PyObject_GetAttrString(_pymsg, "y");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->y = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // z
    PyObject * field = PyObject_GetAttrString(_pymsg, "z");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->z = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // roll
    PyObject * field = PyObject_GetAttrString(_pymsg, "roll");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->roll = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // pitch
    PyObject * field = PyObject_GetAttrString(_pymsg, "pitch");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->pitch = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // yaw
    PyObject * field = PyObject_GetAttrString(_pymsg, "yaw");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->yaw = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }

  return true;
}

ROSIDL_GENERATOR_C_EXPORT
PyObject * dofbot_info__srv__kinemarics__response__convert_to_py(void * raw_ros_message)
{
  /* NOTE(esteve): Call constructor of Kinemarics_Response */
  PyObject * _pymessage = NULL;
  {
    PyObject * pymessage_module = PyImport_ImportModule("dofbot_info.srv._kinemarics");
    assert(pymessage_module);
    PyObject * pymessage_class = PyObject_GetAttrString(pymessage_module, "Kinemarics_Response");
    assert(pymessage_class);
    Py_DECREF(pymessage_module);
    _pymessage = PyObject_CallObject(pymessage_class, NULL);
    Py_DECREF(pymessage_class);
    if (!_pymessage) {
      return NULL;
    }
  }
  dofbot_info__srv__Kinemarics_Response * ros_message = (dofbot_info__srv__Kinemarics_Response *)raw_ros_message;
  {  // joint1
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->joint1);
    {
      int rc = PyObject_SetAttrString(_pymessage, "joint1", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // joint2
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->joint2);
    {
      int rc = PyObject_SetAttrString(_pymessage, "joint2", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // joint3
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->joint3);
    {
      int rc = PyObject_SetAttrString(_pymessage, "joint3", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // joint4
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->joint4);
    {
      int rc = PyObject_SetAttrString(_pymessage, "joint4", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // joint5
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->joint5);
    {
      int rc = PyObject_SetAttrString(_pymessage, "joint5", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // joint6
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->joint6);
    {
      int rc = PyObject_SetAttrString(_pymessage, "joint6", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // x
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->x);
    {
      int rc = PyObject_SetAttrString(_pymessage, "x", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // y
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->y);
    {
      int rc = PyObject_SetAttrString(_pymessage, "y", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // z
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->z);
    {
      int rc = PyObject_SetAttrString(_pymessage, "z", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // roll
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->roll);
    {
      int rc = PyObject_SetAttrString(_pymessage, "roll", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // pitch
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->pitch);
    {
      int rc = PyObject_SetAttrString(_pymessage, "pitch", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // yaw
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->yaw);
    {
      int rc = PyObject_SetAttrString(_pymessage, "yaw", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }

  // ownership of _pymessage is transferred to the caller
  return _pymessage;
}
