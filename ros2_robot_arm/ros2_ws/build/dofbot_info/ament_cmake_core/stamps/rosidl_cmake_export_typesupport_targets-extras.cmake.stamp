# generated from
# rosidl_cmake/cmake/template/rosidl_cmake_export_typesupport_targets.cmake.in

set(_exported_typesupport_targets
  "__rosidl_generator_c:dofbot_info__rosidl_generator_c;__rosidl_typesupport_fastrtps_c:dofbot_info__rosidl_typesupport_fastrtps_c;__rosidl_typesupport_introspection_c:dofbot_info__rosidl_typesupport_introspection_c;__rosidl_typesupport_c:dofbot_info__rosidl_typesupport_c;__rosidl_generator_cpp:dofbot_info__rosidl_generator_cpp;__rosidl_typesupport_fastrtps_cpp:dofbot_info__rosidl_typesupport_fastrtps_cpp;__rosidl_typesupport_introspection_cpp:dofbot_info__rosidl_typesupport_introspection_cpp;__rosidl_typesupport_cpp:dofbot_info__rosidl_typesupport_cpp;__rosidl_generator_py:dofbot_info__rosidl_generator_py")

# populate dofbot_info_TARGETS_<suffix>
if(NOT _exported_typesupport_targets STREQUAL "")
  # loop over typesupport targets
  foreach(_tuple ${_exported_typesupport_targets})
    string(REPLACE ":" ";" _tuple "${_tuple}")
    list(GET _tuple 0 _suffix)
    list(GET _tuple 1 _target)

    set(_target "dofbot_info::${_target}")
    if(NOT TARGET "${_target}")
      # the exported target must exist
      message(WARNING "Package 'dofbot_info' exports the typesupport target '${_target}' which doesn't exist")
    else()
      list(APPEND dofbot_info_TARGETS${_suffix} "${_target}")
    endif()
  endforeach()
endif()
