# Install script for directory: /root/ros2_robot_arm/orocos_kdl/src

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/liborocos-kdl.so.1.4.0"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/liborocos-kdl.so.1.4"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      file(RPATH_CHECK
           FILE "${file}"
           RPATH "$ORIGIN/../lib")
    endif()
  endforeach()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES
    "/root/ros2_robot_arm/orocos_kdl/build/src/liborocos-kdl.so.1.4.0"
    "/root/ros2_robot_arm/orocos_kdl/build/src/liborocos-kdl.so.1.4"
    )
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/liborocos-kdl.so.1.4.0"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/liborocos-kdl.so.1.4"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      file(RPATH_CHANGE
           FILE "${file}"
           OLD_RPATH "::::::::::::::"
           NEW_RPATH "$ORIGIN/../lib")
      if(CMAKE_INSTALL_DO_STRIP)
        execute_process(COMMAND "/usr/bin/strip" "${file}")
      endif()
    endif()
  endforeach()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/liborocos-kdl.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/liborocos-kdl.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/liborocos-kdl.so"
         RPATH "$ORIGIN/../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/root/ros2_robot_arm/orocos_kdl/build/src/liborocos-kdl.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/liborocos-kdl.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/liborocos-kdl.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/liborocos-kdl.so"
         OLD_RPATH "::::::::::::::"
         NEW_RPATH "$ORIGIN/../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/liborocos-kdl.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/kdl" TYPE FILE FILES
    "/root/ros2_robot_arm/orocos_kdl/src/articulatedbodyinertia.hpp"
    "/root/ros2_robot_arm/orocos_kdl/src/chain.hpp"
    "/root/ros2_robot_arm/orocos_kdl/src/chaindynparam.hpp"
    "/root/ros2_robot_arm/orocos_kdl/src/chainfksolver.hpp"
    "/root/ros2_robot_arm/orocos_kdl/src/chainfksolverpos_recursive.hpp"
    "/root/ros2_robot_arm/orocos_kdl/src/chainfksolvervel_recursive.hpp"
    "/root/ros2_robot_arm/orocos_kdl/src/chainidsolver.hpp"
    "/root/ros2_robot_arm/orocos_kdl/src/chainidsolver_recursive_newton_euler.hpp"
    "/root/ros2_robot_arm/orocos_kdl/src/chainidsolver_vereshchagin.hpp"
    "/root/ros2_robot_arm/orocos_kdl/src/chainiksolver.hpp"
    "/root/ros2_robot_arm/orocos_kdl/src/chainiksolverpos_lma.hpp"
    "/root/ros2_robot_arm/orocos_kdl/src/chainiksolverpos_nr.hpp"
    "/root/ros2_robot_arm/orocos_kdl/src/chainiksolverpos_nr_jl.hpp"
    "/root/ros2_robot_arm/orocos_kdl/src/chainiksolvervel_pinv.hpp"
    "/root/ros2_robot_arm/orocos_kdl/src/chainiksolvervel_pinv_givens.hpp"
    "/root/ros2_robot_arm/orocos_kdl/src/chainiksolvervel_pinv_nso.hpp"
    "/root/ros2_robot_arm/orocos_kdl/src/chainiksolvervel_wdls.hpp"
    "/root/ros2_robot_arm/orocos_kdl/src/chainjnttojacdotsolver.hpp"
    "/root/ros2_robot_arm/orocos_kdl/src/chainjnttojacsolver.hpp"
    "/root/ros2_robot_arm/orocos_kdl/src/frameacc.hpp"
    "/root/ros2_robot_arm/orocos_kdl/src/frameacc.inl"
    "/root/ros2_robot_arm/orocos_kdl/src/frameacc_io.hpp"
    "/root/ros2_robot_arm/orocos_kdl/src/frames.hpp"
    "/root/ros2_robot_arm/orocos_kdl/src/frames.inl"
    "/root/ros2_robot_arm/orocos_kdl/src/frames_io.hpp"
    "/root/ros2_robot_arm/orocos_kdl/src/framevel.hpp"
    "/root/ros2_robot_arm/orocos_kdl/src/framevel.inl"
    "/root/ros2_robot_arm/orocos_kdl/src/framevel_io.hpp"
    "/root/ros2_robot_arm/orocos_kdl/src/jacobian.hpp"
    "/root/ros2_robot_arm/orocos_kdl/src/jntarray.hpp"
    "/root/ros2_robot_arm/orocos_kdl/src/jntarrayacc.hpp"
    "/root/ros2_robot_arm/orocos_kdl/src/jntarrayvel.hpp"
    "/root/ros2_robot_arm/orocos_kdl/src/jntspaceinertiamatrix.hpp"
    "/root/ros2_robot_arm/orocos_kdl/src/joint.hpp"
    "/root/ros2_robot_arm/orocos_kdl/src/kdl.hpp"
    "/root/ros2_robot_arm/orocos_kdl/src/kinfam.hpp"
    "/root/ros2_robot_arm/orocos_kdl/src/kinfam_io.hpp"
    "/root/ros2_robot_arm/orocos_kdl/src/motion.hpp"
    "/root/ros2_robot_arm/orocos_kdl/src/path.hpp"
    "/root/ros2_robot_arm/orocos_kdl/src/path_circle.hpp"
    "/root/ros2_robot_arm/orocos_kdl/src/path_composite.hpp"
    "/root/ros2_robot_arm/orocos_kdl/src/path_cyclic_closed.hpp"
    "/root/ros2_robot_arm/orocos_kdl/src/path_line.hpp"
    "/root/ros2_robot_arm/orocos_kdl/src/path_point.hpp"
    "/root/ros2_robot_arm/orocos_kdl/src/path_roundedcomposite.hpp"
    "/root/ros2_robot_arm/orocos_kdl/src/rigidbodyinertia.hpp"
    "/root/ros2_robot_arm/orocos_kdl/src/rotational_interpolation.hpp"
    "/root/ros2_robot_arm/orocos_kdl/src/rotational_interpolation_sa.hpp"
    "/root/ros2_robot_arm/orocos_kdl/src/rotationalinertia.hpp"
    "/root/ros2_robot_arm/orocos_kdl/src/segment.hpp"
    "/root/ros2_robot_arm/orocos_kdl/src/solveri.hpp"
    "/root/ros2_robot_arm/orocos_kdl/src/stiffness.hpp"
    "/root/ros2_robot_arm/orocos_kdl/src/trajectory.hpp"
    "/root/ros2_robot_arm/orocos_kdl/src/trajectory_composite.hpp"
    "/root/ros2_robot_arm/orocos_kdl/src/trajectory_segment.hpp"
    "/root/ros2_robot_arm/orocos_kdl/src/trajectory_stationary.hpp"
    "/root/ros2_robot_arm/orocos_kdl/src/tree.hpp"
    "/root/ros2_robot_arm/orocos_kdl/src/treefksolver.hpp"
    "/root/ros2_robot_arm/orocos_kdl/src/treefksolverpos_recursive.hpp"
    "/root/ros2_robot_arm/orocos_kdl/src/treeiksolver.hpp"
    "/root/ros2_robot_arm/orocos_kdl/src/treeiksolverpos_nr_jl.hpp"
    "/root/ros2_robot_arm/orocos_kdl/src/treeiksolverpos_online.hpp"
    "/root/ros2_robot_arm/orocos_kdl/src/treeiksolvervel_wdls.hpp"
    "/root/ros2_robot_arm/orocos_kdl/src/treejnttojacsolver.hpp"
    "/root/ros2_robot_arm/orocos_kdl/src/velocityprofile.hpp"
    "/root/ros2_robot_arm/orocos_kdl/src/velocityprofile_dirac.hpp"
    "/root/ros2_robot_arm/orocos_kdl/src/velocityprofile_rect.hpp"
    "/root/ros2_robot_arm/orocos_kdl/src/velocityprofile_spline.hpp"
    "/root/ros2_robot_arm/orocos_kdl/src/velocityprofile_trap.hpp"
    "/root/ros2_robot_arm/orocos_kdl/src/velocityprofile_traphalf.hpp"
    "/root/ros2_robot_arm/orocos_kdl/build/src/config.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/kdl/utilities" TYPE FILE FILES
    "/root/ros2_robot_arm/orocos_kdl/src/utilities/error.h"
    "/root/ros2_robot_arm/orocos_kdl/src/utilities/error_stack.h"
    "/root/ros2_robot_arm/orocos_kdl/src/utilities/kdl-config.h"
    "/root/ros2_robot_arm/orocos_kdl/src/utilities/rall1d.h"
    "/root/ros2_robot_arm/orocos_kdl/src/utilities/rall1d_io.h"
    "/root/ros2_robot_arm/orocos_kdl/src/utilities/rall2d.h"
    "/root/ros2_robot_arm/orocos_kdl/src/utilities/rall2d_io.h"
    "/root/ros2_robot_arm/orocos_kdl/src/utilities/rallNd.h"
    "/root/ros2_robot_arm/orocos_kdl/src/utilities/svd_HH.hpp"
    "/root/ros2_robot_arm/orocos_kdl/src/utilities/svd_eigen_HH.hpp"
    "/root/ros2_robot_arm/orocos_kdl/src/utilities/svd_eigen_Macie.hpp"
    "/root/ros2_robot_arm/orocos_kdl/src/utilities/traits.h"
    "/root/ros2_robot_arm/orocos_kdl/src/utilities/utility.h"
    "/root/ros2_robot_arm/orocos_kdl/src/utilities/utility_io.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "/root/ros2_robot_arm/orocos_kdl/build/src/orocos-kdl.pc")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "/root/ros2_robot_arm/orocos_kdl/build/src/orocos_kdl.pc")
endif()

