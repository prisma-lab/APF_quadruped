cmake_minimum_required(VERSION 3.0.2)
project(dogbot_controller)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages

find_package(iDynTree REQUIRED)
find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  roscpp
  sensor_msgs
  std_msgs
  tf
  roslib
  tf_conversions
ifopt
  
)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)
find_package( ifopt REQUIRED )
include_directories( include )
include_directories( /usr/include/eigen3 )

## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## To declare and build messages, services or actions from within this
## package, follow these steps:
## * Let MSG_DEP_SET be the set of packages whose message types you use in
##   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
## * In the file package.xml:
##   * add a build_depend tag for "message_generation"
##   * add a build_depend and a exec_depend tag for each package in MSG_DEP_SET
##   * If MSG_DEP_SET isn't empty the following dependency has been pulled in
##     but can be declared for certainty nonetheless:
##     * add a exec_depend tag for "message_runtime"
## * In this file (CMakeLists.txt):
##   * add "message_generation" and every package in MSG_DEP_SET to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * add "message_runtime" and every package in MSG_DEP_SET to
##     catkin_package(CATKIN_DEPENDS ...)
##   * uncomment the add_*_files sections below as needed
##     and list every .msg/.srv/.action file to be processed
##   * uncomment the generate_messages entry below
##   * add every package in MSG_DEP_SET to generate_messages(DEPENDENCIES ...)

## Generate messages in the 'msg' folder
# add_message_files(
#   FILES
#   Message1.msg
#   Message2.msg
# )

## Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
# generate_messages(
#   DEPENDENCIES
#   geometry_msgs#   sensor_msgs#   std_msgs
# )

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

## To declare and build dynamic reconfigure parameters within this
## package, follow these steps:
## * In the file package.xml:
##   * add a build_depend and a exec_depend tag for "dynamic_reconfigure"
## * In this file (CMakeLists.txt):
##   * add "dynamic_reconfigure" to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * uncomment the "generate_dynamic_reconfigure_options" section below
##     and list every .cfg file to be processed

## Generate dynamic reconfigure parameters in the 'cfg' folder
# generate_dynamic_reconfigure_options(
#   cfg/DynReconf1.cfg
#   cfg/DynReconf2.cfg
# )

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES popt
#  CATKIN_DEPENDS geometry_msgs roscpp sensor_msgs std_msgs
#  DEPENDS system_lib
)

###########
## Build ##
###########




## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

## Declare a C++ library
# add_library(${PROJECT_NAME}
#   src/${PROJECT_NAME}/popt.cpp
# )

## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
# add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide


set(qpSWIFT_src
    src/qpSWIFT/amd_1.c
    src/qpSWIFT/amd_aat.c
    src/qpSWIFT/amd_defaults.c
    src/qpSWIFT/amd_global.c
    src/qpSWIFT/amd_order.c
    src/qpSWIFT/amd_post_tree.c
    src/qpSWIFT/amd_valid.c
    src/qpSWIFT/amd_2.c
    src/qpSWIFT/amd_control.c
    src/qpSWIFT/amd_dump.c
    src/qpSWIFT/amd_info.c
    src/qpSWIFT/amd_postorder.c
    src/qpSWIFT/amd_preprocess.c
    src/qpSWIFT/ldl.c
    src/qpSWIFT/timer.c
    src/qpSWIFT/Auxilary.c
    src/qpSWIFT/qpSWIFT.c
)

SET( ALGLIB_SRC 
  src/alglib/alglibinternal.cpp
  src/alglib/alglibmisc.cpp
  src/alglib/ap.cpp
  src/alglib/linalg.cpp
  src/alglib/optimization.cpp
  src/alglib/solvers.cpp
)

SET ( TOWR_SRC 
  src/towr/nlp_formulation.cc
  src/towr/parameters.cc
  src/towr/nodes_variables.cc
  src/towr/nodes_variables_all.cc
  src/towr/nodes_variables_phase_based.cc
  src/towr/phase_durations.cc
  src/towr/robot_model.cc
  src/towr/dynamic_model.cc
  src/towr/single_rigid_body_dynamics.cc
  src/towr/time_discretization_constraint.cc
  src/towr/base_motion_constraint.cc
  src/towr/terrain_constraint.cc
  src/towr/swing_constraint.cc
  src/towr/force_constraint.cc
  src/towr/total_duration_constraint.cc
  src/towr/dynamic_constraint.cc
  src/towr/range_of_motion_constraint.cc
  src/towr/spline_acc_constraint.cc
  src/towr/linear_constraint.cc
  src/towr/node_cost.cc
  src/towr/soft_constraint.cc
  src/towr/gait_generator.cc
  src/towr/monoped_gait_generator.cc
  src/towr/biped_gait_generator.cc
  src/towr/quadruped_gait_generator.cc
  src/towr/height_map_examples.cc
  src/towr/height_map.cc
  src/towr/state.cc
  src/towr/polynomial.cc
  src/towr/spline.cc
  src/towr/phase_spline.cc
  src/towr/node_spline.cc
  src/towr/nodes_observer.cc
  src/towr/spline_holder.cc
  src/towr/euler_converter.cc
  src/towr/phase_durations_observer.cc
)


add_library( lopt src/lopt.cpp ${ALGLIB_SRC} ${qpSWIFT_src} )
target_link_libraries( lopt ${catkin_LIBRARIES}  )

add_library( topt src/topt.cpp ${TOWR_SRC} ) 
target_link_libraries( topt ${catkin_LIBRARIES})


add_executable( main src/client/main.cpp )
target_link_libraries(  main lopt topt ${catkin_LIBRARIES}  ${iDynTree_LIBRARIES})


add_executable( visualization src/client/tf_pub.cpp )
target_link_libraries(  visualization ${catkin_LIBRARIES} )
