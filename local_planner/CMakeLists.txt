cmake_minimum_required(VERSION 2.8.3)
project(local_planner)

find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    tf
    nav_core
    costmap_2d)

find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED)

catkin_package()

set(CMAKE_CXX_FLAGS "-std=c++11")

include_directories(
    include
    ${catkin_INCLUDE_DIRS}
)

add_library( local_planner
    src/local_planner.cpp
)

add_dependencies(local_planner
    ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS}
)

target_link_libraries(local_planner
    ${catkin_LIBRARIES}
)

