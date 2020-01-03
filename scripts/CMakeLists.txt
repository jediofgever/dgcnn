cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(read_pcd)

find_package(PCL 1.9 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (read_pcd read_pcd.cpp)
target_link_libraries (read_pcd ${PCL_LIBRARIES})
