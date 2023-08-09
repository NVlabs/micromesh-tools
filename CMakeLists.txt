cmake_minimum_required (VERSION 3.20)

project(umeshtools VERSION 1.1.0)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

#
# Dependencies
#

add_subdirectory("ext")

#
# Tools
#

add_subdirectory("src/core")
add_subdirectory("src/gui_utils")

add_subdirectory("src/meshreduce")
add_subdirectory("src/getattribs")
add_subdirectory("src/uvoptimize")
add_subdirectory("src/meshexplode")
add_subdirectory("src/baketextures")
add_subdirectory("src/view")

