project(uvmap VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

set(_src_files
    ${CMAKE_CURRENT_LIST_DIR}/../meshreduce/aabb.h
    ${CMAKE_CURRENT_LIST_DIR}/../meshreduce/adjacency.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../meshreduce/adjacency.h
    ${CMAKE_CURRENT_LIST_DIR}/../meshreduce/bvh.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../meshreduce/bvh.h
    ${CMAKE_CURRENT_LIST_DIR}/../meshreduce/intersection.h
    ${CMAKE_CURRENT_LIST_DIR}/../meshreduce/mesh_io.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../meshreduce/mesh_io.h
    ${CMAKE_CURRENT_LIST_DIR}/../meshreduce/mesh_io_bary.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../meshreduce/mesh_io_bary.h
    ${CMAKE_CURRENT_LIST_DIR}/../meshreduce/mesh_io_gltf.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../meshreduce/mesh_io_gltf.h
    ${CMAKE_CURRENT_LIST_DIR}/../meshreduce/mesh_utils.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../meshreduce/mesh_utils.h
    ${CMAKE_CURRENT_LIST_DIR}/../meshreduce/micro.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../meshreduce/micro.h
    ${CMAKE_CURRENT_LIST_DIR}/../meshreduce/quality.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../meshreduce/quality.h
    ${CMAKE_CURRENT_LIST_DIR}/../meshreduce/space.h
    ${CMAKE_CURRENT_LIST_DIR}/../meshreduce/tangent.h
    ${CMAKE_CURRENT_LIST_DIR}/../meshreduce/utils.h
    ${CMAKE_CURRENT_LIST_DIR}/../uvoptimize/arap.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../uvoptimize/arap.h
    ${CMAKE_CURRENT_LIST_DIR}/../uvoptimize/rectangle_packer.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../uvoptimize/rectangle_packer.h
    uvmap.cpp
)

if (WIN32)
    # XXXX disable truncation warning - FIX THE CODE !
    add_compile_options(/wd4305)
    add_definitions(-DNOMINMAX)
endif()

add_executable(${PROJECT_NAME} ${_src_files})

target_link_libraries(${PROJECT_NAME} 
    bary_core
    bary_utils
    cgltf
    eigen
    version_file)

target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../meshreduce ${CMAKE_CURRENT_LIST_DIR}/../uvoptimize)

install(TARGETS ${PROJECT_NAME} DESTINATION "bin/${PROJECT_NAME}")

if (${PACKAGE_SOURCE})
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} DESTINATION "src/")
endif()

