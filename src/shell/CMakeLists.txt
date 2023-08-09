project(shell VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

find_package(OpenGL REQUIRED)

set(_src_files
    ${CMAKE_CURRENT_LIST_DIR}/../meshreduce/aabb.h
    ${CMAKE_CURRENT_LIST_DIR}/../meshreduce/adjacency.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../meshreduce/adjacency.h
    ${CMAKE_CURRENT_LIST_DIR}/../meshreduce/bvh.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../meshreduce/bvh.h
    ${CMAKE_CURRENT_LIST_DIR}/../meshreduce/camera.h
    ${CMAKE_CURRENT_LIST_DIR}/../meshreduce/clean.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../meshreduce/clean.h
    ${CMAKE_CURRENT_LIST_DIR}/../meshreduce/color.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../meshreduce/color.h
    ${CMAKE_CURRENT_LIST_DIR}/../meshreduce/curvature.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../meshreduce/curvature.h
    ${CMAKE_CURRENT_LIST_DIR}/../meshreduce/decimation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../meshreduce/decimation.h
    ${CMAKE_CURRENT_LIST_DIR}/../meshreduce/direction_field.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../meshreduce/direction_field.h
    ${CMAKE_CURRENT_LIST_DIR}/../meshreduce/flip.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../meshreduce/flip.h
    ${CMAKE_CURRENT_LIST_DIR}/../meshreduce/geodesic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../meshreduce/geodesic.h
    ${CMAKE_CURRENT_LIST_DIR}/../meshreduce/gl_utils.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../meshreduce/gl_utils.h
    ${CMAKE_CURRENT_LIST_DIR}/../meshreduce/gui.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../meshreduce/gui.h
    ${CMAKE_CURRENT_LIST_DIR}/../meshreduce/gui_callback.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../meshreduce/gui_controls.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../meshreduce/gui_controls.h
    ${CMAKE_CURRENT_LIST_DIR}/../meshreduce/gui_widgets.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../meshreduce/intersection.h
    ${CMAKE_CURRENT_LIST_DIR}/../meshreduce/local_operations.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../meshreduce/local_operations.h
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
    ${CMAKE_CURRENT_LIST_DIR}/../meshreduce/quadric.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../meshreduce/quadric.h
    ${CMAKE_CURRENT_LIST_DIR}/../meshreduce/quality.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../meshreduce/quality.h
    ${CMAKE_CURRENT_LIST_DIR}/../meshreduce/shaders
    ${CMAKE_CURRENT_LIST_DIR}/../meshreduce/smooth.h
    ${CMAKE_CURRENT_LIST_DIR}/../meshreduce/smooth.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../meshreduce/session.h
    ${CMAKE_CURRENT_LIST_DIR}/../meshreduce/session.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../meshreduce/space.h
    ${CMAKE_CURRENT_LIST_DIR}/../meshreduce/tangent.h
    ${CMAKE_CURRENT_LIST_DIR}/../meshreduce/utils.h
    shell.cpp
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
    glad
    glfw
    imgui
    nativefiledialog
    version_file)

target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../meshreduce)

install(TARGETS ${PROJECT_NAME} DESTINATION "bin/${PROJECT_NAME}")
install(FILES ${_shader_files} DESTINATION "bin/${PROJECT_NAME}/shaders" )

if (${PACKAGE_SOURCE})
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} DESTINATION "src/")
endif()

