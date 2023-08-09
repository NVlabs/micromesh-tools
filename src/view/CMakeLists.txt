project(view VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

find_package(OpenGL REQUIRED)

set(_src_files
    view.cpp
    gui_view.cpp
    gui_view.h
    gui_view_callback.cpp
    gui_view_widgets.cpp

)

if (WIN32)
    # XXXX disable truncation warning - FIX THE CODE !
    add_compile_options(/wd4305)
    add_definitions(-DNOMINMAX)
endif()

add_executable(${PROJECT_NAME} ${_src_files})

target_link_libraries(${PROJECT_NAME} 
    umeshtools_core
    umeshtools_gui_utils
    bary_core
    bary_utils
    cgltf
    eigen
    glad
    glfw
    imgui
    nativefiledialog)

install(TARGETS ${PROJECT_NAME} DESTINATION "bin/${PROJECT_NAME}")

