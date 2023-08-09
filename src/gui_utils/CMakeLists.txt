set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

set(_gui_utils_srcs
    camera.h
    gl_utils.h
    gl_utils.cpp
    gui_controls.h
    gui_controls.cpp
)

add_library(umeshtools_gui_utils STATIC ${_gui_utils_srcs})

target_link_libraries(umeshtools_gui_utils
    eigen
    glad
    glfw
)

target_include_directories(umeshtools_gui_utils PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../core")

target_include_directories(umeshtools_gui_utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
