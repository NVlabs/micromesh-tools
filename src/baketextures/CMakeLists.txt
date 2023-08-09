project(baketextures VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

set(_src_files
    push_pull.h
    baketextures.cpp
)

if (WIN32)
    # XXXX disable truncation warning
    add_compile_options(/wd4305)
    add_definitions(-DNOMINMAX)
endif()

add_executable(${PROJECT_NAME} ${_src_files})

target_link_libraries(${PROJECT_NAME} 
    umeshtools_core
    bary_core
    bary_utils
    cgltf
    eigen)

install(TARGETS ${PROJECT_NAME} DESTINATION "bin/${PROJECT_NAME}")

