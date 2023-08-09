
set(_src_files
    glad/glad/glad.c
    glad/glad/glad.h
    glad/glad/khrplatform.h
)

add_library(glad STATIC ${_src_files})

target_include_directories(glad INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/glad)
