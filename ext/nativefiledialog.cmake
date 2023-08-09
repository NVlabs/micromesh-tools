
set(_src_files
    ${nativefiledialog_SOURCE_DIR}/src/common.h
    ${nativefiledialog_SOURCE_DIR}/src/nfd_common.c
    ${nativefiledialog_SOURCE_DIR}/src/nfd_common.h
)

if (WIN32)
    list(APPEND _src_files "${nativefiledialog_SOURCE_DIR}/src/nfd_win.cpp")
elseif (UNIX AND NOT APPLE)
    list(APPEND _src_files "${nativefiledialog_SOURCE_DIR}/src/nfd_gtk.c")
endif()

add_library(nativefiledialog STATIC ${_src_files})

target_include_directories(nativefiledialog PUBLIC ${nativefiledialog_SOURCE_DIR}/src/include)

if (UNIX AND NOT APPLE)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(GTK3 REQUIRED gtk+-3.0)

    target_include_directories(nativefiledialog PRIVATE ${GTK3_INCLUDE_DIRS})
    target_link_directories(nativefiledialog PRIVATE ${GTK3_LIBRARY_DIRS})

    target_compile_options(nativefiledialog PRIVATE ${GTK3_CFLAGS_OTHER})

    target_link_libraries(nativefiledialog PRIVATE ${GTK3_LIBRARIES})
endif()

