
if (0)
    option( "EIGEN_BUILD_BTL" OFF)
    option( "EIGEN_BUILD_PKGCONFIG" OFF)
    option( "EIGEN_BUILD_DOC" OFF)
    option( "EIGEN_BUILD_TESTING" OFF)

    add_subdirectory(eigen)

    set_target_properties(eigen PROPERTIES FOLDER ${external_folder})

else()

    add_library(eigen INTERFACE)

    target_include_directories(eigen INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/eigen)

endif()