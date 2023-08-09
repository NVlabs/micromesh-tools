include(FetchContent)

set(external_folder "External Libraries")

#
if (NOT TARGET bary_core)

    option(BARY_CORE_BUILD_UTILS "Build the utility library for bary_core" ON)

    if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/bary_core/.git")
        FetchContent_Declare(bary_core GIT_REPOSITORY https://github.com/NVIDIAGameWorks/Displacement-MicroMap-BaryFile.git GIT_TAG origin/main)
        FetchContent_MakeAvailable(bary_core)
    else()
        add_subdirectory(bary_core EXCLUDE_FROM_ALL)
    endif()

    set_target_properties(bary_core PROPERTIES FOLDER ${external_folder})

endif()

#
if (NOT TARGET cgltf)

    if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cgltf/.git")
        FetchContent_Declare(cgltf GIT_REPOSITORY https://github.com/pixeljetstream/cgltf.git GIT_TAG origin/nvidia/micromap)
        FetchContent_MakeAvailable(cgltf)
        FetchContent_GetProperties(cgltf SOURCE_DIR cgltf_SOURCE_DIR)
    else()
        set(cgltf_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cgltf")
    endif()

    include(cgltf.cmake EXCLUDE_FROM_ALL)

endif()

#
if (NOT TARGET eigen)

    OPTION(EIGEN_BUILD_DOC "" OFF)
    OPTION(EIGEN_BUILD_TESTING "" OFF)

    if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/eigen/.git" )
        FetchContent_Declare(eigen GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git)
        FetchContent_MakeAvailable(eigen)
    else()
        include(eigen.cmake)
        set_target_properties(eigen PROPERTIES FOLDER ${external_folder})
    endif()
endif()

#
if (NOT TARGET glad)

    include(glad.cmake)
    set_target_properties(glad PROPERTIES FOLDER ${external_folder})

endif()


#
if (NOT TARGET glfw)

    option(GLFW_BUILD_EXAMPLES "" OFF)
    option(GLFW_BUILD_TESTS "" OFF)
    option(GLFW_BUILD_DOCS "" OFF)
    option(GLFW_INSTALL "" OFF)
    
    if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/glfw/.git" )
        FetchContent_Declare(glfw GIT_REPOSITORY https://github.com/glfw/glfw.git)
        FetchContent_MakeAvailable(glfw)
    else()
        add_subdirectory(glfw)
    endif()

    set_target_properties(glfw PROPERTIES FOLDER ${external_folder}/GLFW3)
    set_target_properties(update_mappings PROPERTIES FOLDER ${external_folder}/GLFW3)

endif()

#
if (NOT TARGET imgui)
    
    if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/imgui/.git" )
        FetchContent_Declare(imgui GIT_REPOSITORY https://github.com/ocornut/imgui.git GIT_TAG "v1.86")
        FetchContent_MakeAvailable(imgui)
        FetchContent_GetProperties(imgui SOURCE_DIR imgui_SOURCE_DIR)
    else()
        set(imgui_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/imgui")
    endif()

    include(imgui.cmake)

    set_target_properties(imgui PROPERTIES FOLDER ${external_folder})

endif()


#
if (NOT TARGET nativefiledialog)

    if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/nativefiledialog/.git" )
        FetchContent_Declare(nativefiledialog GIT_REPOSITORY https://github.com/mlabbe/nativefiledialog.git)
        FetchContent_MakeAvailable(nativefiledialog)
        FetchContent_GetProperties(nativefiledialog SOURCE_DIR nativefiledialog_SOURCE_DIR)
    else()
        set(nativefiledialog_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/nativefiledialog")
    endif()

    include(nativefiledialog.cmake)

    set_target_properties(nativefiledialog PROPERTIES FOLDER ${external_folder})
    
endif()

