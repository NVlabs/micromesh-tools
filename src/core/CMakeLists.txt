set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

set(_core_srcs
    aabb.h
    adjacency.cpp
    adjacency.h
    ambient_occlusion.cpp
    ambient_occlusion.h
    bvh.cpp
    bvh.h
    clean.cpp
    clean.h
    color.cpp
    color.h
    curvature.cpp
    curvature.h
    direction_field.cpp
    direction_field.h
    flip.cpp
    flip.h
    geodesic.cpp
    geodesic.h
    intersection.h
    local_operations.cpp
    local_operations.h
    mesh_io.cpp
    mesh_io.h
    mesh_io_bary.cpp
    mesh_io_bary.h
    mesh_io_gltf.cpp
    mesh_io_gltf.h
    mesh_utils.cpp
    mesh_utils.h
    micro.cpp
    micro.h
    quadric.cpp
    quadric.h
    quality.cpp
    quality.h
    smooth.cpp
    smooth.h
    space.h
    stb_image.h
    stb_image_write.h
    stb_image_impl.cpp
    tangent.h
    utils.h
    visibility.cpp
    visibility.h
)

add_library(umeshtools_core STATIC ${_core_srcs})

target_link_libraries(umeshtools_core
    bary_core
    bary_utils
    cgltf
    eigen
)

target_include_directories(umeshtools_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

