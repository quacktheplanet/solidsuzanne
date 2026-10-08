# VGEO stream: cluster-LOD builder + streaming runtime as one C-API shared library.
# Loaded by the Blender add-on through ctypes, so it has no Python dependency.

include(FetchContent)
if(VGEO_MESHOPTIMIZER_DIR)
    set(meshoptimizer_SOURCE_DIR ${VGEO_MESHOPTIMIZER_DIR})
else()
    FetchContent_Declare(
        meshoptimizer
        GIT_REPOSITORY https://github.com/zeux/meshoptimizer.git
        GIT_TAG        9e1f07b159d3cb777f1c67ed31fc11fd117986f4
    )
    FetchContent_GetProperties(meshoptimizer)
    if(NOT meshoptimizer_POPULATED)
        FetchContent_Populate(meshoptimizer)
    endif()
endif()

file(GLOB MESHOPT_SOURCES ${meshoptimizer_SOURCE_DIR}/src/*.cpp)

add_library(vgeo_stream SHARED
    build.cpp
    runtime.cpp
    clusterlod_impl.cpp
    ${MESHOPT_SOURCES}
)
target_include_directories(vgeo_stream PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${meshoptimizer_SOURCE_DIR}/src
)
set_target_properties(vgeo_stream PROPERTIES PREFIX "" CXX_VISIBILITY_PRESET hidden)
# the LOD build (clusterlod_mt.h) uses std::thread
find_package(Threads REQUIRED)
target_link_libraries(vgeo_stream PRIVATE Threads::Threads)
if(MSVC)
    target_compile_options(vgeo_stream PRIVATE /O2 /W3 /wd4996)
    # static CRT: the DLL must load inside Blender with no redistributable installed
    set_property(TARGET vgeo_stream PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded")
endif()

# copy next to the add-on for development
add_custom_command(TARGET vgeo_stream POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/addons/vgeo/bin
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:vgeo_stream> ${CMAKE_SOURCE_DIR}/addons/vgeo/bin/
)
