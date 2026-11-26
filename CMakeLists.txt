cmake_minimum_required(VERSION 3.20)
project(vgeo VERSION 0.3.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# vgeo_stream: the cluster-LOD builder and runtime the Blender add-on loads (addons/vgeo/bin).
# No Vulkan SDK or Python headers needed. VGEO_STREAM_ONLY is accepted for older build commands.
option(VGEO_STREAM_ONLY "Kept for compatibility: only vgeo_stream is built" ON)
add_subdirectory(src/stream)
