# Copyright (c) 2013-2022, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# FindCUDA.cmake is deprecated since CMake 3.10.
# Use FindCUDAToolkit.cmake added in CMake 3.17 instead.
cmake_minimum_required(VERSION 3.17)

project( oMCRT )
message("\nPROJECT_NAME = " "${PROJECT_NAME}")

find_package(OpenGL REQUIRED)
find_package(GLFW REQUIRED)
find_package(CUDAToolkit 10.0 REQUIRED)

# OptiX SDK 7.x and 8.x versions are searched inside the top-level CMakeLists.txt.
# Make the build work with all currently released OptiX SDK 7.x and 8.x versions.
if(OptiX80_FOUND)
  set(OPTIX_INCLUDE_DIR "${OPTIX80_INCLUDE_DIR}")
elseif(OptiX77_FOUND)
  set(OPTIX_INCLUDE_DIR "${OPTIX77_INCLUDE_DIR}")
elseif(OptiX76_FOUND)
  set(OPTIX_INCLUDE_DIR "${OPTIX76_INCLUDE_DIR}")
elseif(OptiX75_FOUND)
  set(OPTIX_INCLUDE_DIR "${OPTIX75_INCLUDE_DIR}")
elseif(OptiX74_FOUND)
  set(OPTIX_INCLUDE_DIR "${OPTIX74_INCLUDE_DIR}")
elseif(OptiX73_FOUND)
  set(OPTIX_INCLUDE_DIR "${OPTIX73_INCLUDE_DIR}")
elseif(OptiX72_FOUND)
  set(OPTIX_INCLUDE_DIR "${OPTIX72_INCLUDE_DIR}")
elseif(OptiX71_FOUND)
  set(OPTIX_INCLUDE_DIR "${OPTIX71_INCLUDE_DIR}")
elseif(OptiX70_FOUND)
  set(OPTIX_INCLUDE_DIR "${OPTIX70_INCLUDE_DIR}")
else()
  message(FATAL_ERROR "No OptiX SDK 7.x found.")
endif()
#message("OPTIX_INCLUDE_DIR = " "${OPTIX_INCLUDE_DIR}")

# OptiX SDK 7.5.0 and CUDA 11.7 added support for a new OptiX IR target, which is a binary intermediate format for the module input.
# The default module build target is PTX.
set(USE_OPTIX_IR FALSE)
set(OPTIX_MODULE_EXTENSION ".ptx")
set(OPTIX_PROGRAM_TARGET "--ptx")

if (OptiX80_FOUND OR OptiX77_FOUND OR OptiX76_FOUND OR OptiX75_FOUND)
  # Define USE_OPTIX_IR and change the target to OptiX IR if the combination of OptiX SDK and CUDA Toolkit versions supports this mode.
  if ((${CUDAToolkit_VERSION_MAJOR} GREATER 11) OR ((${CUDAToolkit_VERSION_MAJOR} EQUAL 11) AND (${CUDAToolkit_VERSION_MINOR} GREATER_EQUAL 7)))
    set(USE_OPTIX_IR TRUE)
    set(OPTIX_MODULE_EXTENSION ".optixir")
    set(OPTIX_PROGRAM_TARGET "--optix-ir")
  endif()
endif()

set( HEADERS
${CMAKE_CURRENT_SOURCE_DIR}/CUDABuffer.h
  ${CMAKE_CURRENT_SOURCE_DIR}/model.h
  ${CMAKE_CURRENT_SOURCE_DIR}/optix_helpers.h
  ${CMAKE_CURRENT_SOURCE_DIR}/SampleSimulation.h
  ${CMAKE_CURRENT_SOURCE_DIR}/LaunchParams.h
  ${CMAKE_CURRENT_SOURCE_DIR}/optixclass.h
  ${CMAKE_CURRENT_SOURCE_DIR}/render.h
  ${CMAKE_CURRENT_SOURCE_DIR}/window.h
)

set( SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/io.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/model.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/optixclass.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/render.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/SampleSimulation.cpp
)

# Prefix the shaders with the full path name to allow stepping through errors with F8.
set( SHADERS
  ${CMAKE_CURRENT_SOURCE_DIR}/rendererPrograms.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/simulationPrograms.cu
  )

  # set( SHADERS_HEADERS
  # ${LOCAL_3RDPARTY}/gdt/gdt.h
  # )


# When using OptiX SDK 7.5.0 and CUDA 11.7 or higher, the modules can either be built from OptiX IR input or from PTX input.
# OPTIX_PROGRAM_TARGET and OPTIX_MODULE_EXTENSION switch the NVCC compilation between the two options.
NVCUDA_COMPILE_MODULE(
  SOURCES ${SHADERS}
  DEPENDENCIES
  TARGET_PATH "${MODULE_TARGET_DIR}/oMCRT"
  EXTENSION "${OPTIX_MODULE_EXTENSION}"
  GENERATED_FILES PROGRAM_MODULES
  NVCC_OPTIONS "${OPTIX_PROGRAM_TARGET}" "--machine=64" "--gpu-architecture=compute_61" "--use_fast_math" "--relocatable-device-code=true" "--generate-line-info" "-Wno-deprecated-gpu-targets" "-I${gdt_dir}" "-I${OPTIX_INCLUDE_DIR}" "-I${CMAKE_CURRENT_SOURCE_DIR}/shaders"
)

source_group( "headers"         FILES ${HEADERS} )
source_group( "sources"         FILES ${SOURCES} )
source_group( "shaders"         FILES ${SHADERS} )
source_group( "shaders_headers" FILES ${SHADERS_HEADERS} )
source_group( "prg"             FILES ${PROGRAM_MODULES} )

include_directories(
  "."
  "ext/gdt/math"
  "ext/glfWindow"
  ${GLEW_INCLUDE_DIRS}
  ${GLFW_INCLUDE_DIR}
  ${OPTIX_INCLUDE_DIR}
  ${CUDAToolkit_INCLUDE_DIRS}
)

add_definitions(
  # Disable warnings for file operations fopen etc.
  "-D_CRT_SECURE_NO_WARNINGS"
)

if(USE_OPTIX_IR)
add_definitions(
  # This define switches the OptiX program module filenames to either *.optixir or *.ptx extensions at compile time.
  "-DUSE_OPTIX_IR"
)
endif()

cuda_add_executable( oMCRT
  ${HEADERS}
  ${SOURCES}
  # ${SHADERS_HEADERS}
  # ${SHADERS}
  ${PROGRAM_MODULES}
)

target_link_libraries( oMCRT
  gdt
  OpenGL::GL
  glfWindow
  glfw
  cuda
)

message(STATUS "${OPTIX_INCLUDE_DIR} #################")

if (UNIX)
  target_link_libraries( oMCRT dl)
endif()

set_target_properties( oMCRT PROPERTIES FOLDER "src")