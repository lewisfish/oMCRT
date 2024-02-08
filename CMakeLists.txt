project(OptiX_MCRT)

cmake_minimum_required(VERSION 2.8)

SET(CMAKE_CXX_FLAGS_PROFILE ...)
SET(CMAKE_EXE_LINKER_FLAGS_PROFILE ...)

include(cmake/configure_optix.cmake)

if(NOT SET_UP_CONFIGURATIONS_DONE)
    set(SET_UP_CONFIGURATIONS_DONE 1)

    # No reason to set CMAKE_CONFIGURATION_TYPES if it's not a multiconfig generator
    # Also no reason mess with CMAKE_BUILD_TYPE if it's a multiconfig generator.
    if(CMAKE_CONFIGURATION_TYPES) # multiconfig generator?
        set(CMAKE_CONFIGURATION_TYPES "Debug;Release;Profile" CACHE STRING "" FORCE) 
    else()
        if(NOT CMAKE_BUILD_TYPE)
#            message("Defaulting to release build.")
            set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
        endif()
        set_property(CACHE CMAKE_BUILD_TYPE PROPERTY HELPSTRING "Choose the type of build")
        # set the valid options for cmake-gui drop-down list
        set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug;Release;Profile")
    endif()
endif()
mark_as_advanced(CUDA_SDK_ROOT_DIR)

SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

set(gdt_dir ${PROJECT_SOURCE_DIR}/ext/gdt)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${gdt_dir}/cmake/")
include(${gdt_dir}/cmake/configure_build_type.cmake)
include(${gdt_dir}/cmake/configure_optix.cmake)

include_directories(${gdt_dir})
add_subdirectory(${gdt_dir} EXCLUDE_FROM_ALL)

set(OpenGL_GL_PREFERENCE LEGACY)
if (WIN32)
#  set(glfw_dir ${PROJECT_SOURCE_DIR}/submodules/glfw/)
  set(glfw_dir ${PROJECT_SOURCE_DIR}/ext/glfw/)
  include_directories(${glfw_dir}/include)
  add_subdirectory(${glfw_dir} EXCLUDE_FROM_ALL)
else()
  find_package(glfw3 REQUIRED)
endif()
include_directories(ext)
add_subdirectory(ext/glfWindow EXCLUDE_FROM_ALL)


add_subdirectory(src)

set(PASSED_FIRST_CONFIGURE ON CACHE INTERNAL "Already Configured once?")