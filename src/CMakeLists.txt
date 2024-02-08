find_package(OpenGL REQUIRED)

cuda_compile_and_embed(embedded_ptx_code devicePrograms.cu)

cuda_add_executable(optix_mcrt
    ${embedded_ptx_code}
    io.cpp
    model.h
    model.cpp
    optixclass.cpp
    SampleSimulation.cpp
    main.cpp)

target_link_libraries(optix_mcrt
    gdt
    ${optix_LIBRARY}
    -ldl
    glfWindow
    glfw
    ${OPENGL_gl_LIBRARY}
    cuda)