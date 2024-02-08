#pragma once
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>
#include "gdt/math/vec.h"

struct trianglemeshSBTdata{
    float mus;
    float mua;
    float hgg;
    float g2;
    float albedo;
    float kappa;
    float n;
};

struct LaunchParams
{
    struct {
        float *fluenceBuffer;
        uint32_t *frameBuffer;
        int *nscattBuffer;
        gdt::vec3i size;
        gdt::vec2i nsize;
    } frame;

    struct {
      gdt::vec3f position;
      gdt::vec3f direction;
      gdt::vec3f horizontal;
      gdt::vec3f vertical;
    } camera;

    OptixTraversableHandle traversable;
};
