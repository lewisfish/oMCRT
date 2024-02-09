#pragma once
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>
#include "gdt/math/vec.h"

struct RendererLaunchParams
{
    struct {
        uint32_t *frameBuffer;
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

struct SimulationLaunchParams
{
    struct {
        float *fluenceBuffer;
        int   *nscattBuffer;
        gdt::vec3i size;
        gdt::vec2i nsize;
    } frame;
    OptixTraversableHandle traversable;
    struct {
        float mus[3];
    }optProps;
};