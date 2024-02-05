#pragma once
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>
#include "gdt/math/vec.h"

struct LaunchParams
{
    struct {
      float *fluenceBuffer;
      int *nscattBuffer;
      gdt::vec3i size;
      gdt::vec2i nsize;
    } frame;
    OptixTraversableHandle traversable;
};
