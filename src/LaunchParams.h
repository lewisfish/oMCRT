#pragma once
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>
#include "gdt/math/vec.h"

struct LaunchParams
{
    struct {
      float *fluenceBuffer;
      gdt::vec3i size;
    } frame;
    OptixTraversableHandle traversable;
};
