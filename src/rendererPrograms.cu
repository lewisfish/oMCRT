#include <optix_device.h>
#include "gdt/math/vec.h"
#include "random.cuh"
#include "LaunchParams.h"
#include "render.h"
#include <stdio.h>

extern "C" __constant__ RendererLaunchParams optixLaunchParams;

struct perRayData
{
    gdt::vec3f colour;
};


enum { SURFACE_RAY_TYPE=0, RAY_TYPE_COUNT };

    static __forceinline__ __device__
    void *unpackPointer( uint32_t i0, uint32_t i1 )
    {
        const uint64_t uptr = static_cast<uint64_t>( i0 ) << 32 | i1;
        void*           ptr = reinterpret_cast<void*>( uptr ); 
        return ptr;
        }

    static __forceinline__ __device__
    void  packPointer( void* ptr, uint32_t& i0, uint32_t& i1 )
    {
        const uint64_t uptr = reinterpret_cast<uint64_t>( ptr );
        i0 = uptr >> 32;
        i1 = uptr & 0x00000000ffffffff;
    }

    template<typename T>
    static __forceinline__ __device__ T *getPRD()
    { 
        const uint32_t u0 = optixGetPayload_0();
        const uint32_t u1 = optixGetPayload_1();
        return reinterpret_cast<T*>( unpackPointer( u0, u1 ) );
    }


extern "C" __global__ void __closesthit__render()
{
    perRayData &prd = *(perRayData*)getPRD<perRayData>();
    prd.colour = gdt::vec3f(1.0f, 0.f, 0.f);
}


extern "C" __global__ void __miss__render()
{
    perRayData &prd = *(perRayData*)getPRD<perRayData>();
    prd.colour = gdt::vec3f(1.f, 1.f, 1.f);
}

extern "C" __global__ void __anyhit__render()
{
};

extern "C" __global__ void __raygen__camera()
{

    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;

    const auto &camera = optixLaunchParams.camera;

    perRayData PRD = perRayData();
    PRD.colour = gdt::vec3f(0.f);

    // for storing the payload
    uint32_t u0, u1;
    packPointer(&PRD, u0, u1);

    const gdt::vec2f screen(gdt::vec2f(ix+.5f, iy+.5f) / 
         gdt::vec2f(optixLaunchParams.frame.nsize.x, optixLaunchParams.frame.nsize.y));

    gdt::vec3f rayDir = gdt::normalize(camera.direction + 
                                       (screen.x - 0.5f) * camera.horizontal
                                      +(screen.y - 0.5f) * camera.vertical);

    optixTrace(optixLaunchParams.traversable,
                camera.position,
                rayDir,
                0.f,
                1e20f,
                0.0f,
                OptixVisibilityMask(255),
                OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                SURFACE_RAY_TYPE,
                RAY_TYPE_COUNT,
                SURFACE_RAY_TYPE,
                u0, u1);

    const int r = int(255.99f*PRD.colour.x);
    const int g = int(255.99f*PRD.colour.y);
    const int b = int(255.99f*PRD.colour.z);

    // convert to 32-bit rgba value (we explicitly set alpha to 0xff
    // to make stb_image_write happy ...
    const uint32_t rgba = 0xff000000
      | (r<<0) | (g<<8) | (b<<16);

    // and write to frame buffer ...
    const uint32_t fbIndex = ix+iy*optixLaunchParams.frame.nsize.x;
    optixLaunchParams.frame.frameBuffer[fbIndex] = rgba;
   
}