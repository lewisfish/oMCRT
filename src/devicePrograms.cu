#include <optix_device.h>
#include "gdt/math/vec.h"
#include "random.cuh"
#include "LaunchParams.h"
#include <stdio.h>

#define THRESHOLD 0.01
#define CHANCE    0.1

extern "C" __constant__ LaunchParams optixLaunchParams;

struct perRayData
{
    float t;
    float weight;
    bool alive;
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


extern "C" __global__ void __closesthit__radiance()
{
    const float t = optixGetRayTmax();
    perRayData &prd = *(perRayData*)getPRD<perRayData>();
    prd.t = t;
}
extern "C" __global__ void __anyhit__radiance()
{}
extern "C" __global__ void __miss__radiance()
{
    perRayData &prd = *(perRayData*)getPRD<perRayData>();
    prd.alive = false;
}

extern "C" __device__ gdt::vec3f emit(uint &seed)
{
    float uxx, uyy, uzz;
    float phi = 2.f * M_PI * rnd(seed);
    float cost = 2.f * rnd(seed) - 1.f;
    float sint = sqrtf(1.f - cost * cost);
    uxx = sint * cosf(phi);
    uyy = sint * sinf(phi); 
    uzz = cost;
    return gdt::vec3f(uxx, uyy, uzz);
}

extern "C" __global__ void __raygen__simulate()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    uint seed = tea<64>(dim.x * idx.y + idx.x, 0);

    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;

    const float mus = 10.0f;
    const float mua = 0.01f;

    // per ray data
    perRayData PRD = perRayData();
    PRD.alive = true;
    PRD.weight = 1.0f;
    PRD.t = -1.0f;

    // for storing the payload
    uint32_t u0, u1;
    packPointer(&PRD, u0, u1);

    gdt::vec3f rayPos = gdt::vec3f(0.f);
    gdt::vec3f rayDir = emit(seed);
    float absorb;
    for(;;)
    {
        float L = -log(rnd(seed)) / mus;

        optixTrace(optixLaunchParams.traversable,
        rayPos,
        rayDir,
        0.f,      //tmin
        1e20f,    //tmax
        0.0f,     // rayTime
        OptixVisibilityMask( 255 ),
        OPTIX_RAY_FLAG_DISABLE_ANYHIT,//OPTIX_RAY_FLAG_NONE,
        SURFACE_RAY_TYPE,             // SBT offset
        RAY_TYPE_COUNT,               // SBT stride
        SURFACE_RAY_TYPE,             // missSBTIndex 
        u0, u1 );                     //payload?

        if(L > PRD.t || !PRD.alive)break;

        rayPos += L * rayDir;
        rayDir = emit(seed);
        absorb = PRD.weight*(1.f - expf(-mua * L));
        PRD.weight -= absorb;
        int celli = max(min((int)floorf(100 * (rayPos.x + 1.5f) / (3.0f)), 100), 0);
        int cellj = max(min((int)floorf(100 * (rayPos.y + 1.5f) / (3.0f)), 100), 0);
        int cellk = max(min((int)floorf(100 * (rayPos.z + 1.5f) / (3.0f)), 100), 0);
        uint32_t fbIndex = celli*100*100 + cellj*100+cellk;
        atomicAdd(&optixLaunchParams.frame.fluenceBuffer[fbIndex], absorb);
        if(PRD.weight < THRESHOLD) 
        {
            if(rnd(seed) <= CHANCE)
            {
                PRD.weight /= CHANCE;
            }
        } else 
        {
            PRD.alive = false;
            break;
        }
    }
}