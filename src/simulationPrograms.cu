#include <optix_device.h>
#include "gdt/math/vec.h"
#include "random.cuh"
#include "LaunchParams.h"
#include "SampleSimulation.h"
#include <stdio.h>

#define THRESHOLD 0.01
#define CHANCE    0.1

extern "C" __constant__ SimulationLaunchParams optixLaunchParams;

struct perRayData
{
    float t;
    float weight;
    int nscatt;
    bool alive;
    float mus;
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
    const trianglemeshSBTdata &hg_data =  *(const trianglemeshSBTdata*)(optixGetSbtDataPointer());
    prd.mus = hg_data.mus;
    
}
extern "C" __global__ void __anyhit__radiance()
{
    perRayData &prd = *(perRayData*)getPRD<perRayData>();
    // prd.count++;
    optixIgnoreIntersection();

}
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


extern "C" __device__ gdt::vec3f scatter(gdt::vec3f &dir, const float &hgg, const float &g2, uint &seed)
{
    float cost;

    if(hgg == 0.0f)
    {
        // isotropic scattering
        cost = 2.f * rnd(seed) - 1.f;
    } else
    {
        // HG scattering
        float temp = (1.0f - g2) / (1.0f - hgg + 2.f*hgg*rnd(seed));
        cost = (1.0f + g2 - temp*temp) / (2.f*hgg);
    }

    float sint = sqrtf(1.f - cost*cost);

    float phi = 2.f * M_PI * rnd(seed);
    float cosp = cosf(phi);
    float sinp;
    if(phi < M_PI)
    {
        sinp = sqrtf(1.f - cosp*cosp);
    } else
    {
        sinp = -sqrtf(1.f - cosp*cosp);
    } 

    float uxx, uyy, uzz;
    if (1.0f - fabs(dir.z) <= 1e-12f) // near perpindicular
    {
        uxx = sint * cosp;
        uyy = sint * sinp;
        uzz = copysignf(cost, dir.z);
    } else
    {
        float temp = sqrtf(1.f - dir.z*dir.z);
        uxx = sint * (dir.x * dir.z * cosp - dir.y * sinp) / temp + dir.x * cost;
        uyy = sint * (dir.y * dir.z * cosp + dir.x * sinp) / temp + dir.y * cost;
        uzz = -1.f*sint * cosp * temp + dir.z * cost;
    }

    return gdt::vec3f(uxx, uyy, uzz);
}

extern "C" __global__ void __raygen__weight()
{
    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;
    const uint3 dim = optixGetLaunchDimensions();
    uint seed = tea<64>(dim.x * iy + ix, 0);


    float mus; //= 10.0f;
    float mua; //= 0.0f;
    float hgg; //= 0.0;
    float g2; //= hgg*hgg;

    // per ray data
    perRayData PRD = perRayData();
    PRD.alive = true;
    PRD.weight = 1.0f;
    PRD.nscatt = 0;
    PRD.t = -1.0f;

    // for storing the payload
    uint32_t u0, u1;
    packPointer(&PRD, u0, u1);

    gdt::vec3f rayPos = gdt::vec3f(0.f);
    gdt::vec3f rayDir = emit(seed);
    float absorb;
    for(;;)
    {

        optixTrace(optixLaunchParams.traversable,
        rayPos,
        rayDir,
        0.f,      // tmin
        1e20f,    // tmax
        0.0f,     // rayTime
        OptixVisibilityMask( 255 ),
        OPTIX_RAY_FLAG_DISABLE_ANYHIT,// OPTIX_RAY_FLAG_NONE,
        SURFACE_RAY_TYPE,             // SBT offset
        RAY_TYPE_COUNT,               // SBT stride
        SURFACE_RAY_TYPE,             // missSBTIndex 
        u0, u1 );                     // payload

        // mus = PRD.mat.mus;
        // mua = PRD.mat.mua;
        // hgg = PRD.mat.hgg;
        // g2 = PRD.mat.g2;


        float L = -log(rnd(seed)) / mus;
        if(L > PRD.t || !PRD.alive)break;

        rayPos += L * rayDir;
        rayDir = scatter(rayDir, hgg, g2, seed);
        absorb = PRD.weight*(1.f - expf(-mua * L));
        PRD.weight -= absorb;

        int celli = max(min((int)floorf(100 * (rayPos.x + 1.5f) / (3.0f)), 100), 0);
        int cellj = max(min((int)floorf(100 * (rayPos.y + 1.5f) / (3.0f)), 100), 0);
        int cellk = max(min((int)floorf(100 * (rayPos.z + 1.5f) / (3.0f)), 100), 0);
        uint32_t fbIndex = celli*100*100 + cellj*100+cellk;
        atomicAdd(&optixLaunchParams.frame.fluenceBuffer[fbIndex], absorb);
        PRD.nscatt += 1;
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
    const uint32_t nsbIndex = ix+iy*optixLaunchParams.frame.nsize.x;
    optixLaunchParams.frame.nscattBuffer[nsbIndex] = PRD.nscatt;
}

extern "C" __global__ void __raygen__simulate()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    uint seed = tea<64>(dim.x * idx.y + idx.x, 0);

    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;

    float mus; //= 10.0f;
    // float mua; //= 0.0f;
    // float hgg; //= 0.0;
    // float g2; //= hgg*hgg;

    // per ray data
    perRayData PRD = perRayData();
    PRD.alive = true;
    PRD.weight = 1.0f;
    PRD.nscatt = 0;
    PRD.t = -1.0f;

    // for storing the payload
    uint32_t u0, u1;
    packPointer(&PRD, u0, u1);

    gdt::vec3f rayPos = gdt::vec3f(0.f);
    gdt::vec3f rayDir = emit(seed);

    for(;;)
    {
        optixTrace(optixLaunchParams.traversable,
        rayPos,
        rayDir,
        0.f,      // tmin
        1e20f,    // tmax
        0.0f,     // rayTime
        OptixVisibilityMask( 255 ),
        OPTIX_RAY_FLAG_DISABLE_ANYHIT,// OPTIX_RAY_FLAG_NONE,
        SURFACE_RAY_TYPE,             // SBT offset
        RAY_TYPE_COUNT,               // SBT stride
        SURFACE_RAY_TYPE,             // missSBTIndex 
        u0, u1 );                     // payload

        mus = PRD.mus;
        // mua = PRD.mua;
        // hgg = PRD.hgg;
        // g2 = PRD.g2;

        float L = -log(rnd(seed)) / mus;
        if(L > PRD.t || !PRD.alive)break;

        rayPos += L * rayDir;
        rayDir = emit(seed);
        PRD.nscatt += 1;
    }
    const uint32_t nsbIndex = ix+iy*optixLaunchParams.frame.nsize.x;
    optixLaunchParams.frame.nscattBuffer[nsbIndex] = PRD.nscatt;
}