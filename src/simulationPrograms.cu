#include <optix_device.h>
#include "gdt/math/vec.h"
#include "random.cuh"
#include "LaunchParams.h"
#include "SampleSimulation.h"
#include <stdio.h>
#include "optixclass.h"

#define THRESHOLD 0.01f
#define CHANCE    0.1f
#define PI 3.14159265359f
#define EPS 1e-7f

extern "C" __constant__ SimulationLaunchParams optixLaunchParams;

struct perRayData
{
    float t;
    float kappa;
    float albedo;
    int nscatt;
    bool alive;
};

// struct boundaryRayData
// {
//     int count[3];
//     bool alive;
//     float tmax[3];
// };

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

extern "C" __global__ void __closesthit__simulation()
{
    const float t = optixGetRayTmax();
    perRayData &prd = *(perRayData*)getPRD<perRayData>();
    const HitGroupData &hg_data =  *(const HitGroupData*)(optixGetSbtDataPointer());

    const int primID = optixGetPrimitiveIndex();
    const gdt::vec3i index  = hg_data.index[primID];
    const gdt::vec3f &A     = hg_data.vertex[index.x];
    const gdt::vec3f &B     = hg_data.vertex[index.y];
    const gdt::vec3f &C     = hg_data.vertex[index.z];
    const gdt::vec3f Ng     = gdt::normalize(gdt::cross(B-A,C-A));

    const gdt::vec3f rayDir = optixGetWorldRayDirection();
    const bool inside = gdt::dot(rayDir, Ng) > 0.f ? true : false; 

    prd.t = t;

    if(inside)
    {
        // printf("%f\n",hg_data.opts[0].kappa);
        prd.kappa = hg_data.opts[0].kappa;
        prd.albedo = hg_data.opts[0].albedo;
    }
    else 
    {
        prd.kappa = hg_data.opts[1].kappa;
        prd.albedo = hg_data.opts[1].albedo;
    }
}

extern "C" __global__ void __miss__simulation()
{
    perRayData &prd = *(perRayData*)getPRD<perRayData>();
    prd.alive = false;
}

extern "C" __device__ gdt::vec3f emit(uint64_t &seed)
{
    float uxx, uyy, uzz;
    float phi = 2.f * PI * rnd(seed);
    float cost = 2.f * rnd(seed) - 1.f;
    float sint = sqrtf(1.f - cost * cost);
    uxx = sint * cosf(phi);
    uyy = sint * sinf(phi); 
    uzz = cost;
    return gdt::vec3f(uxx, uyy, uzz);
}


extern "C" __device__ gdt::vec3f scatter(gdt::vec3f &dir, const float &hgg, const float &g2, uint64_t &seed)
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

    float phi = 2.f * PI * rnd(seed);
    float cosp = cosf(phi);
    float sinp;
    if(phi < PI)
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

extern "C" __device__ uint32_t getVoxel(const gdt::vec3f &pos)
{   
    gdt::vec3i size = optixLaunchParams.frame.size;
    int celli = (int)floorf(size.x * (pos.x + 1.5f) / (3.0f));
    int cellj = (int)floorf(size.y * (pos.y + 1.5f) / (3.0f));
    int cellk = (int)floorf(size.z * (pos.z + 1.5f) / (3.0f));

    if(celli < 0 || cellj < 0 || cellk < 0)
    {
        return size.z*size.x*size.y + size.y*size.y+size.x+1;
    }
    else if(celli > size.x || cellj > size.y || cellk > size.z)
    {
        return size.z*size.x*size.y + size.y*size.y+size.x+1;
    } else
    {
        return cellk*size.x*size.y + cellj*size.y+celli;
    }
}

extern "C" __device__ void hitSurface(gdt::vec3f &rayPos, gdt::vec3f &rayDir, perRayData &PRD, float &tau, float &L)
{
    for(;;)
    {
        // hit surface and move just over it
        rayPos += (PRD.t+EPS) * rayDir;

        uint32_t fbIndex = getVoxel(rayPos);
        atomicAdd(&optixLaunchParams.frame.fluenceBuffer[fbIndex], PRD.kappa);

        uint32_t u0, u1;
        packPointer(&PRD, u0, u1);

        // calculate remaining L
        float taurun = (PRD.t+EPS) * PRD.kappa;
        tau -= taurun;
        // get new t
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
                    u0, u1 );

        if(!PRD.alive)break;

        L = tau / PRD.kappa;
        if(L < PRD.t)
        {
            // calculate remaining L
            float taurun = (PRD.t+EPS) * PRD.kappa;
            float tmp;
            if(taurun > tau){
                taurun = tau;
            }
            tmp = taurun / PRD.kappa;
            rayPos += tmp * rayDir;
            uint32_t fbIndex = getVoxel(rayPos);
            atomicAdd(&optixLaunchParams.frame.fluenceBuffer[fbIndex], PRD.kappa);
            tau -= taurun;
            break;
        }
    }   
    // }
    // else
    // {
    //     // PRD.alive = false;
    //     hitSurface(rayPos, rayDir, PRD, tau, L);
    // }
}

extern "C" __global__ void __raygen__weight()
{
    // const int ix = optixGetLaunchIndex().x;
    // const int iy = optixGetLaunchIndex().y;
    // const uint3 dim = optixGetLaunchDimensions();
    // uint seed = tea<64>(dim.x * iy + ix, 0);


    // float mus; //= 10.0f;
    // float mua; //= 0.0f;
    // float hgg; //= 0.0;
    // float g2; //= hgg*hgg;

    // per ray data
    // perRayData PRD = perRayData();
    // PRD.alive = true;
    // PRD.weight = 1.0f;
    // PRD.nscatt = 0;
    // PRD.t = -1.0f;

    // for storing the payload
    // uint32_t u0, u1;
    // packPointer(&PRD, u0, u1);

    // gdt::vec3f rayPos = gdt::vec3f(0.f);
    // gdt::vec3d rayDir = emit(seed);
    // float absorb;
    // for(;;)
    // {

    //     optixTrace(optixLaunchParams.traversable,
    //     rayPos,
    //     rayDir,
    //     0.f,      // tmin
    //     1e20f,    // tmax
    //     0.0f,     // rayTime
    //     OptixVisibilityMask( 255 ),
    //     OPTIX_RAY_FLAG_DISABLE_ANYHIT,// OPTIX_RAY_FLAG_NONE,
    //     SURFACE_RAY_TYPE,             // SBT offset
    //     RAY_TYPE_COUNT,               // SBT stride
    //     SURFACE_RAY_TYPE,             // missSBTIndex 
    //     u0, u1 );                     // payload

        // mus = PRD.mat.mus;
        // mua = PRD.mat.mua;
        // hgg = PRD.mat.hgg;
        // g2 = PRD.mat.g2;

        // float L = -log(rnd(seed)) / mus;
        // if(L > PRD.t || !PRD.alive)break;

        // rayPos += L * rayDir;
        // rayDir = scatter(rayDir, hgg, g2, seed);
        // absorb = PRD.weight*(1.f - expf(-mua * L));
        // PRD.weight -= absorb;

        // uint32_t fbIndex = getVoxel(rayPos);
        // atomicAdd(&optixLaunchParams.frame.fluenceBuffer[fbIndex], absorb);
        // PRD.nscatt += 1;
        // if(PRD.weight < THRESHOLD) 
        // {
        //     if(rnd(seed) <= CHANCE)
        //     {
        //         PRD.weight /= CHANCE;
        //     }
        // } else 
        // {
        //     PRD.alive = false;
        //     break;
        // }
    // }
    // const uint32_t nsbIndex = ix+iy*optixLaunchParams.frame.nsize.x;
    // optixLaunchParams.frame.nscattBuffer[nsbIndex] = PRD.nscatt;
}

extern "C" __global__ void __raygen__simulate()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    uint64_t seed = tea<4>(dim.x * idx.y + idx.x, 0);

    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;

    // per ray data
    perRayData PRD = perRayData();
    // PRD.weight = 1.0f;
    PRD.t = 0.f;
    PRD.nscatt = 0;
    PRD.alive = true;
    PRD.kappa = 10.f;

    // for storing the payload
    uint32_t u0, u1;
    packPointer(&PRD, u0, u1);

    gdt::vec3f rayPos = gdt::vec3f(0.f);
    gdt::vec3f rayDir = emit(seed);
    float tau = -log(rnd(seed));
    float L =  tau / PRD.kappa;

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

        if(!PRD.alive)break;

        if(L < PRD.t){
            rayPos += L * rayDir;
            uint32_t fbIndex = getVoxel(rayPos);
            atomicAdd(&optixLaunchParams.frame.fluenceBuffer[fbIndex], PRD.kappa);

            if(rnd(seed) < PRD.albedo){
                rayDir = scatter(rayDir, 0.f, 0.f, seed);
                PRD.nscatt += 1;
                tau = -log(rnd(seed));
                L =  tau / PRD.kappa;
            }
            else
            {
                PRD.alive = false;
                break;
            }
        } else 
        { 
            uint32_t fbIndex = getVoxel(rayPos);
            atomicAdd(&optixLaunchParams.frame.fluenceBuffer[fbIndex], PRD.kappa);

            hitSurface(rayPos, rayDir, PRD, tau, L);
            if(!PRD.alive)break;
        }
        // uint32_t fbIndex = getVoxel(rayPos);
        // atomicAdd(&optixLaunchParams.frame.fluenceBuffer[fbIndex], PRD.kappa);
    }
    const uint32_t nsbIndex = ix+iy*optixLaunchParams.frame.nsize.x;
    optixLaunchParams.frame.nscattBuffer[nsbIndex] = PRD.nscatt;
}