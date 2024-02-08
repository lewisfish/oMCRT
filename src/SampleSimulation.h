#pragma once

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>
#include <vector>
#include "LaunchParams.h"
#include "gdt/math/vec.h"
#include "CUDABuffer.h"
#include "model.h"
#include "optixclass.h"

class SampleSimulation
{
    public:
        SampleSimulation(const Model *model, const std::string &rg_prog);
        void simulate(const int &nphotonsSqrt);
        void resizeOutputBuffers(const gdt::vec3i &fluenceNewSize, const gdt::vec2i &nscattNewSize);
        void downloadFluence(float h_fluence[]);
        void downloadNscatt(int h_nscatt[]);

    protected:
        OptixTraversableHandle buildAccel();
        OptixTraversableHandle buildSphereAccel();

        OptixClass optixHandle;

        SimulationLaunchParams launchParams;
        CUDABuffer   launchParamsBuffer;

        CUDABuffer frameBuffer;
        CUDABuffer fluenceBuffer;
        CUDABuffer nscattBuffer;

        const Model *model;
        std::vector<CUDABuffer> vertexBuffer;
        std::vector<CUDABuffer> indexBuffer;
        //! buffer that keeps the (final, compacted) accel structure
        CUDABuffer asBuffer;

        std::vector<CUDABuffer> sphereVertexBuffer;
        std::vector<CUDABuffer> sphereRadiusBuffer;
        CUDABuffer iasBuffer;

};