#pragma once

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>
#include <vector>
#include "LaunchParams.h"
#include "gdt/math/vec.h"
#include "CUDABuffer.h"
#include "model.h"

// struct TriangleMesh
// {
    // void addUnitCube(const gdt::affine3f &xfm);

//     std::vector<gdt::vec3f> vertex;
//     std::vector<gdt::vec3i> index;
// };

class SampleSimulation
{
    public:
        SampleSimulation(const Model *model);
        void simulate(const int &nphotonsSqrt);
        void resize(const gdt::vec3i &fluenceNewSize, const gdt::vec2i &nscattNewSize);
        void downloadPixels(float h_pixels[], int h_nscatt[]);

    protected:
        void initOptix();
        void createContext();
        void createModule();
        void createRaygenPrograms();
        void createMissPrograms();
        void createHitGroupPrograms();
        void createPipeline();
        void buildSBT();

        OptixTraversableHandle buildAccel();
    
    protected:
        CUcontext cudaContext;
        CUstream stream;
        cudaDeviceProp deviceProps;

        OptixDeviceContext optixContext;

        OptixPipeline               pipeline;
        OptixPipelineCompileOptions pipelineCompileOptions = {};
        OptixPipelineLinkOptions    pipelineLinkOptions    = {};

        OptixModule                 module;
        OptixModuleCompileOptions   moduleCompileOptions = {};

        std::vector<OptixProgramGroup> raygenPGs;
        CUDABuffer raygenRecordsBuffer;
        std::vector<OptixProgramGroup> missPGs;
        CUDABuffer missRecordsBuffer;
        std::vector<OptixProgramGroup> hitgroupPGs;
        CUDABuffer hitgroupRecordsBuffer;
        OptixShaderBindingTable sbt = {};

        LaunchParams launchParams;
        CUDABuffer   launchParamsBuffer;

        CUDABuffer fluenceBuffer;
        CUDABuffer nscattBuffer;

        const Model *model;
        std::vector<CUDABuffer> vertexBuffer;
        std::vector<CUDABuffer> indexBuffer;
        //! buffer that keeps the (final, compacted) accel structure
        CUDABuffer asBuffer;

};