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

struct Camera {
    /*! camera position - *from* where we are looking */
    gdt::vec3f from;
    /*! which point we are looking *at* */
    gdt::vec3f at;
    /*! general up-vector */
    gdt::vec3f up;
  };

class SampleSimulation
{
    public:
        SampleSimulation(const Model *model, const std::string &rg_prog);
        void simulate(const int &nphotonsSqrt);
        void render();
        void resizeOutputBuffers(const gdt::vec3i &fluenceNewSize, const gdt::vec2i &nscattNewSize);
        void resizeCanvas(const gdt::vec2i &newSize);
        void downloadFluence(float h_fluence[]);
        void downloadNscatt(int h_nscatt[]);
        void downloadPixels(uint32_t h_pixels[]);
        void setCamera(const Camera &camera);

    protected:
        OptixTraversableHandle buildAccel();
        OptixTraversableHandle buildSphereAccel();

        OptixClass optixHandle;

        LaunchParams launchParams;
        CUDABuffer   launchParamsBuffer;

        CUDABuffer frameBuffer;
        CUDABuffer fluenceBuffer;
        CUDABuffer nscattBuffer;

        Camera lastSetCamera;

        const Model *model;
        std::vector<CUDABuffer> vertexBuffer;
        std::vector<CUDABuffer> indexBuffer;
        //! buffer that keeps the (final, compacted) accel structure
        CUDABuffer asBuffer;

        std::vector<CUDABuffer> sphereVertexBuffer;
        std::vector<CUDABuffer> sphereRadiusBuffer;
        CUDABuffer iasBuffer;

};