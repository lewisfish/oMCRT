#pragma once
#include "gdt/math/vec.h"
#include "CUDABuffer.h"
#include "model.h"
#include "optixclass.h"
#include "LaunchParams.h"

extern "C" char embedded_render_ptx_code[];

struct Camera {
    /*! camera position - *from* where we are looking */
    gdt::vec3f from;
    /*! which point we are looking *at* */
    gdt::vec3f at;
    /*! general up-vector */
    gdt::vec3f up;
  };

class Renderer
{
    public:
        Renderer(const Model *model);
        void render();
        void resize(const gdt::vec2i &newSize);
        void downloadPixels(uint32_t h_pixels[]);
        void setCamera(const Camera &camera);

    protected:
        OptixTraversableHandle buildAccel();

        OptixClass optixHandle;

        RendererLaunchParams launchParams;
        CUDABuffer   launchParamsBuffer;

        CUDABuffer frameBuffer;

        Camera lastSetCamera;

        const Model *model;
        std::vector<CUDABuffer> vertexBuffer;
        std::vector<CUDABuffer> indexBuffer;
        //! buffer that keeps the (final, compacted) accel structure
        CUDABuffer asBuffer;
};