#include "render.h"

Renderer::Renderer(const Model *model) : model(model), optixHandle("__raygen__camera", "/home/lewis/postdoc/optix/mcrt/bin/oMCRT/rendererPrograms.optixir", "render")
{
    launchParams.traversable = buildAccel();
    launchParamsBuffer.alloc(sizeof(launchParams));
}

void Renderer::render()
{

    launchParamsBuffer.upload(&launchParams,1);

    optixLaunch(/*! pipeline we're launching launch: */
                            optixHandle.pipeline,optixHandle.stream,
                            /*! parameters and SBT */
                            launchParamsBuffer.d_pointer(),
                            launchParamsBuffer.sizeInBytes,
                            &optixHandle.sbt,
                            /*! dimensions of the launch: */
                            launchParams.frame.nsize.x,
                            launchParams.frame.nsize.y,
                            1
                            );
    // sync - make sure the frame is rendered before we download and
    // display (obviously, for a high-performance application you
    // want to use streams and double-buffering, but for this simple
    // example, this will have to do)
    cudaDeviceSynchronize();
}

void Renderer::resize(const gdt::vec2i &newSize)
{

    // resize framebuffer
    frameBuffer.resize(newSize.x*newSize.y*sizeof(int));

    // update the launch parameters that we'll pass to the optix
    // launch:
    launchParams.frame.nsize  = newSize;
    launchParams.frame.frameBuffer = (uint32_t*)frameBuffer.d_pointer();

}

void Renderer::downloadPixels(uint32_t h_pixels[])
{
    frameBuffer.download(h_pixels,
                         launchParams.frame.nsize.x*launchParams.frame.nsize.y);

}

void Renderer::setCamera(const Camera &camera)
{
    lastSetCamera = camera;
    launchParams.camera.position  = camera.from;
    launchParams.camera.direction = gdt::normalize(camera.at-camera.from);
    const float cosFovy = 0.66f;
    const float aspect = launchParams.frame.nsize.x / float(launchParams.frame.nsize.y);
    launchParams.camera.horizontal
        = cosFovy * aspect * gdt::normalize(cross(launchParams.camera.direction,
                                            camera.up));
    launchParams.camera.vertical
        = cosFovy * gdt::normalize(cross(launchParams.camera.horizontal,
                                    launchParams.camera.direction));
}
OptixTraversableHandle Renderer::buildAccel()
{
    
    vertexBuffer.resize(model->meshes.size());
    indexBuffer.resize(model->meshes.size());
    
    OptixTraversableHandle asHandle { 0 };
    
    // ==================================================================
    // triangle inputs
    // ==================================================================
    std::vector<OptixBuildInput> triangleInput(model->meshes.size());
    std::vector<CUdeviceptr> d_vertices(model->meshes.size());
    std::vector<CUdeviceptr> d_indices(model->meshes.size());
    std::vector<uint32_t> triangleInputFlags(model->meshes.size());

    for (int meshID=0;meshID<model->meshes.size();meshID++) {
      // upload the model to the device: the builder
      TriangleMesh &mesh = *model->meshes[meshID];
      vertexBuffer[meshID].alloc_and_upload(mesh.vertex);
      indexBuffer[meshID].alloc_and_upload(mesh.index);

      triangleInput[meshID] = {};
      triangleInput[meshID].type
        = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

      // create local variables, because we need a *pointer* to the
      // device pointers
      d_vertices[meshID] = vertexBuffer[meshID].d_pointer();
      d_indices[meshID]  = indexBuffer[meshID].d_pointer();
      
      triangleInput[meshID].triangleArray.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3;
      triangleInput[meshID].triangleArray.vertexStrideInBytes = sizeof(gdt::vec3f);
      triangleInput[meshID].triangleArray.numVertices         = (int)mesh.vertex.size();
      triangleInput[meshID].triangleArray.vertexBuffers       = &d_vertices[meshID];
    
      triangleInput[meshID].triangleArray.indexFormat         = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
      triangleInput[meshID].triangleArray.indexStrideInBytes  = sizeof(gdt::vec3i);
      triangleInput[meshID].triangleArray.numIndexTriplets    = (int)mesh.index.size();
      triangleInput[meshID].triangleArray.indexBuffer         = d_indices[meshID];
    
      triangleInputFlags[meshID] = 0 ;
    
      // in this example we have one SBT entry, and no per-primitive
      // materials:
      triangleInput[meshID].triangleArray.flags               = &triangleInputFlags[meshID];
      triangleInput[meshID].triangleArray.numSbtRecords               = 1;
      triangleInput[meshID].triangleArray.sbtIndexOffsetBuffer        = 0; 
      triangleInput[meshID].triangleArray.sbtIndexOffsetSizeInBytes   = 0; 
      triangleInput[meshID].triangleArray.sbtIndexOffsetStrideInBytes = 0; 
    }
    // ==================================================================
    // BLAS setup
    // ==================================================================
    
    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags             = OPTIX_BUILD_FLAG_NONE
      | OPTIX_BUILD_FLAG_ALLOW_COMPACTION
      ;
    accelOptions.motionOptions.numKeys  = 1;
    accelOptions.operation              = OPTIX_BUILD_OPERATION_BUILD;
    
    OptixAccelBufferSizes blasBufferSizes;
    optixAccelComputeMemoryUsage
                (optixHandle.optixContext,
                 &accelOptions,
                 triangleInput.data(),
                 (int)model->meshes.size(),  // num_build_inputs
                 &blasBufferSizes
                 );
    
    // ==================================================================
    // prepare compaction
    // ==================================================================
    
    CUDABuffer compactedSizeBuffer;
    compactedSizeBuffer.alloc(sizeof(uint64_t));
    
    OptixAccelEmitDesc emitDesc;
    emitDesc.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitDesc.result = compactedSizeBuffer.d_pointer();
    
    // ==================================================================
    // execute build (main stage)
    // ==================================================================
    
    CUDABuffer tempBuffer;
    tempBuffer.alloc(blasBufferSizes.tempSizeInBytes);
    
    CUDABuffer outputBuffer;
    outputBuffer.alloc(blasBufferSizes.outputSizeInBytes);
      
    optixAccelBuild(optixHandle.optixContext,
                                /* stream */0,
                                &accelOptions,
                                triangleInput.data(),
                                (int)model->meshes.size(),
                                tempBuffer.d_pointer(),
                                tempBuffer.sizeInBytes,
                                
                                outputBuffer.d_pointer(),
                                outputBuffer.sizeInBytes,
                                
                                &asHandle,
                                
                                &emitDesc,1
                                );
    cudaDeviceSynchronize();
    
    // ==================================================================
    // perform compaction
    // ==================================================================
    uint64_t compactedSize;
    compactedSizeBuffer.download(&compactedSize,1);
    
    asBuffer.alloc(compactedSize);
    optixAccelCompact(optixHandle.optixContext,
                                  /*stream:*/0,
                                  asHandle,
                                  asBuffer.d_pointer(),
                                  asBuffer.sizeInBytes,
                                  &asHandle);
    cudaDeviceSynchronize();
    
    // ==================================================================
    // aaaaaand .... clean up
    // ==================================================================
    outputBuffer.free(); // << the UNcompacted, temporary output buffer
    tempBuffer.free();
    compactedSizeBuffer.free();
    
    return asHandle;
  }