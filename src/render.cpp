#include "render.h"
#include <cstring>
#include <optix_helpers.h>


Renderer::Renderer(const Model *model) : model(model), optixHandle("__raygen__camera", "/home/lewis/postdoc/optix/mcrt/bin/oMCRT/rendererPrograms.optixir", "render")
{
    OptixTraversableHandle traversable = buildAccel();
    launchParams.traversable = buildIAS(traversable);
    buildSBT(model);
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
  OptixTraversableHandle Renderer::buildIAS(const OptixTraversableHandle &trav)
{
    OptixTraversableHandle asHandle { 0 };

    std::vector<OptixInstance> optix_instances(100);

    unsigned int sbt_offset = 0;

    for (size_t i = 0; i < optix_instances.size(); i++)
    {
      optix_instances[i].flags = OPTIX_INSTANCE_FLAG_NONE;
      optix_instances[i].sbtOffset = static_cast<unsigned int>(i);
      optix_instances[i].visibilityMask = 255;
    }
    
    optix_instances[0].traversableHandle = trav;
    float trans[12] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
    std::memcpy(optix_instances[0].transform, &trans[0], sizeof(float) * 12);

    for (size_t i = 1; i < optix_instances.size(); i++)
    {
      optix_instances[i].traversableHandle = trav;
      float Tx, Ty, Tz;
      Tx = (10.f)*drand48()-5.f;
      Ty = (10.f)*drand48()-5.f;
      Tz = (10.f)*drand48()-5.f;
      float scale = 0.9*drand48()+.1;
      float transa[12] = {scale, 0, 0, Tx, 0, scale, 0, Ty, 0, 0, scale, Tz};
      std::memcpy(optix_instances[i].transform, &transa[0], sizeof(float) * 12);
    }

    const size_t instances_size_in_bytes = sizeof(OptixInstance) * size(optix_instances);
    CUDABuffer d_instances;
    d_instances.alloc_and_upload(optix_instances);

    OptixBuildInput buildInput = {};
    buildInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    buildInput.instanceArray.instances = d_instances.d_pointer();
    buildInput.instanceArray.numInstances = static_cast<unsigned int>(optix_instances.size());

    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes iasBufferSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
                optixHandle.optixContext,
                &accelOptions,
                &buildInput,
                1,
                &iasBufferSizes
    ));

    CUDABuffer compactedSizeBuffer;
    compactedSizeBuffer.alloc(sizeof(uint64_t));
    
    OptixAccelEmitDesc emitDesc;
    emitDesc.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitDesc.result = compactedSizeBuffer.d_pointer();

    CUDABuffer d_temp_buffer;
    d_temp_buffer.alloc(iasBufferSizes.tempSizeInBytes);

    CUDABuffer outputBuffer;
    outputBuffer.alloc(iasBufferSizes.outputSizeInBytes);

    OPTIX_CHECK(optixAccelBuild(
                optixHandle.optixContext,
                0,                  // CUDA stream
                &accelOptions,
                &buildInput,
                1,                  // num build inputs
                d_temp_buffer.d_pointer(),
                d_temp_buffer.sizeInBytes,
                outputBuffer.d_pointer(),
                outputBuffer.sizeInBytes,
                &asHandle,
                &emitDesc,            // emitted property list
                1                   // num emitted properties
                ) );
    cudaDeviceSynchronize();

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


    d_temp_buffer.free();
    outputBuffer.free();
    compactedSizeBuffer.free();

    return asHandle;
}
void Renderer::buildSBT(const Model *model)
{
// ------------------------------------------------------------------
    // build raygen records
    // ------------------------------------------------------------------
    std::vector<RaygenRecord> raygenRecords;
    for (int i=0;i<optixHandle.raygenPGs.size();i++) {
        RaygenRecord rec;
        OPTIX_CHECK(optixSbtRecordPackHeader(optixHandle.raygenPGs[i],&rec));
        // rec.data = nullptr; /* for now ... */
        raygenRecords.push_back(rec);
    }
    optixHandle.raygenRecordsBuffer.alloc_and_upload(raygenRecords);
    optixHandle.sbt.raygenRecord = optixHandle.raygenRecordsBuffer.d_pointer();

    // ------------------------------------------------------------------
    // build miss records
    // ------------------------------------------------------------------
    std::vector<MissRecord> missRecords;
    for (int i=0;i<optixHandle.missPGs.size();i++) {
        MissRecord rec;
        OPTIX_CHECK(optixSbtRecordPackHeader(optixHandle.missPGs[i],&rec));
        // rec.data = nullptr; /* for now ... */
        missRecords.push_back(rec);
    }
    optixHandle.missRecordsBuffer.alloc_and_upload(missRecords);
    optixHandle.sbt.missRecordBase          = optixHandle.missRecordsBuffer.d_pointer();
    optixHandle.sbt.missRecordStrideInBytes = sizeof(MissRecord);
    optixHandle.sbt.missRecordCount         = (int)missRecords.size();

    // ------------------------------------------------------------------
    // build hitgroup records
    // ------------------------------------------------------------------

    int numObjects = 100;//(int)model->meshes.size();
    std::vector<HitgroupRecord> hitgroupRecords;
    for (int meshID=0;meshID<numObjects;meshID++) {
        int objectType = 0;
        HitgroupRecord rec;
        OPTIX_CHECK(optixSbtRecordPackHeader(optixHandle.hitgroupPGs[objectType],&rec));
        rec.data.vertex = (gdt::vec3f*)vertexBuffer[meshID].d_pointer();
        rec.data.index  = (gdt::vec3i*)indexBuffer[meshID].d_pointer();
        rec.data.opts   = nullptr;//(opticalProperty*)optsBuffer[meshID].d_pointer();
        rec.data.objID = meshID;
        hitgroupRecords.push_back(rec);
    }
    optixHandle.hitgroupRecordsBuffer.alloc_and_upload(hitgroupRecords);
    optixHandle.sbt.hitgroupRecordBase          = optixHandle.hitgroupRecordsBuffer.d_pointer();
    optixHandle.sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
    optixHandle.sbt.hitgroupRecordCount         = (int)hitgroupRecords.size();
}