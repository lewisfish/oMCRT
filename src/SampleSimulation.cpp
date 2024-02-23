#include "SampleSimulation.h"
#include <iostream>
#include <optix_function_table_definition.h>
#include <optix_helpers.h>

// constructor
SampleSimulation::SampleSimulation(const Model *model, const std::string &rg_prog) : model(model), optixHandle(rg_prog, "/home/lewis/postdoc/optix/mcrt/bin/oMCRT/simulationPrograms.optixir", "simulation")
{
    launchParams.traversable = buildAccel();
    buildSBT(model);
    launchParamsBuffer.alloc(sizeof(launchParams));
}

OptixTraversableHandle SampleSimulation::buildAccel()
{
    vertexBuffer.resize(model->meshes.size());
    indexBuffer.resize(model->meshes.size());
    optsBuffer.resize(model->meshes.size());
    
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
      optsBuffer[meshID].alloc_and_upload(mesh.opts);


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
    OPTIX_CHECK(optixAccelComputeMemoryUsage
                (optixHandle.optixContext,
                 &accelOptions,
                 triangleInput.data(),
                 (int)model->meshes.size(),  // num_build_inputs
                 &blasBufferSizes
                 ));
    
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


void SampleSimulation::buildSBT(const Model *model)
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

    int numObjects = (int)model->meshes.size();
    std::vector<HitgroupRecord> hitgroupRecords;
    for (int meshID=0;meshID<numObjects;meshID++) {
        int objectType = 0;
        HitgroupRecord rec;
        OPTIX_CHECK(optixSbtRecordPackHeader(optixHandle.hitgroupPGs[objectType],&rec));
        rec.data.vertex = (gdt::vec3f*)vertexBuffer[meshID].d_pointer();
        rec.data.index  = (gdt::vec3i*)indexBuffer[meshID].d_pointer();
        rec.data.opts   = (opticalProperty*)optsBuffer[meshID].d_pointer();
        hitgroupRecords.push_back(rec);
    }
    optixHandle.hitgroupRecordsBuffer.alloc_and_upload(hitgroupRecords);
    optixHandle.sbt.hitgroupRecordBase          = optixHandle.hitgroupRecordsBuffer.d_pointer();
    optixHandle.sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
    optixHandle.sbt.hitgroupRecordCount         = (int)hitgroupRecords.size();
}



void SampleSimulation::simulate(const int &nphotonsSqrt)
{   
    launchParamsBuffer.upload(&launchParams,1);

    optixLaunch(/*! pipeline we're launching launch: */
                            optixHandle.pipeline,optixHandle.stream,
                            /*! parameters and SBT */
                            launchParamsBuffer.d_pointer(),
                            launchParamsBuffer.sizeInBytes,
                            &optixHandle.sbt,
                            /*! dimensions of the launch: */
                            nphotonsSqrt,
                            nphotonsSqrt,
                            1
                            );
    // sync - make sure the frame is rendered before we download and
    // display (obviously, for a high-performance application you
    // want to use streams and double-buffering, but for this simple
    // example, this will have to do)
    cudaDeviceSynchronize();
}


void SampleSimulation::resizeOutputBuffers(const gdt::vec3i &fluenceNewSize, const gdt::vec2i &nscattNewSize)
{
    // resize our cuda fluence buffer
    // padded with extra space for garbage data
    fluenceBuffer.resize(((fluenceNewSize.x*fluenceNewSize.y*fluenceNewSize.z)+1)*sizeof(float));
    // resize nscatt buffer
    nscattBuffer.resize(nscattNewSize.x*nscattNewSize.y*sizeof(int));

    // update the launch parameters that we'll pass to the optix
    // launch:
    launchParams.frame.size  = fluenceNewSize;
    launchParams.frame.fluenceBuffer = (float*)fluenceBuffer.d_pointer();
    launchParams.frame.nsize = nscattNewSize;
    launchParams.frame.nscattBuffer = (int*)nscattBuffer.d_pointer();
}

void SampleSimulation::downloadNscatt(int h_nscatt[])
{
    nscattBuffer.download(h_nscatt,
                          launchParams.frame.nsize.x*launchParams.frame.nsize.y);
}

/*! download the rendered color buffer */
void SampleSimulation::downloadFluence(float h_fluence[])
{
    fluenceBuffer.download(h_fluence,
                           launchParams.frame.size.x*launchParams.frame.size.y*launchParams.frame.size.z);
}