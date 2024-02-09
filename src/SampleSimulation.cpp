#include "SampleSimulation.h"
#include <iostream>
#include <optix_function_table_definition.h>

// constructor
SampleSimulation::SampleSimulation(const Model *model, const std::string &rg_prog) : model(model), optixHandle(rg_prog, "/home/lewis/postdoc/optix/mcrt/bin/oMCRT/simulationPrograms.optixir", "simulation")
{
    launchParams.traversable = buildAccel();
    launchParamsBuffer.alloc(sizeof(launchParams));
}

OptixTraversableHandle SampleSimulation::buildAccel()
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


OptixTraversableHandle SampleSimulation::buildSphereAccel()
{

    OptixTraversableHandle asHandle;
    CUdeviceptr            d_gas_output_buffer;

    sphereVertexBuffer.resize(0);
    sphereRadiusBuffer.resize(0);

    std::vector<gdt::vec3f> sphereVertex(5);
    std::vector<gdt::vec3f> sphereRadii(5);

    for (auto i = 0; i < 5; i++)
    {
        sphereVertex.push_back(gdt::vec3f((float)i, 0.f, (0.f)));
        sphereRadii.push_back(1.f);
    }
    
    std::vector<OptixBuildInput> sphereInput(5);
    std::vector<CUdeviceptr> d_vertices(5);
    std::vector<CUdeviceptr> d_radius(5);
    std::vector<uint32_t> sphereInputFlags(5);

    sphereVertexBuffer[0].alloc_and_upload(sphereVertex);
    sphereRadiusBuffer[0].alloc_and_upload(sphereRadii);
    for (int sphereID = 0; sphereID < 1; sphereID++)
    {
        sphereInput[sphereID] = {};
        sphereInput[sphereID].type = OPTIX_BUILD_INPUT_TYPE_SPHERES;

        d_vertices[sphereID] = sphereVertexBuffer[sphereID].d_pointer();
        d_radius[sphereID] = sphereRadiusBuffer[sphereID].d_pointer();

        sphereInput[sphereID].sphereArray.vertexBuffers = &d_vertices[sphereID];
        sphereInput[sphereID].sphereArray.numVertices = 5;
        sphereInput[sphereID].sphereArray.radiusBuffers = &d_radius[sphereID];

        uint32_t sphereInputFlags[sphereID] = {OPTIX_GEOMETRY_FLAG_NONE};
        sphereInput[sphereID].sphereArray.flags = sphereInputFlags;
        sphereInput[sphereID].sphereArray.numSbtRecords = 1;
    }
    // BLAS setup
    OptixAccelBufferSizes blasBufferSizes;
    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
    accelOptions.operation  = OPTIX_BUILD_OPERATION_BUILD;
    optixAccelComputeMemoryUsage(optixHandle.optixContext, &accelOptions, sphereInput.data(), 1, &blasBufferSizes);

    // Prepare compaction

    CUDABuffer compactedSizeBuffer;
    compactedSizeBuffer.alloc(sizeof(uint64_t));

    OptixAccelEmitDesc emitDesc;
    emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitDesc.result = compactedSizeBuffer.d_pointer();

    // execute build (main stage)

    CUDABuffer tempBuffer;
    tempBuffer.alloc(blasBufferSizes.tempSizeInBytes);

    CUDABuffer outputBuffer;
    outputBuffer.alloc(blasBufferSizes.outputSizeInBytes);

    optixAccelBuild(optixHandle.optixContext, 0,
                    &accelOptions, sphereInput.data(),
                    5,
                    tempBuffer.d_pointer(),
                    tempBuffer.sizeInBytes,
                    outputBuffer.d_pointer(),
                    outputBuffer.sizeInBytes,
                    &asHandle,
                    &emitDesc, 1);
    cudaDeviceSynchronize();

    // perfrom compaction
    uint64_t compactedSize;
    compactedSizeBuffer.download(&compactedSize, 1);
    iasBuffer.alloc(compactedSize);
    optixAccelCompact(optixHandle.optixContext,
                        0,
                        asHandle,
                        asBuffer.d_pointer(),
                        iasBuffer.sizeInBytes,
                        &asHandle);
    cudaDeviceSynchronize();

    //clean up
    outputBuffer.free();
    tempBuffer.free();
    compactedSizeBuffer.free();

    return asHandle;   

}

void SampleSimulation::simulate(const int &nphotonsSqrt)
{   
    launchParams.optProps.mus[0] = 10.0f;
    launchParams.optProps.mus[1] = 5.0f;
    launchParams.optProps.mus[2] = 1.0f;
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
    fluenceBuffer.resize(fluenceNewSize.x*fluenceNewSize.y*fluenceNewSize.z*sizeof(float));
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