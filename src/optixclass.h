#pragma once

#include <optix.h>
#include <vector>
#include <string>

#include "CUDABuffer.h"

template <typename T>
struct SbtRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

struct RayGenData {};  
struct MissData {};
struct HitGroupData {
    float mus;
    float mua;
    float hgg;
    float g2;
    float albedo;
    float kappa;
    float n;
};

typedef SbtRecord<RayGenData>     RaygenRecord;
typedef SbtRecord<MissData>       MissRecord;
typedef SbtRecord<HitGroupData>   HitgroupRecord;

class OptixClass
{
public:
    OptixClass(const std::string &rg_prog, const std::string &ptxCode, const std::string &progSuffix);
    OptixClass() = delete;
protected:
    void initOptix();
    void createContext();
    void createModule(const std::string &ptxCode);
    void createRaygenPrograms(const std::string &rg_prog);
    void createMissPrograms(const std::string &progSuffix);
    void createHitGroupPrograms(const std::string &progSuffix);
    void createPipeline();
    void buildSBT();
    std::vector<char> readData(std::string const& filename);

    OptixTraversableHandle buildAccel();

public:
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

};