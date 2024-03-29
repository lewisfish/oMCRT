#include <iostream>
#include <fstream>
#include <sstream>

#include "optixclass.h"
#include "optix_helpers.h"

OptixClass::OptixClass(const std::string &rg_prog, const std::string &ptxCode, const std::string &progSuffix)
{
    initOptix();
    createContext();
    createModule(ptxCode);
    createRaygenPrograms(rg_prog);
    createMissPrograms(progSuffix);
    createHitGroupPrograms(progSuffix);
    createPipeline();
    // buildSBT(model);
}

void OptixClass::initOptix()
{
    cudaFree(0);
    int numDevices;
    cudaGetDeviceCount(&numDevices);
    if(numDevices == 0)
    {
        throw std::runtime_error("No CUDA capable devices found!");
    }
    std::cout << "Found " << numDevices << " CUDA devices" << std::endl;

    OPTIX_CHECK(optixInit());

}

void OptixClass::createContext()
{
    // Assume one device
    const int deviceID = 0;
    cudaSetDevice(deviceID);
    cudaStreamCreate(&stream);
      
    cudaGetDeviceProperties(&deviceProps, deviceID);
    std::cout << "Running on device: " << deviceProps.name << std::endl;
      
    CUresult cuRes = cuCtxGetCurrent(&cudaContext);
    if( cuRes != CUDA_SUCCESS ) 
        std::cerr << "Error querying current context: error code " << cuRes << std::endl;
      
    OPTIX_CHECK(optixDeviceContextCreate(cudaContext, 0, &optixContext));
    OPTIX_CHECK(optixDeviceContextSetLogCallback(optixContext, context_log_cb, nullptr, 4));
}

std::vector<char> OptixClass::readData(std::string const& filename)
{
  std::ifstream inputData(filename, std::ios::binary);

  if (inputData.fail())
  {
    std::cerr << "ERROR: readData() Failed to open file " << filename << '\n';
    return std::vector<char>();
  }

  // Copy the input buffer to a char vector.
  std::vector<char> data(std::istreambuf_iterator<char>(inputData), {});

  if (inputData.fail())
  {
    std::cerr << "ERROR: readData() Failed to read file " << filename << '\n';
    return std::vector<char>();
  }

  return data;
}

void OptixClass::createModule(const std::string &ptxCode)
{

    moduleCompileOptions.maxRegisterCount  = 50;
    moduleCompileOptions.optLevel          = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleCompileOptions.debugLevel        = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    pipelineCompileOptions = {};
    pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
    pipelineCompileOptions.usesMotionBlur     = false;
    pipelineCompileOptions.numPayloadValues   = 2;
    pipelineCompileOptions.numAttributeValues = 2;
    pipelineCompileOptions.exceptionFlags     = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParams";
      
    pipelineLinkOptions.maxTraceDepth          = 2;

    std::vector<char> programData = readData(ptxCode);

    char log[2048];
    size_t sizeof_log = sizeof( log );
      
    OPTIX_CHECK(optixModuleCreate(optixContext,
                                &moduleCompileOptions,
                                &pipelineCompileOptions,
                                programData.data(),
                                programData.size(),
                                log,&sizeof_log,
                                &module
                                ));

    if (sizeof_log > 1)std::cout << log << std::endl;
}

void OptixClass::createRaygenPrograms(const std::string &rg_prog)
{
    // we do a single ray gen program in this example:
    raygenPGs.resize(1);
    
    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc    = {};
    pgDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pgDesc.raygen.module            = module;
    pgDesc.raygen.entryFunctionName = rg_prog.c_str();

    // OptixProgramGroup raypg;
    char log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                    &pgDesc,
                                    1,
                                    &pgOptions,
                                    log,&sizeof_log,
                                    &raygenPGs[0]));
    if (sizeof_log > 1)std::cout << log << std::endl;
}

void OptixClass::createMissPrograms(const std::string &progSuffix)
{
    // we do a single ray gen program in this example:
    missPGs.resize(1);

    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc    = {};
    pgDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_MISS;
    pgDesc.miss.module            = module;
    
    const std::string name = "__miss__" + progSuffix;
    pgDesc.miss.entryFunctionName = name.c_str();

    // OptixProgramGroup raypg;
    char log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                        &pgDesc,
                                        1,
                                        &pgOptions,
                                        log,&sizeof_log,
                                        &missPGs[0]
                                        ));
    if (sizeof_log > 1)std::cout << log << std::endl;
}

void OptixClass::createHitGroupPrograms(const std::string &progSuffix)
{
    // for this simple example, we set up a single hit group
    hitgroupPGs.resize(1);

    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc    = {};
    pgDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    
    pgDesc.hitgroup.moduleCH            = module;
    const std::string nameCH = "__closesthit__" + progSuffix;
    pgDesc.hitgroup.entryFunctionNameCH = nameCH.c_str();

    // pgDesc.hitgroup.moduleAH            = module;
    // const std::string nameAH = "__anyhit__" + progSuffix;
    // pgDesc.hitgroup.entryFunctionNameAH = nameAH.c_str();

    char log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                        &pgDesc,
                                        1,
                                        &pgOptions,
                                        log,&sizeof_log,
                                        &hitgroupPGs[0]
                                        ));
    if (sizeof_log > 1)std::cout << log << std::endl;
}

void OptixClass::createPipeline()
{
    std::vector<OptixProgramGroup> programGroups;
    for (auto pg : raygenPGs)
        programGroups.push_back(pg);
    for (auto pg : missPGs)
        programGroups.push_back(pg);
    for (auto pg : hitgroupPGs)
        programGroups.push_back(pg);

    char log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK(optixPipelineCreate(optixContext,
                                    &pipelineCompileOptions,
                                    &pipelineLinkOptions,
                                    programGroups.data(),
                                    (int)programGroups.size(),
                                    log,&sizeof_log,
                                    &pipeline
                                    ));
    if (sizeof_log > 1) std::cout << log << std::endl;

    OPTIX_CHECK(optixPipelineSetStackSize
                (/* [in] The pipeline to configure the stack size for */
                 pipeline,
                 /* [in] The direct stack size requirement for direct
                    callables invoked from IS or AH. */
                 2*1024,
                 /* [in] The direct stack size requirement for direct
                    callables invoked from RG, MS, or CH.  */
                 2*1024,
                 /* [in] The continuation stack requirement. */
                 2*1024,
                 /* [in] The maximum depth of a traversable graph
                    passed to trace. */
                 2));
    if (sizeof_log > 1)std::cout << log << std::endl;
}