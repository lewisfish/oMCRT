#include <iostream>
#include <iomanip>

#include "SampleSimulation.h"
#include "gdt/math/vec.h"
#include "gdt/math/AffineSpace.h"
#include "io.hpp"

int main(int argc, char const *argv[])
{

    Model *model = loadOBJ("spot.obj");

    SampleSimulation sim(model);

    const gdt::vec3i fbSize(gdt::vec3i(100,100, 100));
    sim.resize(fbSize);
    std::vector<float> pixels(fbSize.x*fbSize.y*fbSize.z);


    cudaEvent_t start, stop;
    float milliseconds = 0;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    sim.simulate();

    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << std::setprecision(4) << "MPhotons/s: " << (10000*10000/(milliseconds/1000.0f))/1000000 << std::endl;

    sim.downloadPixels(pixels.data());
    writeNRRD(pixels);
    return 0;
}
