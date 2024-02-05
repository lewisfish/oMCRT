#include <iostream>
#include <iomanip>
#include <chrono>


#include "SampleSimulation.h"
#include "gdt/math/vec.h"
#include "gdt/math/AffineSpace.h"
#include "io.hpp"

int main(int argc, char const *argv[])
{
    // for more than one mesh make this an vector
    Model *model = loadOBJ("sphere.obj");
    const std::string rg_program = "__raygen__simulate";

    int nphotonsSqrt = 10000;
    SampleSimulation sim(model, rg_program);

    const gdt::vec3i fbSize(gdt::vec3i(100,100, 100));
    const gdt::vec2i nsSize(gdt::vec2i(nphotonsSqrt,nphotonsSqrt));
    sim.resize(fbSize, nsSize);
    std::vector<float> pixels(fbSize.x*fbSize.y*fbSize.z);
    std::vector<int> nscatts(nsSize.x*nsSize.y);

    auto t0 = std::chrono::system_clock::now(); // tic
    sim.simulate(nphotonsSqrt);
    auto t1 = std::chrono::system_clock::now(); // toc

    auto diff = std::chrono::duration<float>(t1 - t0).count();
    std::cout << std::setprecision(4) << "MPhotons/s: " << (nphotonsSqrt*nphotonsSqrt/(diff))/1000000 << std::endl;

    sim.downloadPixels(pixels.data(), nscatts.data());
    writeNRRD(pixels);

    long int total = 0;
    for (auto i : nscatts)
    {   
        total += i;
    }
    float taumax = 10.f;
    std::cout << "<#scatt> MCRT code: " << total / (float)(nsSize.x * nsSize.y) << std::endl;
    std::cout << "<#scatt> Theory:    "<< (taumax*taumax) / 2.f + taumax << std::endl;

    return 0;
}
