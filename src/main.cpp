#include <iostream>
#include "SampleSimulation.h"
#include "gdt/math/vec.h"
#include "gdt/math/AffineSpace.h"
#include "io.hpp"

int main(int argc, char const *argv[])
{

    Model *model = loadOBJ("spot.obj");

    // TriangleMesh model;
    // gdt::vec3f center = gdt::vec3f(0.f,0.f,0.f);
    // gdt::vec3f size = gdt::vec3f(2.f,2.f,2.f);
    // gdt::affine3f xfm;
    // xfm.p = center - 0.5f*size;
    // xfm.l.vx = gdt::vec3f(size.x,0.f,0.f);
    // xfm.l.vy = gdt::vec3f(0.f,size.y,0.f);
    // xfm.l.vz = gdt::vec3f(0.f,0.f,size.z);
    // model.addUnitCube(xfm);

    SampleSimulation sim(model);

    const gdt::vec3i fbSize(gdt::vec3i(100,100, 100));
    sim.resize(fbSize);
    std::vector<float> pixels(fbSize.x*fbSize.y*fbSize.z);


    sim.simulate();


    sim.downloadPixels(pixels.data());

    // for(auto i : pixels)
    // {
    //     if(i > 0)std::cout << i << std::endl; 
    // }
    writeNRRD(pixels);
    return 0;
}
