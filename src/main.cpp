#include <iostream>
#include <iomanip>
#include <chrono>
#include <string>

#include "SampleSimulation.h"
#include "render.h"
#include "gdt/math/vec.h"
#include "io.hpp"
#include "window.h"

void printUsageAndExit( const std::string& argv0 )
{
    std::cerr << "\nUsage: " << argv0 << " [options]\n";
    std::cerr <<
        "App Options:\n"
        "  -h | --help                  Print this usage message and exit.\n"
        "  -f | --file <input_file>     Model to render or simulate.\n"
        "  -r | --render                Selects render mode.\n"
        "  -s | --simulate              Selects MCRT mode (default).\n"
        "\n"
        << std::endl;

    exit(1);
}

int main(int argc, char** argv)
{
    std::string model_file = "models/sphere.obj";
    bool render_mode = false;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg(argv[i]);
        if(arg == "-h" || arg == "--help")
        {
            printUsageAndExit(argv[0]);
        }
        else if(arg == "-f" || arg == "--file")
        {
            if( i == argc-1)
            {
                std::cerr << "Option'" << arg << "' requires additional argument" <<std::endl;
                printUsageAndExit(argv[0]);
            }
            model_file = argv[++i];
        }
        else if(arg == "-s" || arg == "--simulate")
        {
            render_mode = false;
        }
        else if(arg == "-r" || arg == "--render")
        {
            render_mode = true;
        }

    }

    Model *model = loadOBJ(model_file);

    if(render_mode)
    {
        Camera camera = { /*from*/gdt::vec3f(-10.f,10.f,-10.f),
                        /* at */gdt::vec3f(0.f,0.f,0.f),
                        /* up */gdt::vec3f(0.f,1.f,0.f) };
        // something approximating the scale of the world, so the
        // camera knows how much to move for any given user interaction:
        const float worldScale = 1.f;

        SampleWindow *window = new SampleWindow("oMCRT Renderer",
                                                model,camera,worldScale);
        window->run();
    } else
    {
        int nphotonsSqrt = 10000;
        SampleSimulation sim(model, "__raygen__simulate");

        const gdt::vec3i fbSize(gdt::vec3i(100,100, 100));
        const gdt::vec2i nsSize(gdt::vec2i(nphotonsSqrt,nphotonsSqrt));
        sim.resizeOutputBuffers(fbSize, nsSize);
        std::vector<float> fluence(fbSize.x*fbSize.y*fbSize.z);
        std::vector<int> nscatts(nsSize.x*nsSize.y);

        auto t0 = std::chrono::system_clock::now(); // tic
        sim.simulate(nphotonsSqrt);
        auto t1 = std::chrono::system_clock::now(); // toc

        auto diff = std::chrono::duration<float>(t1 - t0).count();
        std::cout << std::setprecision(4) << "MPhotons/s: " << (nphotonsSqrt*nphotonsSqrt/(diff))/1000000 << std::endl;

        sim.downloadFluence(fluence.data());
        sim.downloadNscatt(nscatts.data());
        writeNRRD(fluence);

        long int total = 0;
        for (auto i : nscatts)
        {   
            total += i;
        }
        float taumax = 10.f;
        std::cout << "<#scatt> MCRT code: " << total / (float)(nsSize.x * nsSize.y) << std::endl;
        std::cout << "<#scatt> Theory:    "<< (taumax*taumax) / 2.f + taumax << std::endl;
    }
    return 0;
}
