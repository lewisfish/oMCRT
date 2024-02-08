#include <iostream>
#include <iomanip>
#include <chrono>
#include <string>

#include "SampleSimulation.h"
#include "render.h"
#include "gdt/math/vec.h"
#include "io.hpp"
#include <GL/gl.h>
#include "glfWindow/GLFWindow.h"

struct SampleWindow : public GLFCameraWindow 
{
    SampleWindow(const std::string &title,
                 const Model *model,
                 const Camera &camera,
                 const float worldScale)
      : GLFCameraWindow(title,camera.from,camera.at,camera.up,worldScale),
        sample(model)
    {
      sample.setCamera(camera);
    }
    
    virtual void render() override
    {
      if (cameraFrame.modified) {
        sample.setCamera(Camera{ cameraFrame.get_from(),
                                 cameraFrame.get_at(),
                                 cameraFrame.get_up() });
        cameraFrame.modified = false;
      }
      sample.render();
    }

virtual void draw() override
    {
      sample.downloadPixels(pixels.data());
      if (fbTexture == 0)
        glGenTextures(1, &fbTexture);
      
      glBindTexture(GL_TEXTURE_2D, fbTexture);
      GLenum texFormat = GL_RGBA;
      GLenum texelType = GL_UNSIGNED_BYTE;
      glTexImage2D(GL_TEXTURE_2D, 0, texFormat, fbSize.x, fbSize.y, 0, GL_RGBA,
                   texelType, pixels.data());

      glDisable(GL_LIGHTING);
      glColor3f(1, 1, 1);

      glMatrixMode(GL_MODELVIEW);
      glLoadIdentity();

      glEnable(GL_TEXTURE_2D);
      glBindTexture(GL_TEXTURE_2D, fbTexture);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      
      glDisable(GL_DEPTH_TEST);

      glViewport(0, 0, fbSize.x, fbSize.y);

      glMatrixMode(GL_PROJECTION);
      glLoadIdentity();
      glOrtho(0.f, (float)fbSize.x, 0.f, (float)fbSize.y, -1.f, 1.f);

      glBegin(GL_QUADS);
      {
        glTexCoord2f(0.f, 0.f);
        glVertex3f(0.f, 0.f, 0.f);
      
        glTexCoord2f(0.f, 1.f);
        glVertex3f(0.f, (float)fbSize.y, 0.f);
      
        glTexCoord2f(1.f, 1.f);
        glVertex3f((float)fbSize.x, (float)fbSize.y, 0.f);
      
        glTexCoord2f(1.f, 0.f);
        glVertex3f((float)fbSize.x, 0.f, 0.f);
      }
      glEnd();
    }
    
    virtual void resize(const gdt::vec2i &newSize) 
    {
      fbSize = newSize;
      sample.resize(newSize);
      pixels.resize(newSize.x*newSize.y);
    }

    gdt::vec2i            fbSize;
    GLuint                fbTexture {0};
    Renderer              sample;
    std::vector<uint32_t> pixels;
  };


int main(int argc, char const *argv[])
{
    Model *model = loadOBJ("models/spot.obj");
    // const std::string rg_program = "__raygen__camera";//"__raygen__simulate";

    Camera camera = { /*from*/gdt::vec3f(-10.f,10.f,-10.f),
                      /* at */gdt::vec3f(0.f,0.f,0.f),
                      /* up */gdt::vec3f(0.f,1.f,0.f) };
    // something approximating the scale of the world, so the
    // camera knows how much to move for any given user interaction:
    const float worldScale = 1.f;

      SampleWindow *window = new SampleWindow("oMCRT Renderer",
                                              model,camera,worldScale);
      window->run();


    return 0;

    // int nphotonsSqrt = 10000;
    // SampleSimulation sim(model, rg_program);

    // const gdt::vec3i fbSize(gdt::vec3i(100,100, 100));
    // const gdt::vec2i nsSize(gdt::vec2i(nphotonsSqrt,nphotonsSqrt));
    // sim.resizeOutputBuffers(fbSize, nsSize);
    // std::vector<float> fluence(fbSize.x*fbSize.y*fbSize.z);
    // std::vector<int> nscatts(nsSize.x*nsSize.y);

    // auto t0 = std::chrono::system_clock::now(); // tic
    // sim.simulate(nphotonsSqrt);
    // auto t1 = std::chrono::system_clock::now(); // toc

    // auto diff = std::chrono::duration<float>(t1 - t0).count();
    // std::cout << std::setprecision(4) << "MPhotons/s: " << (nphotonsSqrt*nphotonsSqrt/(diff))/1000000 << std::endl;

    // sim.downloadFluence(fluence.data());
    // sim.downloadNscatt(nscatts.data());
    // writeNRRD(fluence);

    // long int total = 0;
    // for (auto i : nscatts)
    // {   
    //     total += i;
    // }
    // float taumax = 10.f;
    // std::cout << "<#scatt> MCRT code: " << total / (float)(nsSize.x * nsSize.y) << std::endl;
    // std::cout << "<#scatt> Theory:    "<< (taumax*taumax) / 2.f + taumax << std::endl;

    // return 0;
}
