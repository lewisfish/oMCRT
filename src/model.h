// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#pragma once

#include "gdt/math/AffineSpace.h"
#include <vector>

  /*! a simple indexed triangle mesh that our sample renderer will
      render */
  struct TriangleMesh {
    std::vector<gdt::vec3f> vertex;
    std::vector<gdt::vec3f> normal;
    std::vector<gdt::vec2f> texcoord;
    std::vector<gdt::vec3i> index;

    // material data:
    gdt::vec3f              diffuse;
  };
  
  struct Model {
    ~Model()
    { for (auto mesh : meshes) delete mesh; }
    
    std::vector<TriangleMesh *> meshes;
    //! bounding box of all vertices in the model
    gdt::box3f bounds;
  };

  Model *loadOBJ(const std::string &objFile);