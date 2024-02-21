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

#include "model.h"
#include "opticalProperty.h"
#include "json.h"
using json = nlohmann::json;

#define TINYOBJLOADER_IMPLEMENTATION
#include "../ext/tiny/tiny_obj_loader.h"
//std
#include <set>

namespace std {
  inline bool operator<(const tinyobj::index_t &a,
                        const tinyobj::index_t &b)
  {
    if (a.vertex_index < b.vertex_index) return true;
    if (a.vertex_index > b.vertex_index) return false;
    
    if (a.normal_index < b.normal_index) return true;
    if (a.normal_index > b.normal_index) return false;
    
    // if (a.texcoord_index < b.texcoord_index) return true;
    // if (a.texcoord_index > b.texcoord_index) return false;
    
    return false;
  }
}

  /*! find vertex with given position, normal, texcoord, and return
      its vertex ID, or, if it doesn't exit, add it to the mesh, and
      its just-created index */
  int addVertex(TriangleMesh *mesh,
                tinyobj::attrib_t &attributes,
                const tinyobj::index_t &idx,
                std::map<tinyobj::index_t,int> &knownVertices)
  {
    if (knownVertices.find(idx) != knownVertices.end())
      return knownVertices[idx];

    const gdt::vec3f *vertex_array   = (const gdt::vec3f*)attributes.vertices.data();
    const gdt::vec3f *normal_array   = (const gdt::vec3f*)attributes.normals.data();
    const gdt::vec2f *texcoord_array = (const gdt::vec2f*)attributes.texcoords.data();
    
    int newID = mesh->vertex.size();
    knownVertices[idx] = newID;

    mesh->vertex.push_back(vertex_array[idx.vertex_index]);
    if (idx.normal_index >= 0) {
      while (mesh->normal.size() < mesh->vertex.size())
        mesh->normal.push_back(normal_array[idx.normal_index]);
    }
    // if (idx.texcoord_index >= 0) {
    //   while (mesh->texcoord.size() < mesh->vertex.size())
    //     mesh->texcoord.push_back(texcoord_array[idx.texcoord_index]);
    // }

    // // just for sanity's sake:
    // if (mesh->texcoord.size() > 0)
    //   mesh->texcoord.resize(mesh->vertex.size());
    // just for sanity's sake:
    if (mesh->normal.size() > 0)
      mesh->normal.resize(mesh->vertex.size());
    
    return newID;
  }
  
  void parseJson(std::vector<opticalProperty> &opts, std::string &objFile, const std::string &jsonFile)
  {

      std::ifstream f(jsonFile);
      json data = json::parse(f);
      objFile = data["file"];
      float mus = data["inside"]["mus"];
      float mua = data["inside"]["mua"];
      opts.push_back(opticalProperty(mus, mua, 1.f, 0.0f));
      mus = data["outside"]["mus"];
      mua = data["outside"]["mua"];
      opts.push_back(opticalProperty(mus, mua, 1.f, 0.0f));
  }


  Model *loadOBJ(const std::string &jsonFile, std::string &outFile)
  {
    Model *model = new Model;

    std::vector<opticalProperty> opts;
    std::string objFile;
    parseJson(opts, objFile, jsonFile);
    outFile = objFile;

    const std::string mtlDir
      = objFile.substr(0,objFile.rfind('/')+1);
    
    tinyobj::attrib_t attributes;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string err = "";

    bool readOK
      = tinyobj::LoadObj(&attributes,
                         &shapes,
                         &materials,
                         &err,
                         &err,
						             objFile.c_str(),
                         mtlDir.c_str(),
                         /* triangulate */true);
    if (!readOK) {
      throw std::runtime_error("Could not read OBJ model from "+objFile+":"+mtlDir+" : "+err);
    }

    std::cout << "Done loading obj file - found " << shapes.size() << " shapes with " << materials.size() << " materials" << std::endl;
    for (int shapeID=0;shapeID<(int)shapes.size();shapeID++) {
      tinyobj::shape_t &shape = shapes[shapeID];

      std::set<int> materialIDs;
      for (auto faceMatID : shape.mesh.material_ids)
        materialIDs.insert(faceMatID);
      
      for (int materialID : materialIDs) {
        std::map<tinyobj::index_t,int> knownVertices;
        TriangleMesh *mesh = new TriangleMesh;
        
        for (int faceID=0;faceID<shape.mesh.material_ids.size();faceID++) {
          if (shape.mesh.material_ids[faceID] != materialID) continue;
          tinyobj::index_t idx0 = shape.mesh.indices[3*faceID+0];
          tinyobj::index_t idx1 = shape.mesh.indices[3*faceID+1];
          tinyobj::index_t idx2 = shape.mesh.indices[3*faceID+2];
          
          gdt::vec3i idx(addVertex(mesh, attributes, idx0, knownVertices),
                    addVertex(mesh, attributes, idx1, knownVertices),
                    addVertex(mesh, attributes, idx2, knownVertices));
          mesh->index.push_back(idx);
          mesh->diffuse = gdt::randomColor(materialID);
        }

        if (mesh->vertex.empty())
          delete mesh;
        else
          model->meshes.push_back(mesh);
      }
    }

    // of course, you should be using tbb::parallel_for for stuff
    // like this:
    for (auto mesh : model->meshes)
      for (auto vtx : mesh->vertex)
        model->bounds.extend(vtx);


      for (int idx= 0; idx< model->meshes.size(); idx++)
      {
        model->meshes[idx]->opts.push_back(opts[0]);
        model->meshes[idx]->opts.push_back(opts[1]);
      }
    std::cout << "created a total of " << model->meshes.size() << " meshes" << std::endl;
    return model;
  }