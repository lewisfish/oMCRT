#include <string>
#include <iostream>
#include <fstream>
#include <iostream>
#include <vector>
#include "gdt/math/vec.h"

void writeNRRD(const std::string & fileName, const gdt::vec3i &size, std::vector<float> &grid)
{
    std::ofstream outFile(fileName);

    outFile << "NRRD0004" << std::endl;
    outFile << "type: float" << std::endl;
    outFile << "dimension: 3" << std::endl;
    outFile << "sizes: "<< size.x << " " << size.y << " " << size.z << std::endl;
    outFile << "encoding: raw" << std::endl;
    outFile << "" << std::endl;

    outFile.close();

    std::fstream file;
    file.open(fileName, std::ios::app | std::ios::binary);
    file.write(reinterpret_cast<char*>(grid.data()), grid.size() * sizeof(float));
    file.close();
}