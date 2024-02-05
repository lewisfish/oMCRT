#include <string>
#include <iostream>
#include <fstream>
#include <iostream>
#include <vector>

void writeNRRD(std::vector<float> &grid)
{
    std::string name="test.nrrd";

    std::ofstream outFile(name);

    outFile << "NRRD0004" << std::endl;
    outFile << "type: float" << std::endl;
    outFile << "dimension: 3" << std::endl;
    outFile << "sizes: 100 100 100" << std::endl;
    outFile << "encoding: raw" << std::endl;
    outFile << "" << std::endl;

    outFile.close();

    std::fstream file;
    file.open(name, std::ios::app | std::ios::binary);
    file.write(reinterpret_cast<char*>(grid.data()), grid.size() * sizeof(float));
    file.close();
}