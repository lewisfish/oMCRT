#pragma once
#include <vector>
#include <string>

struct opticalProperty
{
    float mus;
    float mua;
    float albedo;
    float kappa;
    float n;
    float hgg;
    float g2;
    opticalProperty()=default;
    opticalProperty(const float &mus, const float &mua, const float &n, const float &hgg) : mus(mus), mua(mua), n(n), hgg(hgg)
    {
        // std::cout << mus << " " << mua << std::endl;
        kappa = mus + mua;
        albedo = mus / kappa;
        // std::cout << albedo << " " << kappa << std::endl;
    
        g2 = hgg * hgg;
    };
};
