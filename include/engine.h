#pragma once

#include <vector>
#include <array>
#include <string>
#include <optional>

class DQN {
public:
    void load();

    std::vector<float> forward(const std::array<float, 49>& x) const;

private:
    std::vector<float> W_conv1; 
    std::vector<float> b_conv1; 

    std::vector<float> W_conv2; 
    std::vector<float> b_conv2;

    std::vector<std::vector<float>> W_fc1; 
    std::vector<float> b_fc1;              

    std::vector<std::vector<float>> Wv1;   
    std::vector<float> bv1;                
    std::vector<float> Wv2;                
    float bv2 = 0.0f;

    std::vector<std::vector<float>> Wa1;   
    std::vector<float> ba1;                
    std::vector<std::vector<float>> Wa2;   
    std::vector<float> ba2;               

    static float relu(float x);
};