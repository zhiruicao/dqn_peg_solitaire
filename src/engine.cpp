#include "engine.h"
#include <fstream>
#include <iostream>
#include <cmath>
#include <iomanip>
#include "json.hpp" 

using json = nlohmann::json;

float DQN::relu(float x) { return x > 0.0f ? x : 0.0f; }

void DQN::load() {
    std::ifstream fin("assets/weights.json");
    json j;
    fin >> j;

    auto read2D = [&](const json& arr, int rows, int cols, std::vector<std::vector<float>>& W, const std::string& name) {
        W.assign(rows, std::vector<float>(cols, 0.0f));
        for (int i = 0; i < rows; ++i) {
            const auto& row = arr[i];
            for (int k = 0; k < cols; ++k) W[i][k] = row[k].get<float>();
        }
        };

    auto read1D = [&](const json& arr, int len, std::vector<float>& v, const std::string& name) {
        v.assign(len, 0.0f);
        for (int i = 0; i < len; ++i) v[i] = arr[i].get<float>();
        };

    read2D(j["feature_layer.0.weight"], 128, 49, W0, "feature_layer.0.weight");
    read1D(j["feature_layer.0.bias"], 128, b0, "feature_layer.0.bias");

    read2D(j["value_stream.0.weight"], 128, 128, Wv1, "value_stream.0.weight");
    read1D(j["value_stream.0.bias"], 128, bv1, "value_stream.0.bias");

    {
        std::vector<std::vector<float>> tmp;
        read2D(j["value_stream.2.weight"], 1, 128, tmp, "value_stream.2.weight");
        Wv2 = tmp[0];
    }
    {
        std::vector<float> tmp;
        read1D(j["value_stream.2.bias"], 1, tmp, "value_stream.2.bias");
        bv2 = tmp[0];
    }

    read2D(j["advantage_stream.0.weight"], 128, 128, Wa1, "advantage_stream.0.weight");
    read1D(j["advantage_stream.0.bias"], 128, ba1, "advantage_stream.0.bias");

    read2D(j["advantage_stream.2.weight"], 76, 128, Wa2, "advantage_stream.2.weight");
    read1D(j["advantage_stream.2.bias"], 76, ba2, "advantage_stream.2.bias");
}

std::vector<float> DQN::forward(const std::array<float, 49>& x49) const {
    std::vector<float> x(49);
    for (int i = 0; i < 49; ++i) x[i] = x49[i];

    std::vector<float> h(128, 0.0f);
    for (int i = 0; i < 128; ++i) {
        float s = b0[i];
        const auto& Wi = W0[i];
        for (int j = 0; j < 49; ++j) s += Wi[j] * x[j];
        h[i] = relu(s);
    }

    std::vector<float> hv1(128, 0.0f);
    for (int i = 0; i < 128; ++i) {
        float s = bv1[i];
        const auto& Wi = Wv1[i];
        for (int j = 0; j < 128; ++j) s += Wi[j] * h[j];
        hv1[i] = relu(s);
    }
    float V = bv2;
    for (int j = 0; j < 128; ++j) V += Wv2[j] * hv1[j];

    std::vector<float> ha1(128, 0.0f);
    for (int i = 0; i < 128; ++i) {
        float s = ba1[i];
        const auto& Wi = Wa1[i];
        for (int j = 0; j < 128; ++j) s += Wi[j] * h[j];
        ha1[i] = relu(s);
    }
    std::vector<float> A(76, 0.0f);
    for (int i = 0; i < 76; ++i) {
        float s = ba2[i];
        const auto& Wi = Wa2[i];
        for (int j = 0; j < 128; ++j) s += Wi[j] * ha1[j];
        A[i] = s;
    }

    float meanA = 0.0f;
    for (float v : A) meanA += v;
    meanA /= (float)A.size();

    std::vector<float> Q(76, 0.0f);
    for (int i = 0; i < 76; ++i) Q[i] = V + A[i] - meanA;
    return Q;
}
