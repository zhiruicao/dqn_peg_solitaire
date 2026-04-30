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

    auto read2D = [&](const json& arr, int rows, int cols, std::vector<std::vector<float>>& W) {
        W.assign(rows, std::vector<float>(cols, 0.0f));
        for (int i = 0; i < rows; ++i) {
            const auto& row = arr[i];
            for (int k = 0; k < cols; ++k) W[i][k] = row[k].get<float>();
        }
        };

    auto read1D = [&](const json& arr, int len, std::vector<float>& v) {
        v.assign(len, 0.0f);
        for (int i = 0; i < len; ++i) v[i] = arr[i].get<float>();
        };

    auto read4D = [&](const json& arr, int out_c, int in_c, int kh, int kw, std::vector<float>& W) {
        W.assign(out_c * in_c * kh * kw, 0.0f);
        int idx = 0;
        for (int oc = 0; oc < out_c; ++oc)
            for (int ic = 0; ic < in_c; ++ic)
                for (int i = 0; i < kh; ++i)
                    for (int j = 0; j < kw; ++j)
                        W[idx++] = arr[oc][ic][i][j].get<float>();
        };

    read4D(j["feature_layer.1.weight"], 32, 1, 3, 3, W_conv1);
    read1D(j["feature_layer.1.bias"], 32, b_conv1);

    read4D(j["feature_layer.3.weight"], 64, 32, 3, 3, W_conv2);
    read1D(j["feature_layer.3.bias"], 64, b_conv2);

    read2D(j["feature_layer.6.weight"], 256, 3136, W_fc1);
    read1D(j["feature_layer.6.bias"], 256, b_fc1);

    read2D(j["value_stream.0.weight"], 128, 256, Wv1);
    read1D(j["value_stream.0.bias"], 128, bv1);
    {
        std::vector<std::vector<float>> tmp;
        read2D(j["value_stream.2.weight"], 1, 128, tmp);
        Wv2 = tmp[0];
    }
    {
        std::vector<float> tmp;
        read1D(j["value_stream.2.bias"], 1, tmp);
        bv2 = tmp[0];
    }

    read2D(j["advantage_stream.0.weight"], 128, 256, Wa1);
    read1D(j["advantage_stream.0.bias"], 128, ba1);
    read2D(j["advantage_stream.2.weight"], 76, 128, Wa2);
    read1D(j["advantage_stream.2.bias"], 76, ba2);
}

std::vector<float> DQN::forward(const std::array<float, 49>& x49) const {
    std::vector<float> out_conv1(32 * 49, 0.0f);
    for (int oc = 0; oc < 32; ++oc) {
        for (int y = 0; y < 7; ++y) {
            for (int x = 0; x < 7; ++x) {
                float s = b_conv1[oc];
                for (int ky = 0; ky < 3; ++ky) {
                    for (int kx = 0; kx < 3; ++kx) {
                        int iy = y + ky - 1; 
                        int ix = x + kx - 1;
                        if (iy >= 0 && iy < 7 && ix >= 0 && ix < 7) {
                            s += x49[iy * 7 + ix] * W_conv1[oc * 9 + ky * 3 + kx];
                        }
                    }
                }
                out_conv1[oc * 49 + y * 7 + x] = relu(s);
            }
        }
    }

    std::vector<float> out_conv2(64 * 49, 0.0f);
    for (int oc = 0; oc < 64; ++oc) {
        for (int y = 0; y < 7; ++y) {
            for (int x = 0; x < 7; ++x) {
                float s = b_conv2[oc];
                for (int ic = 0; ic < 32; ++ic) {
                    for (int ky = 0; ky < 3; ++ky) {
                        for (int kx = 0; kx < 3; ++kx) {
                            int iy = y + ky - 1;
                            int ix = x + kx - 1;
                            if (iy >= 0 && iy < 7 && ix >= 0 && ix < 7) {
                                s += out_conv1[ic * 49 + iy * 7 + ix] * W_conv2[(oc * 32 + ic) * 9 + ky * 3 + kx];
                            }
                        }
                    }
                }
                out_conv2[oc * 49 + y * 7 + x] = relu(s);
            }
        }
    }

    std::vector<float> features(256, 0.0f);
    for (int i = 0; i < 256; ++i) {
        float s = b_fc1[i];
        const auto& Wi = W_fc1[i];
        for (int j = 0; j < 3136; ++j) s += Wi[j] * out_conv2[j];
        features[i] = relu(s);
    }

    std::vector<float> hv1(128, 0.0f);
    for (int i = 0; i < 128; ++i) {
        float s = bv1[i];
        const auto& Wi = Wv1[i];
        for (int j = 0; j < 256; ++j) s += Wi[j] * features[j];
        hv1[i] = relu(s);
    }
    float V = bv2;
    for (int j = 0; j < 128; ++j) V += Wv2[j] * hv1[j];

    std::vector<float> ha1(128, 0.0f);
    for (int i = 0; i < 128; ++i) {
        float s = ba1[i];
        const auto& Wi = Wa1[i];
        for (int j = 0; j < 256; ++j) s += Wi[j] * features[j];
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