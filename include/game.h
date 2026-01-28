#pragma once

#include <vector>
#include <array>
#include <optional>
#include <utility>
#include <raylib.h>
#include "engine.h"

struct Move {
    int x, y, to_x, to_y;
};

struct Board {
    int layout[7][7];

    Board();

    static std::vector<Move> getMoves(const int layout[7][7]);
    static std::vector<int> getActions(const int layout[7][7], const std::vector<Move>& moves);
    static void step(int layout[7][7], const Move& m);
    static std::array<float, 49> flatten(const int layout[7][7]);
    static int count(const int layout[7][7]);

    std::vector<Move> history;
    void undo();
};

struct Renderer {
    const int screenW = 1600;
    const int screenH = 1200;
    int cellSize;
    int boardW;
    int boardH;
    int startX;
    int startY;

    Renderer();
    void draw(const int layout[7][7], const std::optional<Move>& lastMove,
        int remain, const std::optional<std::pair<int, int>>& selected);
    std::optional<std::pair<int, int>> getOpt(int mouseX, int mouseY) const;
};