#include "game.h"
#include <iostream>
#include <array>
#include <string>
#include <iomanip>

Board::Board() {
    int init[7][7] = {
        {-1,-1, 1, 1, 1,-1,-1},
        {-1,-1, 1, 1, 1,-1,-1},
        { 1,  1,  1, 1, 1, 1, 1},
        { 1,  1,  1, 0, 1, 1, 1},
        { 1,  1,  1, 1, 1, 1, 1},
        {-1,-1, 1, 1, 1,-1,-1},
        {-1,-1, 1, 1, 1,-1,-1}
    };
    for (int i = 0; i < 7; ++i)
        for (int j = 0; j < 7; ++j)
            layout[i][j] = init[i][j];
}

std::vector<Move> Board::getMoves(const int layout[7][7]) {
    auto b = Board::flatten(layout);
    std::vector<Move> moves;
    const int dx[] = { -2, 2, 0, 0 };
    const int dy[] = { 0, 0, -2, 2 };
    for (int f = 0; f < 49; ++f) {
        int x = f / 7;
        int y = f % 7;

        if (b[f] == -1) continue;
        for (int i = 0; i < 4; ++i) {
            int tx = x + dx[i];
            int ty = y + dy[i];
            if (tx >= 0 && tx < 7 && ty >= 0 && ty < 7) {
                int t = tx * 7 + ty;
                if (b[t] != -1) {
                    moves.push_back({ x, y, tx, ty });
                }
            }
        }
    }
    return moves;
}

std::vector<int> Board::getActions(const int layout[7][7], const std::vector<Move>& moves) {
    std::vector<int> actions;
    for (int i = 0; i < 76; ++i) {
        const auto& m = moves[i];
        if (layout[m.x][m.y] == -1 || layout[m.to_x][m.to_y] == -1) continue;
        int mid_x = (m.x + m.to_x) / 2;
        int mid_y = (m.y + m.to_y) / 2;
        if (mid_x < 0 || mid_x >= 7 || mid_y < 0 || mid_y >= 7) continue;
        if (layout[mid_x][mid_y] == -1) continue;
        if (layout[m.x][m.y] == 1 && layout[mid_x][mid_y] == 1 && layout[m.to_x][m.to_y] == 0) {
            actions.push_back(i);
        }
    }
    return actions;
}

void Board::step(int layout[7][7], const Move& m) {
    int mid_x = (m.x + m.to_x) / 2;
    int mid_y = (m.y + m.to_y) / 2;
    layout[m.x][m.y] = 0;
    layout[mid_x][mid_y] = 0;
    layout[m.to_x][m.to_y] = 1;
}

void Board::undo() {
    if (history.empty()) return;

    Move m = history.back();
    history.pop_back();
    int mid_x = (m.x + m.to_x) / 2;
    int mid_y = (m.y + m.to_y) / 2;

    layout[m.x][m.y] = 1;
    layout[mid_x][mid_y] = 1;
    layout[m.to_x][m.to_y] = 0;
}

std::array<float, 49> Board::flatten(const int layout[7][7]) {
    std::array<float, 49> f{};
    int i = 0;
    for (int x = 0; x < 7; ++x) {
        for (int y = 0; y < 7; ++y) {
            f[i++] = static_cast<float>(layout[x][y]);
        }
    }
    return f;
}

int Board::count(const int layout[7][7]) {
    int c = 0;
    for (int x = 0; x < 7; ++x)
        for (int y = 0; y < 7; ++y)
            if (layout[x][y] == 1) ++c;
    return c;
}

Renderer::Renderer()
    : cellSize(150), boardW(cellSize * 7), boardH(cellSize * 7),
    startX((screenW - boardW) / 2), startY((screenH - boardH) / 2)
{
}

void Renderer::draw(const int layout[7][7], const std::optional<Move>& lastMove,
    int remain, const std::optional<std::pair<int, int>>& selected) {
    ClearBackground(BLACK);
    DrawText("Space: AI step | R: reset | Esc: quit", 40, 40, 28, WHITE);

    std::string pegText = std::string("Pegs: ") + std::to_string(remain);
    int pegW = MeasureText(pegText.c_str(), 50);
    DrawText(pegText.c_str(), screenW - pegW - 40, 30, 50, WHITE);
    for (int x = 0; x < 7; ++x) {
        for (int y = 0; y < 7; ++y) {
            int px = startX + y * cellSize;
            int py = startY + x * cellSize;
            Rectangle cell = { (float)px, (float)py, (float)cellSize, (float)cellSize };
            if (layout[x][y] == -1) {
                DrawRectangleRec(cell, BLACK);
                continue;
            }

            DrawRectangleLines(px, py, cellSize, cellSize, { 80,80,80,255 });
            Vector2 center = { (float)(px + cellSize / 2), (float)(py + cellSize / 2) };
            if (layout[x][y] == 1) {
                DrawCircleV(center, cellSize * 0.40f, WHITE);
                DrawCircleLines(center.x, center.y, cellSize * 0.40f, { 200,200,200,255 });
            }
            else {
                DrawCircleLines(center.x, center.y, cellSize * 0.38f, { 160,160,160,255 });
            }
        }
    }

    if (selected) {
        int sx = selected->first;
        int sy = selected->second;
        if (sx >= 0 && sx < 7 && sy >= 0 && sy < 7 && layout[sx][sy] != -1) {
            int px = startX + sy * cellSize;
            int py = startY + sx * cellSize;
            DrawRectangle(px, py, cellSize, cellSize, Fade({ 0,180,255,200 }, 0.25f));
            DrawRectangleLines(px, py, cellSize, cellSize, { 0,180,255,200 });
        }
    }

    if (lastMove) {
        const auto& m = lastMove.value();
        int sx = startX + m.y * cellSize;
        int sy = startY + m.x * cellSize;
        int tx = startX + m.to_y * cellSize;
        int ty = startY + m.to_x * cellSize;
        DrawRectangleLines(sx, sy, cellSize, cellSize, YELLOW);
        DrawRectangleLines(tx, ty, cellSize, cellSize, YELLOW);
        DrawLine(sx + cellSize / 2, sy + cellSize / 2, tx + cellSize / 2, ty + cellSize / 2, YELLOW);
    }
}

std::optional<std::pair<int, int>> Renderer::getOpt(int mouseX, int mouseY) const {
    if (mouseX < startX || mouseY < startY) return std::nullopt;
    int relX = mouseY - startY;
    int relY = mouseX - startX;
    if (relX < 0 || relY < 0) return std::nullopt;
    int row = relX / cellSize;
    int col = relY / cellSize;
    if (row < 0 || row >= 7 || col < 0 || col >= 7) return std::nullopt;
    return std::make_pair(row, col);
}