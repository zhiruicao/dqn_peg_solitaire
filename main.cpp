#include <raylib.h>
#include <iostream>
#include <optional>
#include <iomanip>
#include <algorithm>
#include "game.h"

int main() {
    const int screenW = 1600;
    const int screenH = 1200;
    InitWindow(screenW, screenH, "Peg Solitaire With AI Assistant");
    SetTargetFPS(60);

    Board board;
    Renderer renderer;
    DQN net;

    std::vector<Move> moves = Board::getMoves(board.layout);
    net.load();

    std::optional<Move> lastMove;;
    std::optional<std::pair<int, int>> selected;
    bool gameOver = false;
    bool gameWin = false;

    auto reset = [&]() {
        Board tmp;
        board = tmp;
        lastMove.reset();
        selected.reset();
        board.history.clear();
        gameOver = false;
        gameWin = false;
        };

    auto check = [&]() {
        int remain = Board::count(board.layout);
        if (remain <= 1) {
            gameOver = true;
            gameWin = true;
            return;
        }
        auto actions = Board::getActions(board.layout, moves);
        if (actions.empty()) {
            gameOver = true; gameWin = false;
            return;
        }
        };

    check();
    while (!WindowShouldClose()) {
        if (!gameOver) {
            if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
                int mx = GetMouseX(), my = GetMouseY();
                auto opt = renderer.getOpt(mx, my);
                if (opt) {
                    int row = opt->first, col = opt->second;
                    if (board.layout[row][col] != -1) {
                        if (!selected) {
                            if (board.layout[row][col]) selected = std::make_pair(row, col);
                        }
                        else {
                            int sx = selected->first, sy = selected->second;
                            int tx = row, ty = col;
                            int a = -1;
                            for (int i = 0; i < 76; ++i) {
                                const auto& m = moves[i];
                                if (m.x == sx && m.y == sy && m.to_x == tx && m.to_y == ty) {
                                    a = i;
                                    break;
                                }
                            }
                            if (a == -1) {
                                selected.reset();
                            }
                            else {
                                Move m = moves[a];
                                std::vector<int> actions = Board::getActions(board.layout, moves);
                                if (std::find(actions.begin(), actions.end(), a) != actions.end()) {
                                    Board::step(board.layout, m);
                                    board.history.push_back(m);
                                    lastMove = m;
                                    check();
                                }
                                selected.reset();
                            }
                        }
                    }
                }
                else {
                    if (selected) {
                        selected.reset();
                    }
                }
            }

            if (IsKeyPressed(KEY_BACKSPACE)) {
                if (!board.history.empty()) {
                    board.undo();
                    if (!board.history.empty()) {
                        lastMove = board.history.back();
                    }
                    else {
                        lastMove.reset();
                    }
                    check();
                }
            }

            if (IsKeyPressed(KEY_SPACE)) {
                auto x = Board::flatten(board.layout);
                auto Q = net.forward(x);
                auto actions = Board::getActions(board.layout, moves);

                if (!actions.empty()) {
                    int a = actions[0];
                    float q = Q[a];
                    for (size_t k = 1; k < actions.size(); ++k) {
                        int idx = actions[k];
                        if (Q[idx] > q) { q = Q[idx]; a = idx; }
                    }
                    Board::step(board.layout, moves[a]);
                    board.history.push_back(moves[a]);
                    lastMove = moves[a];
                }
                check();
            }
        }
        if (IsKeyPressed(KEY_R)) reset();
        BeginDrawing();
        if (!gameOver) {
            renderer.draw(board.layout, lastMove, Board::count(board.layout), selected);
        }
        else {
            ClearBackground(BLACK);
            const char* msg = gameWin ? "Victory!" : "Defeat!";
            int w = MeasureText(msg, 96);
            DrawText(msg, (screenW - w) / 2, screenH / 2 - 80, 96, WHITE);
        }
        EndDrawing();
    }
    CloseWindow();
    return 0;
}