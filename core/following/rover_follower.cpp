#include <iostream>
#include <iomanip>
#include <string>
#include "rover_follower.hpp"

static const char* state_str(int s) {
    switch(s) {
        case 0: return "forward";
        case 1: return "turning";
        case 2: return "stop";
        default: return "unknown";
    }
}

void run_test(const std::string& name,
              RoverFollower& follower,
              int x, int y, int w, int h,
              const std::string& expect)
{
    Command cmd = follower.compute(x, y, w, h);
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\n[" << name << "]\n"
              << "  bbox      : (" << x << "," << y << "," << w << "," << h << ")\n"
              << "  linear    : " << cmd.linear  << "\n"
              << "  angular   : " << cmd.angular << "\n"
              << "  state     : " << state_str(cmd.state) << "\n"
              << "  error_x   : " << cmd.error_x << "\n"
              << "  error_y   : " << cmd.error_y << "\n"
              << "  bbox_h    : " << cmd.bbox_h  << "\n"
              << "  expect    : " << expect       << "\n";
}

int main() {
    RoverFollower follower;

    // case 1 — dead center
    run_test("center",
             follower, 181, 181, 150, 150,
             "state=forward, angular=0.0");

    // case 2 — object far left
    run_test("far left",
             follower, 20, 181, 150, 150,
             "state=turning, angular=+0.4 (turn left)");

    // case 3 — object far right
    run_test("far right",
             follower, 300, 181, 150, 150,
             "state=turning, angular=-0.4 (turn right)");

    // case 4 — object far away, high in frame
    run_test("far away",
             follower, 181, 50, 150, 80,
             "state=forward, linear=0.3 (full speed)");

    // case 5 — object close, stop
    run_test("too close",
             follower, 100, 100, 150, 200,
             "state=stop, linear=0.0, angular=0.0");

    return 0;
}