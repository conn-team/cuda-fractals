#include <chrono>
#include <iostream>
#include <iomanip>

#include "cuda_helper.hpp"
#include "benchmark.hpp"
#include "renderer.hpp"
#include "mandelbrot.hpp"

constexpr int WIDTH = 800;
constexpr int HEIGHT = 600;
constexpr int REPEAT = 20;

using Clock = std::chrono::high_resolution_clock;
using Time = std::chrono::time_point<Clock>;

void benchmarkLocation(const char *name, const char *centerX, const char *centerY, double scale, int iters) {
    Renderer<Mandelbrot> view;
    view.width = WIDTH;
    view.height = HEIGHT;

    view.maxIters = iters;
    view.setScale(scale);
    view.center.x = BigFloat(centerX);
    view.center.y = BigFloat(centerY);

    CudaArray<Color> image(WIDTH*HEIGHT);

    Time start = Clock::now();

    for (int i = 0; i < REPEAT; i++) {
        view.render(image.data());
        gpuErrchk(cudaDeviceSynchronize());
    }

    Time finish = Clock::now();
    int time = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count();

    std::cout << std::setfill(' ') << std::setw(25) << name << "  ";
    std::cout << std::fixed << std::setprecision(3) << double(time)/1000 << "s" << std::endl;
}

void runBenchmarks() {
    benchmarkLocation("default", "-0.7", "0", 1.5, 250);
    benchmarkLocation("default-high-iters", "-0.7", "0", 1.5, 5000);

    benchmarkLocation(
        "deep",
        "-0.00608110996414361738609067953992572361899653523214435533970021571665715422246314237437352765062921732023885",
        "0.80710509130889151108972120096753442734230451562765569259191208053695855881108570059709735793308459433434174",
        6.23362e-103, 15000
    );

    benchmarkLocation(
        "mosaic",
        "0.372137738770323258373356630885867793129508737859268",
        "-0.090398245434178161692952411151009819302665482561413",
        3.47252e-47, 20000
    );
}
