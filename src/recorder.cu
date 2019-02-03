#include <cstdio>
#include <iostream>
#include <iomanip>
#include <sstream>

#include "recorder.hpp"
#include "autozoom.hpp"
#include "mandelbrot.hpp"
#include "renderer.hpp"
#include "position_library.hpp"

constexpr int PRESET = 6;
constexpr double ZOOM_RATE = 0.9;
constexpr int FRAMERATE = 30;
constexpr int WIDTH = 1280;
constexpr int HEIGHT = 720;
constexpr int SUPERSAMPLING = 4;
const char *OUTPUT_FILE = "output.mkv";

FILE *ffmpegRecord(const char *file, int fps, int srcWidth, int srcHeight, int dstWidth, int dstHeight) {
    std::ostringstream cmd;
    cmd << "ffmpeg -y -f rawvideo -pix_fmt bgra";
    cmd << " -s " << srcWidth << "x" << srcHeight << " -r " << fps << " -i -";
    cmd << " -vf scale=\"" << dstWidth << ":" << dstHeight << "\"";
    cmd << " -c libx264 -preset slow -crf 18 -pix_fmt yuv420p -an " << file;

    std::string tmp = cmd.str();
    FILE *pipe = popen(tmp.c_str(), "w");

    if (!pipe) {
        std::cout << "Failed to open FFMPEG" << std::endl;
        exit(1);
    }

    return pipe;
}

void runRecorder() {
    if (PRESET >= int(position_library::mandelbrot.size())) {
        std::cout << "Invalid preset" << std::endl;
        exit(1);
    }

    auto& preset = position_library::mandelbrot[PRESET];

    AutoZoom autoZoom;
    autoZoom.rate = ZOOM_RATE;
    autoZoom.maxIters = preset.maxIters;
    autoZoom.destScale = BigFloat(preset.scale);
    BigFloat::default_precision(max(30L, lround(10 - log10(autoZoom.destScale))));
    autoZoom.center.x = BigFloat(preset.real);
    autoZoom.center.y = BigFloat(preset.imag);

    Renderer<Mandelbrot> view;
    view.width = WIDTH*SUPERSAMPLING;
    view.height = HEIGHT*SUPERSAMPLING;

    FILE *output = ffmpegRecord(OUTPUT_FILE, FRAMERATE, view.width, view.height, WIDTH, HEIGHT);

    CudaArray<Color> devImage(view.width*view.height);
    std::vector<Color> image;
    autoZoom.init(&view);

    for (int i = 1; autoZoom.update(&view); i++) {
        if (i % 5 == 0) {
            std::cout << "Frame #" << i << ": " << view.getScale() << " / " << autoZoom.destScale << std::endl;
        }
        view.render(devImage.data());
        devImage.get(image);
        fwrite(image.data(), 4, image.size(), output);
    }

    fflush(output);
    pclose(output);
}
