#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda_gl_interop.h>

#include "autozoom.hpp"
#include "benchmark.hpp"
#include "cuda_helper.hpp"
#include "renderer.hpp"
#include "mandelbrot.hpp"
#include "julia.hpp"
#include "position_library.hpp"

constexpr double SUPERSAMPLING_RATIO = 1;
constexpr double ZOOM_SPEED = 1.2;

struct cudaGraphicsResource *cudaViewBuffer;
GLuint viewBuffer, viewTexture;
int width, height;

int lastX, lastY;
bool isMoving = false;
std::string lastTitle;

Renderer<Mandelbrot> mandelbrotView;
Renderer<Julia> juliaView;
std::vector<BaseRenderer*> views = { &mandelbrotView, &juliaView };
int fractalIdx = 0;

bool inPickMode = false;
bool inAutoZoom = false;
AutoZoom autoZoom;

BaseRenderer& getView() {
    return *views[fractalIdx];
}

Complex<float> mouseToCoords(int x, int y) {
    x -= width / 2.f;
    y -= height / 2.f;
    return {
        float(getView().center.x) + (2.f * x / width) * float(getView().getScale()),
        float(getView().center.y) + (2.f * y / width) * float(getView().getScale())
    };
}

void initPickMode(void) {
    inPickMode = true;
    mandelbrotView.reset();
    juliaView.reset();
    fractalIdx = 0;
}

void updatePickMode(int x, int y) {
    juliaView.params.seed = mouseToCoords(x, y);
}

void endPickMode(void) {
    fractalIdx = 1;
    inPickMode = false;
}

void updateTitle() {
    std::ostringstream tmp;
    tmp << std::boolalpha;
    tmp << "zoom: " << (1 / getView().getScale()) << ", iters: " << getView().maxIters;

    if (getView().useSeriesApproximation) {
        tmp << ", skip: " << getView().skippedIters;
    }
    tmp << ", avg: " << getView().avgIters << ", max: " << getView().realMaxIters;

    std::string title = tmp.str();

    if (title != lastTitle) {
        glutSetWindowTitle(title.c_str());
        lastTitle = title;
    }
}

void printCoordinates() {
    std::cout << std::fixed << std::setprecision(std::max(0L, lround(-log10(getView().getScale()))) + 5);
    std::cout << "center real: " << getView().center.x << std::endl;
    std::cout << "center imag: " << getView().center.y << std::endl;
    std::cout << std::scientific << std::setprecision(5);
    std::cout << "scale: " << getView().getScale() << std::endl << std::endl;
}

template<bool Corner>
void renderQuad() {
    glBegin(GL_QUADS);
        glTexCoord2f(0, 0); glVertex2f(0, 0);
        glTexCoord2f(1, 0); glVertex2f(1, 0);
        glTexCoord2f(1, 1); glVertex2f(1, 1);
        glTexCoord2f(0, 1); glVertex2f(0, 1);
    glEnd();
}

template<>
void renderQuad<true>() {
    glBegin(GL_QUADS);
        glTexCoord2f(0, 0); glVertex2f(0   , 0.66);
        glTexCoord2f(1, 0); glVertex2f(0.33, 0.66);
        glTexCoord2f(1, 1); glVertex2f(0.33, 1.0 );
        glTexCoord2f(0, 1); glVertex2f(0   , 1.0 );
    glEnd();
}

template<bool Corner>
void renderView(BaseRenderer &view) {
    // Map PBO to CUDA
    void *devImage;
    size_t mappedSize;
    gpuErrchk(cudaGraphicsMapResources(1, &cudaViewBuffer, 0));
    gpuErrchk(cudaGraphicsResourceGetMappedPointer(&devImage, &mappedSize, cudaViewBuffer));

    // Render image
    view.render(reinterpret_cast<Color*>(devImage));

    // Unmap PBO
    gpuErrchk(cudaGraphicsUnmapResources(1, &cudaViewBuffer, 0));

    // Copy PBO to texture
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, viewBuffer);
    glBindTexture(GL_TEXTURE_2D, viewTexture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, view.width, view.height, GL_BGRA, GL_UNSIGNED_BYTE, nullptr);

    // Render full-screen quad
    renderQuad<Corner>();
}

void onRender() {
    if (inPickMode) {
        renderView<false>(mandelbrotView);
        renderView<true>(juliaView);
    } else {
        renderView<false>(getView());
    }

    glBindTexture(GL_TEXTURE_2D, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    glutSwapBuffers();
}

void onMouse(int button, int state, int x, int y) {
    if ((button == 3 || button == 4) && !inPickMode) {
        if (state == GLUT_UP) {
            return;
        }

        double zoom = (button == 3 ? ZOOM_SPEED : 1/ZOOM_SPEED);
        double dx = 2.0*x/width - 1;
        double dy = 2.0*y/width - double(height)/width;

        getView().center.x -= dx*(zoom-1)*getView().getScale();
        getView().center.y += dy*(zoom-1)*getView().getScale();
        getView().setScale(getView().getScale() * zoom);
        glutPostRedisplay();
    } else if (button == GLUT_LEFT_BUTTON && !inPickMode) {
        isMoving = (state == GLUT_DOWN);
    } else if (button == GLUT_LEFT_BUTTON && inPickMode) {
        endPickMode();
        glutPostRedisplay();
    } else if (button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN) {
        printCoordinates();
    }
}

void onMotion(int x, int y) {
    int dx = x-lastX, dy = y-lastY;
    lastX = x;
    lastY = y;

    if (inPickMode) {
        updatePickMode(x, y);
        glutPostRedisplay();
    }

    if (isMoving) {
        getView().center.x -= 2*dx*getView().getScale()/width;
        getView().center.y += 2*dy*getView().getScale()/width;
        glutPostRedisplay();
    }
}

void onKeyboard(unsigned char key, int, int) {
    if (key == 's') {
        getView().useSeriesApproximation = !getView().useSeriesApproximation;
    } else if (key == 'i') {
        getView().useSmoothing = !getView().useSmoothing;
    } else if (key == 'r') {
        getView().reset();
    } else if (key == 'p' && !inPickMode) {
        initPickMode();
    } else if (key == 'z' && !inAutoZoom) {
        inPickMode = false;
        inAutoZoom = true;
        autoZoom.init(&getView());
    } else if (key == 'x') {
        inAutoZoom = false;
        if (inPickMode) {
            inPickMode = false;
            glutPostRedisplay();
        }
    } else if (key >= '0' && key <= '9') {
        int id = key - '0';
        if (id < int(position_library::mandelbrot.size())) {
            auto& preset = position_library::mandelbrot[id];
            fractalIdx = 0;
            getView().maxIters = autoZoom.maxIters = preset.maxIters;
            getView().setScale(BigFloat(preset.scale));
            autoZoom.destScale = getView().getScale();
            getView().center.x = autoZoom.center.x = BigFloat(preset.real);
            getView().center.y = autoZoom.center.y = BigFloat(preset.imag);
        }
    } else {
        return;
    }

    glutPostRedisplay();
}

void onSpecialKeyboard(int key, int, int) {
    if (key == GLUT_KEY_UP && !inPickMode) {
        getView().maxIters += 250;
    } else if (key == GLUT_KEY_DOWN && !inPickMode) {
        getView().maxIters = std::max(getView().maxIters-250, 0);
    } else if (key == GLUT_KEY_RIGHT && !inPickMode) {
        fractalIdx = (fractalIdx + 1) % views.size();
    } else if (key == GLUT_KEY_LEFT && !inPickMode) {
        if (--fractalIdx < 0) {
            fractalIdx = views.size() - 1;
        }
    } else {
        return;
    }

    glutPostRedisplay();
}

void onReshape(int w, int h) {
    width = w;
    height = h;

    for (auto view : views) {
        view->width = int(lround(w*SUPERSAMPLING_RATIO));
        view->height = int(lround(h*SUPERSAMPLING_RATIO));
    }

    // Free old buffers
    if (viewBuffer) {
        cudaGraphicsUnregisterResource(cudaViewBuffer);
        glDeleteBuffers(1, &viewBuffer);
        viewBuffer = 0;
    }
    if (viewTexture) {
        glDeleteTextures(1, &viewTexture);
        viewTexture = 0;
    }

    // Allocate texture
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &viewTexture);
    glBindTexture(GL_TEXTURE_2D, viewTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, getView().width, getView().height, 0, GL_BGRA, GL_UNSIGNED_BYTE, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);

    // Allocate and register PBO
    glGenBuffers(1, &viewBuffer);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, viewBuffer);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, getView().width*getView().height*4, nullptr, GL_DYNAMIC_COPY);
    gpuErrchk(cudaGraphicsGLRegisterBuffer(&cudaViewBuffer, viewBuffer, cudaGraphicsMapFlagsWriteDiscard));
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // Setup scene
    glViewport(0, 0, width, height);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glOrtho(0, 1, 0, 1, 0, 1);
}

int counter = 0;
void onTimer(int) {
    if (++counter % 12 == 0) {
        updateTitle();
    }

    if (inAutoZoom) {
        autoZoom.update(&getView());
        glutPostRedisplay();
    }

    glutTimerFunc(41, onTimer, 0);
}

int main(int argc, char **argv) {
    if (argc == 2 && strcmp(argv[1], "bench") == 0) {
        runBenchmarks();
        return 0;
    }

    if (argc == 6 && strcmp(argv[1], "zoom") == 0) {
        autoZoom.destScale = BigFloat(argv[4]);
        BigFloat::default_precision(max(30L, lround(10 - log10(autoZoom.destScale))));
        autoZoom.center.x = BigFloat(argv[2]);
        autoZoom.center.y = BigFloat(argv[3]);
        autoZoom.maxIters = atoi(argv[5]);
        
        std::cout << "======== AutoZoom params ========" << std::endl;
        std::cout << "center.x  " << autoZoom.center.x << std::endl;
        std::cout << "center.y  " << autoZoom.center.y << std::endl;
        std::cout << "destScale " << autoZoom.destScale << std::endl;
        std::cout << "maxIters  " << autoZoom.maxIters << std::endl;
        std::cout << std::endl;
    }

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(800, 600);
    glutInitWindowPosition(200, 200);
    glutCreateWindow("cuda-fractals");

    glewInit();
    onTimer(0);

    glutDisplayFunc(onRender);
    glutMouseFunc(onMouse);
    glutMotionFunc(onMotion);
    glutPassiveMotionFunc(onMotion);
    glutKeyboardFunc(onKeyboard);
    glutSpecialFunc(onSpecialKeyboard);
    glutReshapeFunc(onReshape);
    glutMainLoop();
    return 0;
}
