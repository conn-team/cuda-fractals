#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda_gl_interop.h>

#include "benchmark.hpp"
#include "cuda_helper.hpp"
#include "renderer.hpp"
#include "mandelbrot.hpp"
#include "julia.hpp"

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

// first is mandelbrot
// second is julia
int const pickViews[] = {0, 1};
bool inPickMode = false;

void initPickMode(void) {
    inPickMode = true;
    fractalIdx = pickViews[0];
    for (int i = 0; i < 2; ++i) {
        views[pickViews[i]]->reset();
    }
}

void updatePickMode(int y, int x) {
    const Renderer<Mandelbrot>& mandelbrot = *dynamic_cast<Renderer<Mandelbrot>*>(views[pickViews[0]]);
    Renderer<Julia>& julia = *dynamic_cast<Renderer<Julia>*>(views[pickViews[1]]);
    const auto seed = mandelbrot.mouseToCoords(y, x);
    julia.params.seed = seed;
}

void endPickMode(void) {
    fractalIdx = pickViews[1];
    inPickMode = false;
}

BaseRenderer& getView() {
    return *views[fractalIdx];
}

void updateTitle() {
    std::ostringstream tmp;
    tmp << std::boolalpha;
    tmp << "zoom: " << (1 / getView().getScale()) << ", iterations: " << getView().maxIters << ", skipped: " << getView().skippedIters;
    tmp << ", approx: " << getView().useSeriesApproximation << ", smoothing: " << getView().useSmoothing;
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

template <bool Corner>
void _renderQuad() {
    glBegin(GL_QUADS);
        glTexCoord2f(0, 0); glVertex2f(0, 0);
        glTexCoord2f(1, 0); glVertex2f(1, 0);
        glTexCoord2f(1, 1); glVertex2f(1, 1);
        glTexCoord2f(0, 1); glVertex2f(0, 1);
    glEnd();
}

template <>
void _renderQuad<true>() {
    glBegin(GL_QUADS);
        glTexCoord2f(0, 0); glVertex2f(0   , 0.66);
        glTexCoord2f(1, 0); glVertex2f(0.33, 0.66);
        glTexCoord2f(1, 1); glVertex2f(0.33, 1.0 );
        glTexCoord2f(0, 1); glVertex2f(0   , 1.0 );
    glEnd();
}

template <bool Corner>
void _renderView(BaseRenderer &view) {
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
    _renderQuad<Corner>();
}

inline void renderViewFull(BaseRenderer &view) {
    _renderView<false>(view);
}

inline void renderViewCorner(BaseRenderer &view) {
    _renderView<true>(view);
}

void onRender() {
    if (inPickMode) {
        BaseRenderer& mandelbrot = *views[pickViews[0]];
        BaseRenderer& julia = *views[pickViews[1]];

        renderViewFull(mandelbrot);
        renderViewCorner(julia);
    } else {
        renderViewFull(getView());
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
        updatePickMode(y, x);
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

void onTimer(int) {
    updateTitle();
    glutTimerFunc(500, onTimer, 0);
}

int main(int argc, char **argv) {
    if (argc == 2 && strcmp(argv[1], "bench") == 0) {
        runBenchmarks();
        return 0;
    }

    // NaN series breaking zoom
    // mandelbrotView.maxIters = 1250;
    // mandelbrotView.setScale(2.80969e-104);
    // mandelbrotView.center.x = BigFloat("-0.4968141896256946114192256490519277983341532366871239006397328938102282969608105818918291392319167436980711814");
    // mandelbrotView.center.y = BigFloat("-0.6359556404531552450576807825161928936851063796124890071820830821264561315128502700495367013919277125766152971");

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
