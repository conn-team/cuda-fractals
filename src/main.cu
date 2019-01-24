#include <cstdint>
#include <cstdio>
#include <iostream>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda_gl_interop.h>

#include "cuda_helper.hpp"
#include "generator.hpp"
#include "mandelbrot.hpp"
#include "juliabuczek.hpp"

constexpr double SUPERSAMPLING_RATIO = 1;
constexpr double ZOOM_SPEED = 1.2;

struct cudaGraphicsResource *cudaViewBuffer;
GLuint viewBuffer, viewTexture;
Viewport view;
int width, height;

int lastX, lastY;
bool isMoving = false;

void updateTitle() {
    std::ostringstream tmp;
    tmp << "cuda-fractals (zoom: " << (1 / view.getScale()) << ", maxIters: " << view.maxIters << ")";
    std::string title = tmp.str();
    glutSetWindowTitle(title.c_str());
}

void printCoordinates() {
    std::cout << std::fixed << std::setprecision(std::max(0L, lround(-log10(view.getScale()))) + 5);
    std::cout << "center real: " << view.center.real() << std::endl;
    std::cout << "center imag: " << view.center.imag() << std::endl;
    std::cout << std::scientific << std::setprecision(5);
    std::cout << "scale: " << view.getScale() << std::endl << std::endl;
}

void onRender() {
    // Map PBO to CUDA
    size_t mappedSize;
    gpuErrchk(cudaGraphicsMapResources(1, &cudaViewBuffer, 0));
    gpuErrchk(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&view.devImage), &mappedSize, cudaViewBuffer));

    // Render image
    view.renderImage(Mandelbrot{});

    // Unmap PBO
    view.devImage = nullptr;
    gpuErrchk(cudaGraphicsUnmapResources(1, &cudaViewBuffer, 0));

    // Copy PBO to texture
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, viewBuffer);
    glBindTexture(GL_TEXTURE_2D, viewTexture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, view.width, view.height, GL_BGRA, GL_UNSIGNED_BYTE, nullptr);

    // Render full-screen quad
    glBegin(GL_QUADS);
        glTexCoord2f(0, 0); glVertex2f(0, 0);
        glTexCoord2f(1, 0); glVertex2f(1, 0);
        glTexCoord2f(1, 1); glVertex2f(1, 1);
        glTexCoord2f(0, 1); glVertex2f(0, 1);
    glEnd();

    glBindTexture(GL_TEXTURE_2D, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    glFlush();
}

void onMouse(int button, int state, int x, int y) {
    if (button == 3 || button == 4) {
        if (state == GLUT_UP) {
            return;
        }

        double zoom = (button == 3 ? ZOOM_SPEED : 1/ZOOM_SPEED);
        double dx = 2.0*x/width - 1;
        double dy = 2.0*y/width - double(height)/width;

        view.center.real(view.center.real() - dx*(zoom-1)*view.getScale());
        view.center.imag(view.center.imag() + dy*(zoom-1)*view.getScale());
        view.setScale(view.getScale() * zoom);

        updateTitle();
        glutPostRedisplay();
    } else if (button == GLUT_LEFT_BUTTON) {
        isMoving = (state == GLUT_DOWN);
    } else if (button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN) {
        printCoordinates();
    }
}

void onMotion(int x, int y) {
    int dx = x-lastX, dy = y-lastY;
    lastX = x;
    lastY = y;

    if (isMoving) {
        view.center.real(view.center.real() - 2*dx*view.getScale()/width);
        view.center.imag(view.center.imag() + 2*dy*view.getScale()/width);
        glutPostRedisplay();
    }
}

void onKeyboard(unsigned char key, int, int) {
    if (key == 's') {
        view.useSeriesApproximation = !view.useSeriesApproximation;
        glutPostRedisplay();
    }
}

void onSpecialKeyboard(int key, int, int) {
    if (key == GLUT_KEY_UP) {
        view.maxIters += 250;
    } else if (key == GLUT_KEY_DOWN) {
        view.maxIters = std::max(view.maxIters-250, 0);
    } else {
        return;
    }

    glutPostRedisplay();
    updateTitle();
}

void onReshape(int w, int h) {
    width = w;
    height = h;
    view.width = int(lround(w*SUPERSAMPLING_RATIO));
    view.height = int(lround(h*SUPERSAMPLING_RATIO));

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
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, view.width, view.height, 0, GL_BGRA, GL_UNSIGNED_BYTE, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);

    // Allocate and register PBO
    glGenBuffers(1, &viewBuffer);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, viewBuffer);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, view.width*view.height*4, nullptr, GL_DYNAMIC_COPY);
    gpuErrchk(cudaGraphicsGLRegisterBuffer(&cudaViewBuffer, viewBuffer, cudaGraphicsMapFlagsWriteDiscard));
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // Setup scene
    glViewport(0, 0, width, height);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glOrtho(0, 1, 0, 1, 0, 1);
}

int main(int argc, char **argv) {
    view.center = -0.7;
    view.setScale(1.5);
    view.maxIters = 250;

    // NaN series breaking zoom
    // view.maxIters = 1250;
    // view.setScale(2.80969e-104);
    // view.center.real(BigFloat("-0.4968141896256946114192256490519277983341532366871239006397328938102282969608105818918291392319167436980711814"));
    // view.center.imag(BigFloat("-0.6359556404531552450576807825161928936851063796124890071820830821264561315128502700495367013919277125766152971"));

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_SINGLE);
    glutInitWindowSize(800, 600);
    glutInitWindowPosition(200, 200);
    glutCreateWindow("cuda-fractals");

    glewInit();
    updateTitle();

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
