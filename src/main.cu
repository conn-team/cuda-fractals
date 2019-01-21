#include <cstdint>
#include <cstdio>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda_gl_interop.h>

#include "cuda_helper.hpp"
#include "generator.hpp"

constexpr double SUPERSAMPLING_RATIO = 2;

struct cudaGraphicsResource *cudaViewBuffer;
GLuint viewBuffer, viewTexture;
Viewport view;
int width, height;

void onRender() {
    // Map PBO to CUDA
    size_t mappedSize;
    gpuErrchk(cudaGraphicsMapResources(1, &cudaViewBuffer, 0));
    gpuErrchk(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&view.image), &mappedSize, cudaViewBuffer));

    // Render image
    renderImage(view);

    // Unmap PBO
    view.image = nullptr;
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
    view.center = { -0.7, 0 };
    view.scale = 1.5;

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_SINGLE);
    glutInitWindowSize(800, 600);
    glutInitWindowPosition(200, 200);
    glutCreateWindow("cuda-fractals");

    glewInit();

    glutDisplayFunc(onRender);
    glutReshapeFunc(onReshape);
    glutMainLoop();
    return 0;
}
