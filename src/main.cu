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

bool inAutoZoom = false;
AutoZoom autoZoom;

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
    } else if (key == 'z' && !inAutoZoom) {
        inAutoZoom = true;
        autoZoom.init(&getView());
    } else if (key == 'x' && inAutoZoom) {
        inAutoZoom = false;
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

// for mandelbrot
// ./bin/main zoom -0.86351690793367237879098526735500363171122900926623390235656859029095985817479106667893417017331775248894204667404104759173049314094055150218014325200616883094376002965516933657614243657952728054695501187855097054392324039595415883494985229715906678884870521543681491308761687017203355756095815611887401303438035 0.24770085085542684897920154941114532978571652912585207591199032605489162434475579901621342900504326332001572471388836875257693078071821918832702805395251556576917743455093070180103998083138219966104076957094394557391349705788109482159372116384541942314989586824711647398290030452624776670470371203410076798241659 1.05924e-306 14000
// ./bin/main zoom -0.863516907933672378790985267355003631711229009266233902356568590290959858174791066678934170173317752488942046674041047591730493140940551502180143252006168830943760029655169336576142436579527280546955011878550970543923240395954158834949852297159066788848705215436812860635332602278160099751996136063974639 0.247700850855426848979201549411145329785716529125852075911990326054891624344755799016213429005043263320015724713888368752576930780718219188327028053952515565769177434550930701801039980831382199661040769570943945573913497057881094821593721163845419423149895868247117015536061719196690131056301225176190768 1.05269e-298 14000

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

    // Some deep location
    // mandelbrotView.maxIters = 14000;
    // mandelbrotView.setScale(1.05269e-298);
    // mandelbrotView.center.x = BigFloat("-0.863516907933672378790985267355003631711229009266233902356568590290959858174791066678934170173317752488942046674041047591730493140940551502180143252006168830943760029655169336576142436579527280546955011878550970543923240395954158834949852297159066788848705215436812860635332602278160099751996136063974639");
    // mandelbrotView.center.y = BigFloat("0.247700850855426848979201549411145329785716529125852075911990326054891624344755799016213429005043263320015724713888368752576930780718219188327028053952515565769177434550930701801039980831382199661040769570943945573913497057881094821593721163845419423149895868247117015536061719196690131056301225176190768");

    // Some very deep location
    // mandelbrotView.maxIters = 420000;
    // mandelbrotView.setScale(BigFloat("1e-1000"));
    // mandelbrotView.center.x = BigFloat("-1.768610493014677074503175653270226520239677907588665494837677257575304640978808475274635707362464044253014370289948538552508877464736415873052958422861932774670165994201643419934807500290056179906392909880374230601661671965436663874506006355684166693059189687544326482526337453326360163639772818993753021740632937840115380957766425092940720439911920812397880443241274616212526380871555846532502156439892026352831619587768336768186345867565251889103622267866223055366872757385322485553606302984011695749730200727740242949661790906981449438923948817795927101980894917081591610562406554244675206099799522186446427884314773626993347929810277790888202019035845973880637832335294368222957931354735878969938534303074032237618397187328436715391758039680667871461788151793412286894565873237610467572174105629653438005433391873958639508124883860426968801537270756998560434335574379853659221182422319763412022530545421664765603500398209444536908432136868648907188939238968853841659746686717617541828417173199448336773447645561102873230388632254317334566170379314718589610910079079436175134148945650553411");
    // mandelbrotView.center.y = BigFloat("0.001266613503868717702066411192242601576193940560471409817185010171762524792588903616691501346028502452530417599269384116816237002586460261272462170615382790262110756215389780859682964779212455295242650488799024701023353984576434859496345393442867544784349509799966996827374525729583822627564832207860235000491856039278975203253540119195661182532106440194050352510825207428197675168479460252154208762204074041030502712772770772439567249008997886131809082319952112293668096363959700371035596685905429248221153089843201890985651976151989928496969024027810874574434857210174914227391125217932725188214796457327981771026544613194033736960542354861910879704489564999937473456191049937984461971508132204319961501958583967780282332682705656745932852354591955251196335374396883193221988201865629549575259395090238463522557833659758739138043696167112257784649600743807944457388512639475417466113111928274012056049434349358618953361438127758918999578120953045365596358997480091072548929426951083179599722132179281125708039705266879359303320165515458347343055671220673027817611220892213570374041225632346");

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
