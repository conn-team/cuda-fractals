#include "complex.hpp"
#include "ext_float.hpp"
#include "renderer.hpp"

extern bool inAutoZoom;

struct AutoZoom {
    void init(BaseRenderer *view) {
        view->reset();

        view->center = center;
        view->maxIters = maxIters;
    }

    void update(BaseRenderer *view) {
        BigFloat scale = view->getScale();
        if (destScale >= scale) {
            // callback ?
            inAutoZoom = false;
            return;
        }

        view->setScale(view->getScale() * rate);
    }

    BigComplex center;
    BigFloat destScale;
    int maxIters;

    double rate = 0.98;
    BigFloat currScale;
};