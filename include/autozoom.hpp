#include "complex.hpp"
#include "ext_float.hpp"
#include "renderer.hpp"

extern bool inAutoZoom;

struct AutoZoom {
    void init(BaseRenderer *view) {
        view->reset();
        view->center = center;
        view->maxIters = 250;
    }

    void update(BaseRenderer *view) {
        BigFloat scale = view->getScale();
        if (destScale >= scale) {
            inAutoZoom = false;
            return;
        }

        double frac = double(log10(scale) / log10(destScale));
        view->maxIters = max(int(maxIters*frac), 250);
        view->center = center;
        view->setScale(view->getScale() * rate);
    }

    BigComplex center;
    BigFloat destScale;
    int maxIters;
    double rate = 0.96;
};
