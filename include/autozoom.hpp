#include "complex.hpp"
#include "ext_float.hpp"
#include "renderer.hpp"

struct AutoZoom {
    void init(BaseRenderer *view) {
        view->setScale(destScale);
        view->center = center;
        view->maxIters = maxIters;
        view->preprocess();
        view->resetScale();
        view->maxIters = 500;
    }

    bool update(BaseRenderer *view) {
        BigFloat scale = view->getScale();
        if (destScale >= scale) {
            return false;
        }

        BigFloat nextScale = view->getScale() * rate;
        double frac = double(log10(scale) / log10(destScale));

        view->maxIters = (nextScale > 1e-5 ? 500 : maxIters);
        view->center = center;
        view->setScale(std::max(nextScale, destScale));
        return true;
    }

    BigComplex center;
    BigFloat destScale;
    int maxIters;
    double rate = 0.85;
};
