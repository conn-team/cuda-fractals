#include "complex.hpp"
#include "ext_float.hpp"
#include "renderer.hpp"

struct AutoZoom {
    void init(BaseRenderer *view) {
        view->reset();
        view->center = center;
        view->maxIters = maxIters;
    }

    bool update(BaseRenderer *view) {
        BigFloat scale = view->getScale();
        if (destScale >= scale) {
            return false;
        }

        BigFloat nextScale = view->getScale() * rate;
        double frac = double(log10(scale) / log10(destScale));

        //view->maxIters = max(int(maxIters*frac), 250);
        view->center = center;
        view->setScale(std::max(nextScale, destScale));
        return true;
    }

    BigComplex center;
    BigFloat destScale;
    int maxIters;
    double rate = 0.85;
};
