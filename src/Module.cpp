#include "Module.h"

#include <cstdlib>
#include <cmath>
#include <stdexcept>

static void linearBackward(Tensor* self) {
    Tensor* x = self->getParent(0);
    Tensor* W = self->getParent(1);
    Tensor* b = self->getParent(2);

    if (x && x->needGrad()) {
        for (int i = 0; i < x->rowCount(); i++) {
            for (int k = 0; k < x->colCount(); k++) {
                double grad = 0.0;
                for (int j = 0; j < self->colCount(); j++) {
                    grad += self->getGrad(i * self->colCount() + j) * W->at(k, j);
                }
                x->addGrad(i * x->colCount() + k, grad);
            }
        }
    }

    if (W && W->needGrad()) {
        for (int k = 0; k < W->rowCount(); k++) {
            for (int j = 0; j < W->colCount(); j++) {
                double grad = 0.0;
                for (int i = 0; i < self->rowCount(); i++) {
                    grad += x->at(i, k) * self->getGrad(i * self->colCount() + j);
                }
                W->addGrad(k * W->colCount() + j, grad);
            }
        }
    }

    if (b && b->needGrad()) {
        for (int j = 0; j < b->colCount(); j++) {
            double grad = 0.0;
            for (int i = 0; i < self->rowCount(); i++) {
                grad += self->getGrad(i * self->colCount() + j);
            }
            b->addGrad(j, grad);
        }
    }
}

Linear::Linear(int inFeatures, int outFeatures)
    : W(inFeatures, outFeatures, 0.0, true),
      b(1, outFeatures, 0.0, true) {
    if (inFeatures <= 0 || outFeatures <= 0) {
        throw std::invalid_argument("Linear dimensions must be positive");
    }

    static bool seeded = false;
    if (!seeded) {
        std::srand(42);
        seeded = true;
    }

    double limit = std::sqrt(6.0 / (inFeatures + outFeatures));

    for (int i = 0; i < W.rowCount(); i++) {
        for (int j = 0; j < W.colCount(); j++) {
            double r = std::rand() * 1.0 / RAND_MAX;
            W.at(i, j) = (2.0 * r - 1.0) * limit;
        }
    }
}

Tensor Linear::forward(Tensor& x) {
    if (x.colCount() != W.rowCount()) {
        throw std::invalid_argument("Linear forward requires x.colCount() == inFeatures");
    }

    bool req = x.needGrad() || W.needGrad() || b.needGrad();
    Tensor out(x.rowCount(), W.colCount(), 0.0, req);

    for (int i = 0; i < x.rowCount(); i++) {
        for (int j = 0; j < W.colCount(); j++) {
            double sum = b.at(0, j);
            for (int k = 0; k < x.colCount(); k++) {
                sum += x.at(i, k) * W.at(k, j);
            }
            out.at(i, j) = sum;
        }
    }

    out.addParent(&x);
    out.addParent(&W);
    out.addParent(&b);
    if (req) {
        out.setBackwardFunc(linearBackward);
    }
    return out;
}

Tensor& Linear::weight() {
    return W;
}

Tensor& Linear::bias() {
    return b;
}

void Linear::zeroGrad() {
    W.zeroGrad();
    b.zeroGrad();
}
