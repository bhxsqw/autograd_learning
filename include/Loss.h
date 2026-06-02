#ifndef LOSS_H
#define LOSS_H

#include "Module.h"
#include "Optimizer.h"
#include "Tensor.h"

struct TrainResult {
    double finalLoss;
    int epochs;
};

TrainResult trainLoss(
    Linear& layer,
    Tensor& x,
    Tensor& target,
    SGD& optimizer,
    int epochs,
    int logInterval = 0//每隔多少轮打印一次 loss
);

TrainResult trainLoss(
    Linear& layer1,
    Linear& layer2,
    Tensor& x,
    Tensor& target,
    SGD& optimizer,
    int epochs,
    int logInterval = 0//每隔多少轮打印一次 loss
);

#endif
