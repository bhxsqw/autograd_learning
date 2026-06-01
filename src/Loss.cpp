#include <iomanip>
#include <iostream>

#include "Loss.h"
#include "Ops.h"

static void printLoss(int epoch, double value) {
    std::cout << "epoch " << std::setw(5) << epoch
              << " loss = " << std::fixed << std::setprecision(6)
              << value << std::endl;
}

TrainResult trainLoss(
    Linear& layer,
    Tensor& x,
    Tensor& target,
    SGD& optimizer,
    int epochs,
    int logInterval
) {
    TrainResult result;
    result.finalLoss = 0.0;
    result.epochs = epochs;

    for (int epoch = 1; epoch <= epochs; epoch++) {
        optimizer.zeroGrad();

        Tensor pred = layer.forward(x);
        Tensor loss = mseLoss(pred, target);

        result.finalLoss = loss.value();

        loss.backward();
        optimizer.step();

        if (logInterval > 0 && (epoch % logInterval == 0 || epoch == 1)) {
            printLoss(epoch, loss.value());
        }
    }

    return result;
}

TrainResult trainLoss(
    Linear& layer1,
    Linear& layer2,
    Tensor& x,
    Tensor& target,
    SGD& optimizer,
    int epochs,
    int logInterval
) {
    TrainResult result;
    result.finalLoss = 0.0;
    result.epochs = epochs;

    for (int epoch = 0; epoch <= epochs; epoch++) {
        optimizer.zeroGrad();

        Tensor h = layer1.forward(x);
        Tensor a = sigmoid(h);
        Tensor z = layer2.forward(a);
        Tensor pred = sigmoid(z);
        Tensor loss = mseLoss(pred, target);

        result.finalLoss = loss.value();

        loss.backward();
        optimizer.step();

        if (logInterval > 0 && (epoch % logInterval == 0 || epoch == epochs)) {
            printLoss(epoch, loss.value());
        }
    }

    return result;
}
