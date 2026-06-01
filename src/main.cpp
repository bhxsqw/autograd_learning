#include <iostream>
#include <cstring>
#include "Tensor.h"
#include "Ops.h"
#include "Module.h"
#include "Optimizer.h"

using namespace std;

void runBasic() {
    Tensor x(2.0);
    Tensor w(3.0);
    Tensor b(1.0);
    Tensor target(10.0);

    Tensor wx = w * x;
    Tensor y = wx + b;
    Tensor diff = y - target;
    Tensor loss = diff * diff;

    loss.backward();

    cout << "===== Basic AutoGrad Demo =====" << endl;
    cout << "loss = " << loss.value() << endl;
    cout << "grad of w = " << w.gradValue() << endl;
    cout << "grad of b = " << b.gradValue() << endl;
}

void runStandard() {
    cout << "===== Standard Linear Regression Demo =====" << endl;
    cout << "TODO: train y = w1*x1 + w2*x2 + b" << endl;
    //1. 准备训练数据
    const double true_w1 = 2.5;
    const double true_w2 = -1.2;
    const double true_b = 0.5;

    const int num_samples = 200;
    Tensor X(num_samples, 2, 0.0, false);   // 输入特征，不需要梯度
    Tensor y_true(num_samples, 1, 0.0, false); // 真实标签，不需要梯度

    srand(42);
    for (int i = 0; i < num_samples; ++i) {
        double x1 = (rand() % 100) / 100.0;
        double x2 = (rand() % 100) / 100.0;
        X.at(i, 0) = x1;
        X.at(i, 1) = x2;
        y_true.at(i, 0) = true_w1 * x1 + true_w2 * x2 + true_b;
    }

    //2. 定义可训练参数
    Tensor W(2, 1, 0.0, true);   // 权重 [2,1]，需要梯度
    Tensor b(1, 1, 0.0, true);   // 偏置 [1,1]，需要梯度

    //3. 创建优化器并注册参数
    SGD optimizer(0.02);
    optimizer.addParam(&W);
    optimizer.addParam(&b);

    //4. 训练循环
    int epochs = 500;
    cout << "\n[训练循环] 开始训练，共 " << epochs << " 轮迭代..." << endl;

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        //A: 清零上一轮的梯度
        optimizer.zeroGrad();
        //B: 前向传播
        Tensor matmul_res = matmul(X, W);
        Tensor y_pred = matmul_res + b;
        //C: 计算损失
        Tensor loss = mseLoss(y_pred, y_true);
        //D: 反向传播，计算 W 和 b 的梯度
        loss.backward();
        //E: 更新参数（梯度下降）
        optimizer.step();

        // 打印进度
        if (epoch % 50 == 0 || epoch == 1) {
            cout << "Epoch " << epoch << " | loss = " << loss.value() << endl;
        }
    }

    cout << "训练完成" << endl;
    //5. 验证部分（仅调试用，可删除）
    cout << "\n===== 验证部分（仅用于调试）=====" << endl;
    cout << "训练得到的参数：" << endl;
    cout << "w1 = " << W.at(0, 0) << ", w2 = " << W.at(1, 0) << ", b = " << b.at(0, 0) << endl;
    cout << "真实参数：" << endl;
    cout << "w1 = " << true_w1 << ", w2 = " << true_w2 << ", b = " << true_b << endl;

    double err_w1 = abs(W.at(0, 0) - true_w1) / true_w1;
    double err_w2 = abs(W.at(1, 0) - true_w2) / true_w2;
    double err_b = abs(b.at(0, 0) - true_b) / true_b;
    cout << "相对误差：w1=" << err_w1 * 100 << "%, w2=" << err_w2 * 100 << "%, b=" << err_b * 100 << "%" << endl;

    if (err_w1 < 0.05 && err_w2 < 0.05 && err_b < 0.05) {
        cout << "验证通过：梯度下降正确实现了参数收敛。" << endl;
    }
    else {
        cout << "验证失败：参数未收敛，请检查学习率或训练轮数。" << endl;
    }
    cout << "=============================================" << endl;

}

void runChallenge() {
    cout << "===== Challenge XOR Neural Network Demo =====" << endl;
    cout << "TODO: train 2-4-1 network for XOR" << endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cout << "Usage: tinytensor.exe [basic | standard | challenge | all]" << endl;
        return 0;
    }

    string mode = argv[1];

    if (mode == "basic") {
        runBasic();
    } else if (mode == "standard") {
        runStandard();
    } else if (mode == "challenge") {
        runChallenge();
    } else if (mode == "all") {
        runBasic();
        runStandard();
        runChallenge();
    } else {
        cout << "Unknown mode." << endl;
    }

    return 0;
}
