/******************************************************************
  File name: NN.c
  Auther：SDL                                                         
  Date：2019.4.4
 ******************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define ALPHA 5                 // 学习率
#define EPSILON 0.32            // 用于初始化参数矩阵
#define SAMPAL 500              // 样本个数
#define INPUT_LAYER 40          // 输入层
#define HIDDEN_LAYER 18         // 隐藏层
#define OUTPUT_LAYER 10         // 输出层
#define ITERATION 20000         // 迭代次数    
//函数声明
struct Data read_data(char* path);
double sigmoid(double z);
double sigmoid_gradient(double z);
struct Theata init_theata();
struct Theata nn_cost_fun(struct Data data, struct Theata t);
void predict(struct Data data, struct Theata t);
/* ------------------------------------------------------------------------------------------------- */
// 此结构体用于声明存放样本数据和两种用不同规格表示的标签的变量
struct Data{ 
    double X[SAMPAL][INPUT_LAYER + 1];
    int label[SAMPAL][OUTPUT_LAYER];
    int y[SAMPAL];
};
// 此结构体用于声明存放参数矩阵的变量和代价值
struct Theata{
    double theata1[HIDDEN_LAYER][INPUT_LAYER + 1];
    double theata2[OUTPUT_LAYER][HIDDEN_LAYER + 1];
    double J;
};

int main(){
    struct Data data = read_data("F:/Study/C/Workspace/stu.txt"); 
    struct Theata t = init_theata();
    printf("Start training...");
    // 记录训练开始的时间
    clock_t start_clock = clock();
    // 开始训练
    for(int i = 0; i < ITERATION; i++){
        t = nn_cost_fun(data, t);
        if(i % 200 == 0){
            printf("\nCost = %lf", t.J);
        }
    }
    // 打印训练消耗的时间
    printf("\ntime consuming: %g\n", (clock() - start_clock) / (double)CLOCKS_PER_SEC);
    system("pause");
    printf("start predcting...\n");  
    double h[SAMPAL][OUTPUT_LAYER];
    int pred[SAMPAL];
    struct Data testData = read_data("F:/Study/C/Workspace/tst.txt");
    predict(testData, t);
}

// 从文件中读取数据
struct Data read_data(char* path)
{
    struct Data data;
    int X_row = 0;
    int y_row = 0;
    int X_set = 0;
    int y_set = 0;
    // 获取文件流
    FILE *fp = fopen(path, "r"); 
    // 获取失败则中止程序
    if(fp == NULL){
        printf("Fail to read the file!");
        abort();
    }
    // 循环读入样本值和样本标签存放在数组中
    while(X_row < SAMPAL && y_row < SAMPAL){
        while(X_set < INPUT_LAYER){
            fscanf(fp, "%lf,", &data.X[X_row][X_set + 1]);         
            X_set++;
        }
        X_set = 0;
        X_row++;
        while(y_set < OUTPUT_LAYER){
            fscanf(fp, "%d,", &data.label[y_row][y_set]);
            y_set++;
        }
        y_set = 0;
        y_row++;
    }
    // 添加偏差单元
    for(int i = 0; i < SAMPAL; i++){
        data.X[i][0] = 1.0;
    }
    for(int i = 0; i < SAMPAL; i++){
        for(int j = 0; j < OUTPUT_LAYER; j++){
            if(data.label[i][j] == 1){
                data.y[i] = j;
                break;    
            }    
        }
    }
    // 关闭文件流
    fclose(fp);
    return data;
};

// S型函数
double sigmoid(double z){
    return 1.0 / (1.0 + exp(-z));
}

// S型函数的导数
double sigmoid_gradient(double z){
    return sigmoid(z) * (1.0 - sigmoid(z));
}

// 初始化参数矩阵Theata1和Theata2
struct Theata init_theata(){
    struct Theata t;
    for(int i = 0; i < HIDDEN_LAYER; i++){
        for(int j = 0; j < INPUT_LAYER + 1; j++){
            t.theata1[i][j] = ((double)rand() / (RAND_MAX + 1.0)) * (2 * EPSILON) - EPSILON;
            // printf("%e ", theata1[i][j]);
        }
    }

    for(int i = 0; i < OUTPUT_LAYER; i++){
        for(int j = 0; j < HIDDEN_LAYER + 1; j++){
            t.theata2[i][j] = ((double)rand() / (RAND_MAX + 1.0)) * (2 * EPSILON) - EPSILON;
            // printf("%e ", theata2[i][j]);
        }
    }
    return t;
}

// 用于执行前馈网络和反向传播神经网络
struct Theata nn_cost_fun(struct Data data, struct Theata t){
/* --------------------------------Part1 前馈网络，计算代价值-------------------------------------- */    
    double a1[SAMPAL][INPUT_LAYER];
    double a2[SAMPAL][HIDDEN_LAYER + 1];
    double z2[SAMPAL][HIDDEN_LAYER + 1];
    double a3[SAMPAL][OUTPUT_LAYER];
    double temp;
    for(int i = 0; i < SAMPAL; i++){
        for(int j = 0; j < INPUT_LAYER + 1; j++){
            a1[i][j] = data.X[i][j];
        }
    }
    for(int i = 0; i < SAMPAL; i++){
        for(int j = 0; j < HIDDEN_LAYER; j++){
            temp = 0.0;
            for(int k = 0; k < INPUT_LAYER + 1; k++){
                temp += a1[i][k] * t.theata1[j][k];
            }
            z2[i][j + 1] = temp;
        }
    }
    for(int i = 0; i < SAMPAL; i++){
        for(int j = 0; j < HIDDEN_LAYER + 1; j++){
            a2[i][j] = sigmoid(z2[i][j]);
        }
    }
    for(int i = 0; i < SAMPAL; i++){
        a2[i][0] = 1.0;
    }
    for(int i = 0; i < SAMPAL; i++){
        for(int j = 0; j < OUTPUT_LAYER; j++){
            temp = 0.0;
            for(int k = 0; k < HIDDEN_LAYER + 1; k++){
                temp += a2[i][k] * t.theata2[j][k];
            }
            a3[i][j] = temp;
        }
    }
    double h[SAMPAL][OUTPUT_LAYER];
    for(int i = 0; i < SAMPAL; i++){
        for(int j = 0; j < OUTPUT_LAYER; j++){
            h[i][j] = sigmoid(a3[i][j]);
        }
    }

    t.J = 0.0;
    for(int i = 0; i < SAMPAL; i++){
        for(int j = 0; j < OUTPUT_LAYER; j++){
            t.J += data.label[i][j] * log(h[i][j]) + (1 - data.label[i][j]) * log(1 - h[i][j]);
        }
    }
    t.J = t.J / (-SAMPAL);
/*---------------------------------Part2 反向传播，计算新的参数矩阵----------------------------------*/
    double d3[SAMPAL][OUTPUT_LAYER];
    double d2[SAMPAL][OUTPUT_LAYER];
    for(int i = 0; i < SAMPAL; i++){
        for(int j = 0; j < OUTPUT_LAYER; j++){
            d3[i][j] = h[i][j] - data.label[i][j];
        }
    }
    for(int i = 0; i < SAMPAL; i++){
        for(int j = 0; j < HIDDEN_LAYER; j++){
            temp = 0;
            for(int k = 0; k < OUTPUT_LAYER; k++){
                temp += d3[i][k] * t.theata2[k][j + 1];              
            }
            d2[i][j] = temp;
            d2[i][j] *= sigmoid_gradient(z2[i][j + 1]);
        }
    }
    double Theata1_grad[HIDDEN_LAYER][INPUT_LAYER + 1];
    double Theata2_grad[OUTPUT_LAYER][HIDDEN_LAYER + 1];
    for(int i = 0; i < OUTPUT_LAYER; i++){
        for(int j = 0; j < HIDDEN_LAYER + 1; j++){
            temp = 0;
            for(int k = 0; k < SAMPAL; k++){
                temp += d3[k][i] * a2[k][j];
            }
            Theata2_grad[i][j] = temp / SAMPAL;
            t.theata2[i][j] -= ALPHA * Theata2_grad[i][j];
        }
    }
    for(int i = 0; i < HIDDEN_LAYER; i++){
        for(int j = 0; j < INPUT_LAYER + 1; j++){
            temp = 0;
            for(int k = 0; k < SAMPAL; k++){
                temp += d2[k][i] * a1[k][j];
            }
            Theata1_grad[i][j] = temp / SAMPAL;
            t.theata1[i][j] -= ALPHA * Theata1_grad[i][j];
        }
    }
    // for(int i = 0; i < HIDDEN_LAYER; i++){
    //     for(int j = 0; j < INPUT_LAYER; j++){
    //         t.theata1[i][j] -= ALPHA * Theata1_grad[i][j];
    //     }
    // }
    // for(int i = 0; i < OUTPUT_LAYER; i++){
    //     for(int j = 0; j < HIDDEN_LAYER + 1; j++){
    //         t.theata2[i][j] -= ALPHA * Theata2_grad[i][j];
    //     } 
    // }
    return t;
}
// 用于预测测试集中的数据
void predict(struct Data data, struct Theata t){
    double a1[SAMPAL][INPUT_LAYER];
    double a2[SAMPAL][HIDDEN_LAYER + 1];
    double z2[SAMPAL][HIDDEN_LAYER + 1];
    double a3[SAMPAL][OUTPUT_LAYER];
    double temp;
    for(int i = 0; i < SAMPAL; i++){
        for(int j = 0; j < INPUT_LAYER + 1; j++){
            a1[i][j] = data.X[i][j];
        }
    }
    for(int i = 0; i < SAMPAL; i++){
        for(int j = 0; j < HIDDEN_LAYER; j++){
            temp = 0.0;
            for(int k = 0; k < INPUT_LAYER + 1; k++){
                temp += a1[i][k] * t.theata1[j][k];
            }
            z2[i][j + 1] = temp;
        }
    }
    for(int i = 0; i < SAMPAL; i++){
        for(int j = 0; j < HIDDEN_LAYER + 1; j++){
            a2[i][j] = sigmoid(z2[i][j]);
        }
    }
    for(int i = 0; i < SAMPAL; i++){
        a2[i][0] = 1.0;
    }
    for(int i = 0; i < SAMPAL; i++){
        for(int j = 0; j < OUTPUT_LAYER; j++){
            temp = 0.0;
            for(int k = 0; k < HIDDEN_LAYER + 1; k++){
                temp += a2[i][k] * t.theata2[j][k];
            }
            a3[i][j] = temp;
        }
    }
    double h[SAMPAL][OUTPUT_LAYER];
    for(int i = 0; i < SAMPAL; i++){
        for(int j = 0; j < OUTPUT_LAYER; j++){
            h[i][j] = sigmoid(a3[i][j]);
        }
    }
    int pred[SAMPAL];
    for(int i = 0; i < SAMPAL; i++){
        temp = 0.0;
        for(int j = 0; j < OUTPUT_LAYER; j++){
            if(temp <= h[i][j]){
                temp = h[i][j];
                pred[i] = j;
            }
        }
    }
    // 匹配正确的个数
    double right = 0.0;
    for(int i = 0; i < SAMPAL; i++){
        if(pred[i] == data.y[i]){
            right += 1;
        }
    }
    // 打印匹配成功率
    printf("\nTesting Set Accuracy: %.2lf%%", (right / SAMPAL) * 100);
}