#include <iostream>
#include <random>
#include <chrono>
#include <fstream>
#include <cstring>

using namespace std;
using namespace std::chrono;

#define MATRIXSIZE 4096
#define DATAMAX 1000
#define T long long

typedef T Matrix [MATRIXSIZE][MATRIXSIZE];
typedef T (*Matrixp)[MATRIXSIZE];
typedef void (*func)();

static mt19937 pseudoRandNumGen;
Matrix matrix1, matrix2;
Matrix result;

T Rand() { return pseudoRandNumGen() % DATAMAX; };

void DisplayMatrix(ostream& os, Matrixp matrix) {
      for(int i = 0; i < MATRIXSIZE; ++i) {
            for(int j = 0; j < MATRIXSIZE; ++j) {
                  os << matrix[i][j] << ' ';
            }
            os << endl;
      }
};

void GenData(Matrixp matrix) {
      for(int i = 0; i < MATRIXSIZE; ++i) {
            for(int j = 0; j < MATRIXSIZE; ++j) {
                  T t = Rand();

                  matrix[i][j] = t;
            }
      }
};

// IJK顺序
void MatrixMuitply1() {
      for(int i = 0; i < MATRIXSIZE; ++i) {
            for(int j = 0; j < MATRIXSIZE; ++j) {
                  for(int k = 0; k < MATRIXSIZE; ++k) {
                        result[i][j] += matrix1[i][k] * matrix2[k][j];
                  }
            } 
      }
};

// IKJ顺序
void MatrixMuitply2() {
      for(int i = 0; i < MATRIXSIZE; ++i) {
            for(int k = 0; k < MATRIXSIZE; ++k) {
                  for(int j = 0; j < MATRIXSIZE; ++j) {
                        result[i][j] += matrix1[i][k] * matrix2[k][j];
                  }
            } 
      }
};

void MatrixMuitply3() {
      const int BlockSize = 32;

      for(int ii = 0; ii < MATRIXSIZE; ii += BlockSize) {
            for(int jj = 0; jj < MATRIXSIZE; jj += BlockSize) {
                  for(int kk = 0; kk < MATRIXSIZE; kk += BlockSize) {
                        for(int i = ii; i < ii + BlockSize; ++i) {
                              for(int k = kk; k < kk + BlockSize; ++k) {
                                    for(int j = jj; j < jj + BlockSize; ++j) {
                                          result[i][j] += matrix1[i][k] * matrix2[k][j];
                                    }
                              }
                        }
                  }
            }
      }
};

void MatrixMuitply4() {
    for (int i = 0; i < MATRIXSIZE; ++i) {
        for (int j = 0; j < MATRIXSIZE; ++j) {
            int sum = 0; // 使用局部变量存储累加结果
            for (int k = 0; k < MATRIXSIZE; k += 8) {
                sum += matrix1[i][k]     * matrix2[k][j];
                sum += matrix1[i][k + 1] * matrix2[k + 1][j];
                sum += matrix1[i][k + 2] * matrix2[k + 2][j];
                sum += matrix1[i][k + 3] * matrix2[k + 3][j];
                sum += matrix1[i][k + 4] * matrix2[k + 4][j];
                sum += matrix1[i][k + 5] * matrix2[k + 5][j];
                sum += matrix1[i][k + 6] * matrix2[k + 6][j];
                sum += matrix1[i][k + 7] * matrix2[k + 7][j];           
            }
            result[i][j] = sum; // 将结果存回 result[i][j]
        }
    }
}

// Strassen
void strassen(T **A, T **B, T **C, int size);

// 调用函数
uint64_t Work(func f, ostream & os) {
      // 初始化result矩阵为0
      for(int i = 0; i < MATRIXSIZE; ++i) {
            memset(result[i], 0, sizeof(T) * 32);
      }

      steady_clock::time_point StartTime = steady_clock::now();
      MatrixMuitply1();
      steady_clock::time_point EndTime = steady_clock::now();

      DisplayMatrix(os, result);

      return duration_cast<microseconds>(EndTime - StartTime).count();
};

int main(int argc, char** argv) {
      // int seed = static_cast<int>(time(nullptr));
      int seed = 0;
      // int seed = 101;
      // int seed = 30120;
      // int seed = 50000123;
      // int seed = 732819634;
      pseudoRandNumGen = mt19937(seed);

      GenData(matrix1);
      GenData(matrix2);

      ofstream input1("./data/smallmatrix/matrix1.txt"), input2("./data/smallmatrix/matrix2.txt");
      ofstream output1("./data/smallmatrix/output1.txt");
      ofstream output2("./data/smallmatrix/output2.txt");
      ofstream output3("./data/smallmatrix/output3.txt");
      ofstream output4("./data/smallmatrix/output4.txt");

      cout << "Save matrix1." << endl;
      DisplayMatrix(input1, matrix1);
      cout << "Save matrix2." << endl;
      DisplayMatrix(input2, matrix2);

      uint64_t duration1 = Work(MatrixMuitply1, output1);
      // uint64_t duration2 = Work(MatrixMuitply2, output2);
      // uint64_t duration3 = Work(MatrixMuitply3, output3);
      // uint64_t duration4 = Work(MatrixMuitply4, output4);

      cout << "MatrixMuitply1  Run Time: " << duration1 << " ms" << endl;
      // cout << "MatrixMuitply2 Run Time: " << duration2 << " ms" << endl;
      // cout << "MatrixMuitply3 Run Time: " << duration3 << " ms" << endl;
      // cout << "MatrixMuitply4 Run Time: " << duration4 << " ms" << endl;
      

      return 0;
}