#include <iostream>
#include <random>
#include <chrono>
#include <fstream>
#include <cstring>

using namespace std;
using namespace std::chrono;

#define MATRIXSIZE 1024
#define DATAMAX 1000
#define T int

typedef T Matrix [MATRIXSIZE][MATRIXSIZE];
typedef T (*Matrixp)[MATRIXSIZE];

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
// void MatrixMuitply() {
//       for(int i = 0; i < MATRIXSIZE; ++i) {
//             for(int j = 0; j < MATRIXSIZE; ++j) {
//                   for(int k = 0; k < MATRIXSIZE; ++k) {
//                         result[i][j] += matrix1[i][k] * matrix2[k][j];
//                   }
//             } 
//       }
// };

// IKJ顺序
void MatrixMuitply() {
      for(int i = 0; i < MATRIXSIZE; ++i) {
            for(int k = 0; k < MATRIXSIZE; ++k) {
                  for(int j = 0; j < MATRIXSIZE; ++j) {
                        result[i][j] += matrix1[i][k] * matrix2[k][j];
                  }
            } 
      }
};

void MatrixMuitply1() {
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


// Strassen
void strassen(T **A, T **B, T **C, int size);

int main(int argc, char** argv) {
      // int seed = static_cast<int>(time(nullptr));
      int seed = 0;
      pseudoRandNumGen = mt19937(seed);

      GenData(matrix1);
      GenData(matrix2);

      ofstream input1("./data/smallmatrix/matrix1.txt"), input2("./data/smallmatrix/matrix2.txt");
      ofstream output("./data/smallmatrix/output.txt");

      if(!output.is_open())
            cout << "Error." << endl;

      cout << "Save matrix1." << endl;
      DisplayMatrix(input1, matrix1);
      cout << "Save matrix2." << endl;
      DisplayMatrix(input2, matrix2);

      // 初始化result矩阵为0
      for(int i = 0; i < MATRIXSIZE; ++i) {
            memset(result[i], 0, sizeof(T) * 32);
      }

      steady_clock::time_point StartTime = steady_clock::now();
      MatrixMuitply();
      steady_clock::time_point EndTime = steady_clock::now();

      uint64_t duration = duration_cast<milliseconds>(EndTime - StartTime).count();

      StartTime = steady_clock::now();
      MatrixMuitply1();
      EndTime = steady_clock::now();

      uint64_t duration1 = duration_cast<milliseconds>(EndTime - StartTime).count();

      cout << "Save result." << endl;
      DisplayMatrix(output, result);

      cout << "MatrixMuitply Run Time: " << duration << " ms" << endl;
      cout << "MatrixMuitply1 Run Time: " << duration1 << " ms" << endl;
      

      return 0;
}