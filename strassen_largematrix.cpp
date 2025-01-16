#include <iostream>
#include <random>
#include <chrono>
#include <fstream>
#include <cstring>
#include <filesystem>

using namespace std;
using namespace std::chrono;

#define MATRIXSIZE 4096
#define DATAMAX 1000
#define T long long 

static mt19937 pseudoRandNumGen;
T **matrix1, **matrix2;
T **result;

T Rand() { return pseudoRandNumGen() % DATAMAX; };

void DisplayMatrix(ostream& os, T **matrix) {
      for(int i = 0; i < MATRIXSIZE; ++i) {
            for(int j = 0; j < MATRIXSIZE; ++j) {
                  os << matrix[i][j] << ' ';
            }
            os << endl;
      }
};

void GenData(T **matrix) {
      for(int i = 0; i < MATRIXSIZE; ++i) {
            for(int j = 0; j < MATRIXSIZE; ++j) {
                  T t = Rand() - Rand();

                  matrix[i][j] = t;
            }
      }
};

void MatrixMuitply(T **A, T **B, T **C, int size);
// Strassen
void Strassen(T **A, T **B, T **C, int size);

steady_clock::time_point StartTime, EndTime;
int main(int argc, char** argv) {
      // int seed = static_cast<int>(time(nullptr));
      int seed = 0;
      pseudoRandNumGen = mt19937(seed);

      matrix1 = new T* [MATRIXSIZE];
      matrix2 = new T* [MATRIXSIZE];
      result = new T* [MATRIXSIZE];
      for(int i = 0; i < MATRIXSIZE; ++i) {
            matrix1[i] = new T [MATRIXSIZE];
            matrix2[i] = new T [MATRIXSIZE];
            result[i] = new T [MATRIXSIZE];
      }

      GenData(matrix1);
      GenData(matrix2);

      ofstream input1(".\\data\\largemartix\\martix1.txt"), input2(".\\data\\largematrix\\matrix2.txt");
      ofstream output("C:\\Users\\chena\\Desktop\\Martix  Muitply\\data\\largemartix\\output.txt");

      cout << "Save matrix1." << endl;
      DisplayMatrix(input1, matrix1);
      cout << "Save matrix2." << endl;
      DisplayMatrix(input2, matrix2);

      // 初始化result矩阵为0
      for(int i = 0; i < MATRIXSIZE; ++i) {
            memset(result[i], 0, sizeof(T) * MATRIXSIZE);
      }

      cout << "Matrix Mutiply" << endl;
      StartTime = steady_clock::now();
      MatrixMuitply(matrix1, matrix2, result, MATRIXSIZE);
      EndTime = steady_clock::now();
      uint64_t duration1 = duration_cast<milliseconds>(EndTime - StartTime).count();

      for(int i = 0; i < MATRIXSIZE; ++i) {
            memset(result[i], 0, sizeof(T) * MATRIXSIZE);
      }

      cout << "Strassen" << endl;
      Strassen(matrix1, matrix2, result, MATRIXSIZE);

      uint64_t duration2 = duration_cast<milliseconds>(EndTime - StartTime).count();

      cout << "Save result." << endl;
      DisplayMatrix(output, result);

      cout << "Matrix Mutiply Run Time: " << duration1 << " ms" << endl;
      cout << "Strassen Run Time: " << duration2 << " ms" << endl;

      for(int i = 0; i < MATRIXSIZE; ++i) {
            delete matrix1[i];
            delete matrix2[i];
            delete result[i]; 
      }

      return 0;
}

void MatrixAdd(T **A, T **B, T **C, int size) {
      for(int i = 0; i < size; ++i) {
            for(int j = 0; j < size; ++j) {
                  C[i][j] = A[i][j] + B[i][j];
            }
      }
};

void MatrixSub(T **A, T **B, T **C, int size) {
      for(int i = 0; i < size; ++i) {
            for(int j = 0; j < size; ++j) {
                  C[i][j] = A[i][j] - B[i][j];
            }
      }
};

void MatrixMuitply(T **A, T **B, T **C, int size) {
      for(int i = 0; i < size; ++i) {
            for(int k = 0; k < size; ++k) {
                  for(int j = 0; j < size; ++j) {
                        C[i][j] += A[i][k] * B[k][j];
                  }
            } 
      }
};

void Strassen(T **A, T **B, T **C, int size) {
      if(size == 2)  {
            int x1 = (A[0][0] + A[1][1]) * (B[0][0] + B[1][1]);
            int x2 = (A[1][0] + A[1][1]) * B[0][0];
            int x3 = A[0][0] * (B[0][1] - B[1][1]);
            int x4 = A[1][1] * (B[1][0] - B[0][0]);
            int x5 = (A[0][0] + A[0][1]) * B[1][1];
            int x6 = (A[1][0] - A[0][0]) * (B[0][0] + B[0][1]);
            int x7 = (A[0][1] - A[1][1]) * (B[1][0] + B[1][1]);

            C[0][0] = x1 + x4 - x5 + x7;
            C[0][1] = x3 + x5;
            C[1][0] = x2 + x4;
            C[1][1] = x1 + x3 - x2 + x6;

            return ;
      }
      
      size /= 2;

      T **A11 = new T*[size];
      T **A12 = new T*[size];
      T **A21 = new T*[size];
      T **A22 = new T*[size];
      T **B11 = new T*[size];
      T **B12 = new T*[size];
      T **B21 = new T*[size];
      T **B22 = new T*[size];

      T **X1 = new T *[size];
      T **X2 = new T *[size];
      T **X3 = new T *[size];
      T **X4 = new T *[size];
      T **X5 = new T *[size];
      T **X6 = new T *[size];
      T **X7 = new T *[size];

      T **T1 = new T *[size];
      T **T2 = new T *[size];

      for(int i = 0; i < size; ++i) {
            A11[i] = new T[size];
            A12[i] = new T[size];
            A21[i] = new T[size];
            A22[i] = new T[size];
            B11[i] = new T[size];
            B12[i] = new T[size];
            B21[i] = new T[size];
            B22[i] = new T[size];  


            X1[i] = new T [size];
            X2[i] = new T [size];
            X3[i] = new T [size];
            X4[i] = new T [size];
            X5[i] = new T [size];
            X6[i] = new T [size];
            X7[i] = new T [size];

            T1[i] = new T [size];
            T2[i] = new T [size];
      }

      for(int i = 0; i < size; ++i) {
            for(int j = 0; j < size; ++j) {
                  A11[i][j] = A[i][j];
                  A12[i][j] = A[i][j + size];
                  A21[i][j] = A[i + size][j];
                  A22[i][j] = A[i + size][j + size]; 

                  B11[i][j] = B[i][j];
                  B12[i][j] = B[i][j + size];
                  B21[i][j] = B[i + size][j];
                  B22[i][j] = B[i + size][j + size]; 
            }
      }

      StartTime = steady_clock::now();
      MatrixAdd(A11, A22, T1, size);
      MatrixAdd(B11, B22, T2, size);
      MatrixMuitply(T1, T2, X1, size);

      MatrixAdd(A21, A22, T1, size);
      MatrixMuitply(T1, B11, X2, size);

      MatrixSub(B12, B22, T1, size);
      MatrixMuitply(A11, T1, X3, size);

      MatrixSub(B21, B11, T1, size);
      MatrixMuitply(A22, T1, X4, size);

      MatrixAdd(A11, A12, T1, size);
      MatrixMuitply(T1, B22, X5, size);

      MatrixSub(A21, A11, T1, size);
      MatrixAdd(B11, B12, T2, size);
      MatrixMuitply(T1, T2, X6, size);

      MatrixSub(A12, A22, T1, size);
      MatrixAdd(B21, B22, T2, size);
      MatrixMuitply(T1, T2, X7, size);

      for(int i = 0; i < size; i++) {
            for(int j = 0; j < size; ++j) {
                  C[i][j] = X1[i][j] + X4[i][j] - X5[i][j] + X7[i][j];
                  C[i][j + size] = X3[i][j] + X5[i][j];
                  C[i + size][j] = X2[i][j] + X4[i][j];
                  C[i + size][j + size] = X1[i][j] + X3[i][j] - X2[i][j] + X6[i][j];
            }
      }
      EndTime = steady_clock::now();

      for(int i = 0; i < size; ++i) {
            delete A11[i];
            delete A12[i];
            delete A21[i];
            delete A22[i];
            delete B11[i];
            delete B12[i];
            delete B21[i];
            delete B22[i];  


            delete X1[i];
            delete X2[i];
            delete X3[i];
            delete X4[i];
            delete X5[i];
            delete X6[i];
            delete X7[i];

            delete T1[i];
            delete T2[i];
      }      
};