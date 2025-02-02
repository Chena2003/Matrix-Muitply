#include <iostream>
#include <random>
#include <chrono>
#include <fstream>
#include <cstring>
#include <immintrin.h>
#include <omp.h>
#include <cstdlib>

using namespace std;
using namespace std::chrono;

// 矩阵参数设置
#define MATRIXSIZE 4096 // 矩阵阶数
#define DATAMAX 100000000 // 矩阵数据规模
#define T long long // 数据类型

typedef void (*func)(T **, T **, T **, int );

static mt19937 pseudoRandNumGen;
alignas(32) T **matrix1;
alignas(32) T **matrix2;
alignas(32) T **result;

// 生成随机数
T Rand() { return pseudoRandNumGen() % DATAMAX; };

// 输出矩阵
void DisplayMatrix(ostream& os, T **matrix) {
      for(int i = 0; i < MATRIXSIZE; ++i) {
            for(int j = 0; j < MATRIXSIZE; ++j) {
                  os << matrix[i][j] << ' ';
            }
            os << endl;
      }
};

// 产生数据
void GenData(T **matrix) {
      for(int i = 0; i < MATRIXSIZE; ++i) {
            for(int j = 0; j < MATRIXSIZE; ++j) {
                  T t = Rand() - Rand();

                  matrix[i][j] = t;
            }
      }
};

// 调用函数
uint64_t Work(func f, T **A, T **B, T **C, int size, ostream & os) {
      // 初始化result矩阵为0
      for(int i = 0; i < size; ++i) {
            memset(C[i], 0, sizeof(T) * size);
      }

      // 矩阵乘法
      steady_clock::time_point StartTime = steady_clock::now();
      f(A, B, C, size);
      steady_clock::time_point EndTime = steady_clock::now();     

      // 输出矩阵
      DisplayMatrix(os, result);

      // 返回运行时间
      return duration_cast<milliseconds>(EndTime - StartTime).count();
};

// 校验函数
void CheckResult(istream &input, istream &ref) {
      input.seekg(0, ios::beg);
      ref.seekg(0, ios::beg);

      T t1, t2;
      bool flag = true;
      while((input >> t1) && (ref >> t2)) {
            if(t1 != t2) {
                  flag = false;

                  break;
            }
      }

      if(flag) cout << "Match Successsfully." << endl;
      else cout << "Match Unsuccesssfully." << endl;
}

// 普通矩阵乘法 ikj顺序
void MatrixMultiply(T **A, T **B, T **C, int size);

// Strassen
void Strassen(T **A, T **B, T **C, int size);

// SIMD
void MatrixMultiplyAVX(T **A, T **B, T **C, int size);

// AVX512
void MatrixMultiplyAVX512(T **A, T **B, T **C, int size);

int main(int argc, char** argv) {
      // int seed = static_cast<int>(time(nullptr));
      int seed = 0;
      pseudoRandNumGen = mt19937(seed);

      matrix1 = new T* [MATRIXSIZE];
      matrix2 = new T* [MATRIXSIZE];
      result = new T* [MATRIXSIZE];

      cout << reinterpret_cast<uintptr_t>(matrix1) % 32 << endl;
      cout << reinterpret_cast<uintptr_t>(matrix2) % 32 << endl;
      cout << reinterpret_cast<uintptr_t>(result) % 32 << endl;

      for(int i = 0; i < MATRIXSIZE; ++i) {
            matrix1[i] = (T*)aligned_alloc(64, MATRIXSIZE * sizeof(T));  // 32 字节对齐
            matrix2[i] = (T*)aligned_alloc(64, MATRIXSIZE * sizeof(T));  // 32 字节对齐
            result[i]  = (T*)aligned_alloc(64, MATRIXSIZE * sizeof(T));  // 32 字节对齐
      }

      GenData(matrix1);
      GenData(matrix2);

      ofstream input1("./data/smallmatrix/matrix1.txt"), input2("./data/smallmatrix/matrix2.txt");
      fstream output1("./data/smallmatrix/output1.txt", ios::in|ios::out);
      fstream output2("./data/smallmatrix/output2.txt", ios::in|ios::out);
      fstream output3("./data/smallmatrix/output3.txt", ios::in|ios::out);
      fstream output4("./data/smallmatrix/output4.txt", ios::in|ios::out);

      if(!input1.is_open() || !input2.is_open() || !output1.is_open() || !output2.is_open() || !output3.is_open() || !output4.is_open())
            cout << "Error" << endl;

      cout << "Save matrix1." << endl;
      DisplayMatrix(input1, matrix1);
      cout << "Save matrix2." << endl;
      DisplayMatrix(input2, matrix2);

      cout << "MatrixMultiply" << endl;
      uint64_t duration1 = Work(MatrixMultiply, matrix1, matrix2, result, MATRIXSIZE, output1);
      cout << "Strassen" << endl;
      uint64_t duration2 = Work(Strassen, matrix1, matrix2, result, MATRIXSIZE, output2);
      cout << "MatrixMultiplyAVX" << endl;
      uint64_t duration3 = Work(MatrixMultiplyAVX, matrix1, matrix2, result, MATRIXSIZE, output3);
      cout << "MatrixMultiplyAVX512" << endl;
      uint64_t duration4 = Work(MatrixMultiplyAVX512, matrix1, matrix2, result, MATRIXSIZE, output4);

      cout << "Matrix Mutiply Run Time: " << duration1 << " ms" << endl;
      cout << "Strassen Run Time: " << duration2 << " ms" << endl;
      cout << "MatrixMultiplyAVX Run Time: " << duration3 << " ms" << endl;
      cout << "MatrixMultiplyAVX512 Run Time: " << duration4 << " ms" << endl;


      CheckResult(output2, output1);
      CheckResult(output3, output1);
      CheckResult(output4, output1);

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

// 矩阵乘法 ijk顺序
// void MatrixMultiply(T **A, T **B, T **C, int size) {
//       for(int i = 0; i < size; ++i) {
//             for(int j = 0; j < size; ++j) {
//                   for(int k = 0; k < size; ++k) {
//                         C[i][j] += A[i][k] * B[k][j];
//                   }
//             } 
//       }
// };

// 矩阵乘法 ikj顺序
void MatrixMultiply(T **A, T **B, T **C, int size) {
      for(int i = 0; i < size; ++i) {
            for(int k = 0; k < size; ++k) {
                  for(int j = 0; j < size; ++j) {
                        C[i][j] += A[i][k] * B[k][j];
                  }
            } 
      }
};

// 矩阵乘法 循环展开
// void MatrixMultiply(T **A, T **B, T **C, int size) {
//       const int BlockSize = 32;

//       for(int ii = 0; ii < size; ii += BlockSize) {
//             for(int jj = 0; jj < size; jj += BlockSize) {
//                   for(int kk = 0; kk < size; kk += BlockSize) {
//                         for(int i = ii; i < ii + BlockSize; ++i) {
//                               for(int k = kk; k < kk + BlockSize; ++k) {
//                                     for(int j = jj; j < jj + BlockSize; ++j) {
//                                           C[i][j] += A[i][k] * B[k][j];
//                                     }
//                               }
//                         }
//                   }
//             }
//       }
// };

// Strassen算法
void Strassen(T **A, T **B, T **C, int size) {
      // 矩阵大小小于等于512时直接使用普通矩阵运算
      if(size <= 256) {
            MatrixMultiply(A, B, C, size);
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

      T **X1 = new T*[size];
      T **X2 = new T*[size];
      T **X3 = new T*[size];
      T **X4 = new T*[size];
      T **X5 = new T*[size];
      T **X6 = new T*[size];
      T **X7 = new T*[size];

      T **T1 = new T*[size];
      T **T2 = new T*[size];

      for(int i = 0; i < size; ++i) {
            A11[i] = new T[size];
            A12[i] = new T[size];
            A21[i] = new T[size];
            A22[i] = new T[size];
            B11[i] = new T[size];
            B12[i] = new T[size];
            B21[i] = new T[size];
            B22[i] = new T[size];  

            X1[i] = new T[size];
            X2[i] = new T[size];
            X3[i] = new T[size];
            X4[i] = new T[size];
            X5[i] = new T[size];
            X6[i] = new T[size];
            X7[i] = new T[size];

            T1[i] = new T[size];
            T2[i] = new T[size];

            memset(X1[i], 0, sizeof(T) * size);
            memset(X2[i], 0, sizeof(T) * size);
            memset(X3[i], 0, sizeof(T) * size);
            memset(X4[i], 0, sizeof(T) * size);
            memset(X5[i], 0, sizeof(T) * size);
            memset(X6[i], 0, sizeof(T) * size);
            memset(X7[i], 0, sizeof(T) * size);
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

      MatrixAdd(A11, A22, T1, size);
      MatrixAdd(B11, B22, T2, size);
      Strassen(T1, T2, X1, size);

      MatrixAdd(A21, A22, T1, size);
      Strassen(T1, B11, X2, size);

      MatrixSub(B12, B22, T1, size);
      Strassen(A11, T1, X3, size);

      MatrixSub(B21, B11, T1, size);
      Strassen(A22, T1, X4, size);

      MatrixAdd(A11, A12, T1, size);
      Strassen(T1, B22, X5, size);

      MatrixSub(A21, A11, T1, size);
      MatrixAdd(B11, B12, T2, size);
      Strassen(T1, T2, X6, size);

      MatrixSub(A12, A22, T1, size);
      MatrixAdd(B21, B22, T2, size);
      Strassen(T1, T2, X7, size);

      for(int i = 0; i < size; i++) {
            for(int j = 0; j < size; ++j) {
                  C[i][j] = X1[i][j] + X4[i][j] - X5[i][j] + X7[i][j];
                  C[i][j + size] = X3[i][j] + X5[i][j];
                  C[i + size][j] = X2[i][j] + X4[i][j];
                  C[i + size][j + size] = X1[i][j] + X3[i][j] - X2[i][j] + X6[i][j];
            }
      }

      // 不释放内存，加速矩阵乘法
      // for(int i = 0; i < size; ++i) {
      //       delete A11[i];
      //       delete A12[i];
      //       delete A21[i];
      //       delete A22[i];
      //       delete B11[i];
      //       delete B12[i];
      //       delete B21[i];
      //       delete B22[i];  


      //       delete X1[i];
      //       delete X2[i];
      //       delete X3[i];
      //       delete X4[i];
      //       delete X5[i];
      //       delete X6[i];
      //       delete X7[i];

      //       delete T1[i];
      //       delete T2[i];
      // }      
};

// SIMD AVX2指令
void MatrixMultiplyAVX(T **A, T **B, T **C, int size) {
    // 初始化结果矩阵 C 为 0
    for (int i = 0; i < size; ++i) {
        memset(C[i], 0, size * sizeof(T));
    }

    // 遍历矩阵 A 和 C
    for (int m = 0; m < size; m += 4) {
        for (int k = 0; k < size; k += 4) {
            // 定义 AVX 向量
            __m256i C0v, C1v, C2v, C3v;
            __m256i B0v;

            // 初始化累加器
            C0v = _mm256_setzero_si256();
            C1v = _mm256_setzero_si256();
            C2v = _mm256_setzero_si256();
            C3v = _mm256_setzero_si256();

            // 对矩阵 B 和 A 进行遍历，计算乘积并累加到 C[m:m+3][k:k+3]
            for (int n = 0; n < size; ++n) {
                // 加载 B[n][k:k+3] 到 AVX 向量
                B0v = _mm256_loadu_si256((__m256i*)&B[n][k]);

                // 加载 A[m:m+3][n] 到 AVX 向量
                __m256i vecA0 = _mm256_set1_epi64x(A[m][n]);
                __m256i vecA1 = _mm256_set1_epi64x(A[m + 1][n]);
                __m256i vecA2 = _mm256_set1_epi64x(A[m + 2][n]);
                __m256i vecA3 = _mm256_set1_epi64x(A[m + 3][n]);

                // 逐元素乘法并累加
                C0v = _mm256_add_epi64(C0v, _mm256_mullo_epi64(vecA0, B0v));
                C1v = _mm256_add_epi64(C1v, _mm256_mullo_epi64(vecA1, B0v));
                C2v = _mm256_add_epi64(C2v, _mm256_mullo_epi64(vecA2, B0v));
                C3v = _mm256_add_epi64(C3v, _mm256_mullo_epi64(vecA3, B0v));
            }

            // 将结果存回 C[m:m+3][k:k+3]
            _mm256_storeu_si256((__m256i*)&C[m][k], C0v);
            _mm256_storeu_si256((__m256i*)&C[m + 1][k], C1v);
            _mm256_storeu_si256((__m256i*)&C[m + 2][k], C2v);
            _mm256_storeu_si256((__m256i*)&C[m + 3][k], C3v);
        }
    }
}

// SIMD AVX512指令
void MatrixMultiplyAVX512(T **A, T **B, T **C, int size) {
    // 遍历矩阵 A 和 C
    for (int m = 0; m < size; m += 8) {
        for (int k = 0; k < size; k += 8) {
            // 定义 AVX-512 向量
            __m512i C0v, C1v, C2v, C3v, C4v, C5v, C6v, C7v;
            __m512i B0v;

            // 初始化累加器
            C0v = _mm512_setzero_si512();
            C1v = _mm512_setzero_si512();
            C2v = _mm512_setzero_si512();
            C3v = _mm512_setzero_si512();
            C4v = _mm512_setzero_si512();
            C5v = _mm512_setzero_si512();
            C6v = _mm512_setzero_si512();
            C7v = _mm512_setzero_si512();

            // 对矩阵 B 和 A 进行遍历，计算乘积并累加到 C[m:m+7][k:k+7]
            for (int n = 0; n < size; ++n) {
                // 加载 B[n][k:k+7] 到 AVX-512 向量
                B0v = _mm512_loadu_si512((__m512i*)&B[n][k]);

                // 加载 A[m:m+7][n] 到 AVX-512 向量
                __m512i vecA0 = _mm512_set1_epi64(A[m][n]);
                __m512i vecA1 = _mm512_set1_epi64(A[m + 1][n]);
                __m512i vecA2 = _mm512_set1_epi64(A[m + 2][n]);
                __m512i vecA3 = _mm512_set1_epi64(A[m + 3][n]);
                __m512i vecA4 = _mm512_set1_epi64(A[m + 4][n]);
                __m512i vecA5 = _mm512_set1_epi64(A[m + 5][n]);
                __m512i vecA6 = _mm512_set1_epi64(A[m + 6][n]);
                __m512i vecA7 = _mm512_set1_epi64(A[m + 7][n]);

                // 逐元素乘法并累加
                C0v = _mm512_add_epi64(C0v, _mm512_mullo_epi64(vecA0, B0v));
                C1v = _mm512_add_epi64(C1v, _mm512_mullo_epi64(vecA1, B0v));
                C2v = _mm512_add_epi64(C2v, _mm512_mullo_epi64(vecA2, B0v));
                C3v = _mm512_add_epi64(C3v, _mm512_mullo_epi64(vecA3, B0v));
                C4v = _mm512_add_epi64(C4v, _mm512_mullo_epi64(vecA4, B0v));
                C5v = _mm512_add_epi64(C5v, _mm512_mullo_epi64(vecA5, B0v));
                C6v = _mm512_add_epi64(C6v, _mm512_mullo_epi64(vecA6, B0v));
                C7v = _mm512_add_epi64(C7v, _mm512_mullo_epi64(vecA7, B0v));
            }

            // 将结果存回 C[m:m+7][k:k+7]
            _mm512_storeu_si512((__m512i*)&C[m][k], C0v);
            _mm512_storeu_si512((__m512i*)&C[m + 1][k], C1v);
            _mm512_storeu_si512((__m512i*)&C[m + 2][k], C2v);
            _mm512_storeu_si512((__m512i*)&C[m + 3][k], C3v);
            _mm512_storeu_si512((__m512i*)&C[m + 4][k], C4v);
            _mm512_storeu_si512((__m512i*)&C[m + 5][k], C5v);
            _mm512_storeu_si512((__m512i*)&C[m + 6][k], C6v);
            _mm512_storeu_si512((__m512i*)&C[m + 7][k], C7v);
        }
    }
}
