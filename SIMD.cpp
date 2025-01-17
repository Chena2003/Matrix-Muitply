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

#define MATRIXSIZE 4096
#define DATAMAX 1000
#define T long long 
typedef void (*func)(T **, T **, T **, int );

static mt19937 pseudoRandNumGen;
alignas(32) T **matrix1;
alignas(32) T **matrix2;
alignas(32) T **result;

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

// 调用函数
uint64_t Work(func f, T **A, T **B, T **C, int size, ostream & os) {
      // 初始化result矩阵为0
      for(int i = 0; i < size; ++i) {
            memset(C[i], 0, sizeof(T) * size);
      }

      steady_clock::time_point StartTime = steady_clock::now();
      f(A, B, C, size);
      steady_clock::time_point EndTime = steady_clock::now();

      DisplayMatrix(os, result);

      // milliseconds
      return duration_cast<milliseconds>(EndTime - StartTime).count();
};

// 校验函数
void CheckResult(istream &input, istream &ref) {
      input.seekg(0, ios::beg);
      ref.seekg(0, ios::beg);

      T t1, t2;
      bool flag = true;
      while((input >> t1) && (ref >> t2)) {
            // cout << t1 << ' ' << t2 << endl;
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
      // int seed = 0;
      // int seed = 101;
      int seed = 30120;
      // int seed = 50000123;
      // int seed = 732819634;
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

void MatrixMultiply(T **A, T **B, T **C, int size) {
      for(int i = 0; i < size; ++i) {
            for(int k = 0; k < size; ++k) {
                  for(int j = 0; j < size; ++j) {
                        C[i][j] += A[i][k] * B[k][j];
                  }
            } 
      }
};

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

// 矩阵乘法函数
// void MatrixMultiplyAVX(T **A, T **B, T **C, int size) {
// //     #pragma omp parallel for collapse(2) // 多线程并行化
//     for (int i = 0; i < size; ++i) {
//         for (int j = 0; j < size; ++j) {
//             // 加载A[i][j]到一个AVX向量中
//             __m256i vecA = _mm256_set1_epi32(A[i][j]);
//             int k = 0;
//             for (k = 0; k < size; k += 8) {
//                 // 加载B[j][k:k+7]到一个AVX向量中
//                 __m256i vecB = _mm256_loadu_si256((__m256i*)&B[j][k]);
//                 // 加载C[i][k:k+7]到一个AVX向量中
//                 __m256i vecC = _mm256_loadu_si256((__m256i*)&C[i][k]);
//                 // 计算乘加：C[i][k:k+7] += A[i][j] * B[j][k:k+7]
//                 __m256i mul = _mm256_mullo_epi32(vecA, vecB); // 逐元素乘法
//                 vecC = _mm256_add_epi32(vecC, mul);          // 逐元素加法
//                 // 将结果存回C[i][k:k+7]
//                 _mm256_storeu_si256((__m256i*)&C[i][k], vecC);
//             }
//         }
//     }
// };

// void MatrixMultiplyAVX(T **A, T **B, T **C, int size) {
//     const int blockSize = 64; // 分块大小，根据缓存大小调整

// //     #pragma omp parallel for collapse(2) // 多线程并行化
//     for (int ii = 0; ii < size; ii += blockSize) {
//         for (int jj = 0; jj < size; jj += blockSize) {
//             for (int kk = 0; kk < size; kk += blockSize) {
//                 // 分块计算
//                 for (int i = ii; i < ii + blockSize && i < size; ++i) {
//                     for (int j = jj; j < jj + blockSize && j < size; ++j) {
//                         // 加载A[i][j]到一个AVX向量中
//                         __m256i vecA = _mm256_set1_epi32(A[i][j]);

//                         int k = kk;
//                         // 每次处理 8 个元素
//                         for (; k <= kk + blockSize - 8 && k < size - 7; k += 8) {
//                             // 加载B[j][k:k+7]和C[i][k:k+7]到AVX向量中
//                             __m256i vecB = _mm256_load_si256((__m256i*)&B[j][k]);
//                             __m256i vecC = _mm256_load_si256((__m256i*)&C[i][k]);

//                             // 计算乘加
//                             __m256i mul = _mm256_mullo_epi32(vecA, vecB);
//                             vecC = _mm256_add_epi32(vecC, mul);

//                             // 将结果存回C[i][k:k+7]
//                             _mm256_storeu_si256((__m256i*)&C[i][k], vecC);
//                         }

//                         // 处理剩余的不足 8 的部分
//                         for (; k < kk + blockSize && k < size; ++k) {
//                             C[i][k] += A[i][j] * B[j][k];
//                         }
//                     }
//                 }
//             }
//         }
//     }
// }


// 矩阵乘法函数
// void MatrixMultiplyAVX(int32_t **A, int32_t **B, int32_t **C, int size) {
//     // 初始化结果矩阵 C 为 0
//     for (int i = 0; i < size; ++i) {
//         memset(C[i], 0, size * sizeof(int32_t));
//     }

//     // 遍历矩阵 A 和 C
//     for (int i = 0; i < size; ++i) {
//         for (int j = 0; j < size; ++j) {
//             // 初始化累加器
//             __m256i vecC = _mm256_setzero_si256();

//             // 对矩阵 B 和 A 进行遍历，计算乘积并累加到 C[i][j]
//             for (int k = 0; k < size; k += 8) {
//                 // 加载 A[i][k:k+7] 到 AVX 向量
//                 __m256i vecA = _mm256_loadu_si256((__m256i*)&A[i][k]);

//                 // 加载 B[k:k+7][j] 到 AVX 向量
//                 __m256i vecB = _mm256_setr_epi32(
//                     B[k][j], B[k + 1][j], B[k + 2][j], B[k + 3][j],
//                     B[k + 4][j], B[k + 5][j], B[k + 6][j], B[k + 7][j]
//                 );

//                 // 逐元素乘法
//                 __m256i mul = _mm256_mullo_epi32(vecA, vecB);

//                 // 累加到 vecC
//                 vecC = _mm256_add_epi32(vecC, mul);
//             }

//             // 将 vecC 中的结果累加到 C[i][j]
//             alignas(32) int32_t temp[8];
//             _mm256_store_si256((__m256i*)temp, vecC); // 将向量存储到临时数组
//             for (int t = 0; t < 8; ++t) {
//                 C[i][j] += temp[t]; // 累加向量中的元素
//             }
//         }
//     }
// }

// 矩阵乘法函数
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
