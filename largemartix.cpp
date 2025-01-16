#include <iostream>
#include <random>
#include <chrono>
#include <fstream>

using namespace std;
using namespace std::chrono;

#define MARTIXSIZE 4096
#define DATAMAX 100000000
#define T long long

typedef T Martix [MARTIXSIZE][MARTIXSIZE];
typedef T (*Martixp)[MARTIXSIZE];

static mt19937 pseudoRandNumGen;
Martix martix1, martix2;
Martix result;

T Rand() { return pseudoRandNumGen() % DATAMAX; };

void DisplayMartix(ostream& os, Martixp martix) {
      for(int i = 0; i < MARTIXSIZE; ++i) {
            for(int j = 0; j < MARTIXSIZE; ++j) {
                  os << martix[i][j] << ' ';
            }
            os << endl;
      }
};

void GenData(Martixp martix) {
      for(int i = 0; i < MARTIXSIZE; ++i) {
            for(int j = 0; j < MARTIXSIZE; ++j) {
                  T t = Rand();

                  martix[i][j] = t;
            }
      }
};

// IJK顺序
void MartixMuitply() {
      for(int i = 0; i < MARTIXSIZE; ++i) {
            for(int j = 0; j < MARTIXSIZE; ++j) {
                  for(int k = 0; k < MARTIXSIZE; ++k) {
                        result[i][j] += martix1[i][k] * martix2[k][j];
                  }
            } 
      }
};

// IKJ顺序
// void MartixMuitply() {
//       for(int i = 0; i < MARTIXSIZE; ++i) {
//             for(int k = 0; k < MARTIXSIZE; ++k) {
//                   for(int j = 0; j < MARTIXSIZE; ++j) {
//                         result[i][j] += martix1[i][k] * martix2[k][j];
//                   }
//             } 
//       }
// };

int main(int argc, char** argv) {
      // int seed = static_cast<int>(time(nullptr));
      int seed = 0;
      pseudoRandNumGen = mt19937(seed);

      GenData(martix1);
      GenData(martix2);

      ofstream input1("./data/largemartix/martix1.txt"), input2("./data/largemartix/martix2.txt");
      ofstream output("./data/largemartix/output.txt");
      cout << "Save martix1." << endl;
      DisplayMartix(input1, martix1);
      cout << "Save martix2." << endl;
      DisplayMartix(input2, martix2);

      steady_clock::time_point StartTime = steady_clock::now();
      MartixMuitply();
      steady_clock::time_point EndTime = steady_clock::now();

      uint64_t duration = duration_cast<microseconds>(EndTime - StartTime).count();

      cout << "Save result." << endl;
      DisplayMartix(output, result);

      cout << "Run Time: " << duration << " ms" << endl;

      return 0;
}