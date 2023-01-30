#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <mpi.h>
#include <stdint.h>
#include <sys/time.h>
#include <time.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#include <time.h>
#if not defined(WIN32) && not defined(__USE_POSIX199309)
#include <sys/time.h>
#endif
#include "Chronometer.hpp"

Chronometer::Chronometer() : m_time(0) {}
// ------------------------------------------------------------------------
double Chronometer::click() {
#ifdef WIN32
  clock_t chrono;
  chrono = clock();
  double t = ((double)chrono) / CLOCKS_PER_SEC;
#elif defined(CLOCK_MONOTONIC_RAW)
  struct timespec tp;
  clock_gettime(CLOCK_MONOTONIC_RAW, &tp);
  double t = tp.tv_sec + 1.E-9 * tp.tv_nsec;
#else
  struct timeval tv;
  gettimeofday(&tv, NULL);
  double t = tv.tv_sec + 1.E-6 * tv.tv_usec;
#endif
  double dt = t - m_time;
  m_time = t;
  return dt;
}

//
// Init
//
void init(int *ndim_tab, int *dim, double *T0, double *x, double *y,
          double *dx) {
  const double lx = 10.;
  const double ly = 10.;

  dx[0] = lx / dim[0];
  dx[1] = ly / dim[1];

  const double x0 = 0;
  const double y0 = 0;
  const double xinit = 5;
  const double yinit = 5;

  for (int64_t i = 0; i < ndim_tab[0]; ++i) {
    x[i] = (i - 2) * dx[0] + x0;
    for (int64_t j = 0; j < ndim_tab[1]; ++j) {
      y[j] = (j - 2) * dx[1] + y0;

      int l = j * ndim_tab[0] + i;

      double r = std::sqrt((x[i] - xinit) * (x[i] - xinit) +
                           (y[j] - yinit) * (y[j] - yinit));
      T0[l] = 300 + 10 * std::exp(-r / 0.2);
    }
  }
}

//
// mise a jour
//
/* TODO fonction global*/
__global__ void mise_a_jour(int *ndim_tab, double *T0, double *T1,
                            double *bilan, const double dt);

//
// advection
//
__global__ void advection(int *ndim_tab, double *T, double *bilan, double *dx,
                          double *a, int step);

//
// diffusion
//
__global__ void diffusion(int *ndim_tab, double *T, double *bilan, double *dx,
                          const double mu);

double *InitGPUVector(long int N);
double *InitVector(long int N);

//
// Bord
//
__global__ void bord(int *ndim_tab, double *Tout, int nfic);

int main(int nargc, char *argv[]) {

  cudaError_t err = cudaSuccess;

  char fileName[255];
  FILE *out;

  int dim[2];
  dim[0] = 500;
  dim[1] = 500;
  int nfic = 2;

  sprintf(fileName, "Sortie.txt");
  out = fopen(fileName, "w");

  //
  //
  // Determination la taille des grilles dans les direction X ( Ndim_tab[0]) et
  // Y ( Ndim_tab[1]) avec les cellules fantomes
  //
  //
  int Ndim_tab[2];
  Ndim_tab[0] = dim[0] + 2 * nfic;
  Ndim_tab[1] = dim[1] + 2 * nfic;

  double *x, *y, *T1, *T0, *bilan;
  x = InitVector(Ndim_tab[0]);
  y = InitVector(Ndim_tab[1]);
  bilan = InitVector(Ndim_tab[0] * Ndim_tab[1]);
  T1 = InitVector(Ndim_tab[0] * Ndim_tab[1]);
  T0 = InitVector(Ndim_tab[0] * Ndim_tab[1]);

  double dx[2];

  init(Ndim_tab, dim, T0, x, y, dx);
  fprintf(out, "dim blocX =  %d, dim blocY =  %d, dx= %f, dy= %f \n",
          Ndim_tab[0], Ndim_tab[1], dx[0], dx[1]);

  for (int64_t j = 0; j < Ndim_tab[1]; ++j) {
    for (int64_t i = 0; i < Ndim_tab[0]; ++i) {

      int l = j * Ndim_tab[0] + i;
      fprintf(out, " Init: %f %f %f   \n", x[i], y[j], T0[l]);
    }
    fprintf(out, " Init: \n");
  }

  const double dt = 0.005; // pas de temps
  double U[2];
  U[0] = 1.; // vitesse advection
  U[1] = 1.;

  const double mu = 0.0005; // coeff diffusion
  int Nitmax = 2000;
  int Stepmax = 2;

  // Allocation en mémoire pour cuda
  double *xCuda, *yCuda, *T1Cuda, *T0Cuda, *bilanCuda;
  xCuda = InitGPUVector(Ndim_tab[0]);
  yCuda = InitGPUVector(Ndim_tab[1]);
  bilanCuda = InitGPUVector(Ndim_tab[0] * Ndim_tab[1]);
  T1Cuda = InitGPUVector(Ndim_tab[0] * Ndim_tab[1]);
  T0Cuda = InitGPUVector(Ndim_tab[0] * Ndim_tab[1]);
  double *dxCuda = InitGPUVector(2);
  double *UCuda = InitGPUVector(2);
  int *Ndim_tabCuda;
  cudaMalloc((void **)&Ndim_tabCuda, 2 * sizeof(int));
  cudaDeviceSynchronize();

  // Copy CPU -> GPU
  err = cudaMemcpy(xCuda, x, Ndim_tab[0] * sizeof(double),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    fprintf(
        stderr,
        "Failed to copy vector x from host to device xCuda (error code %s)!\n",
        cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  cudaDeviceSynchronize();

  err = cudaMemcpy(yCuda, y, Ndim_tab[1] * sizeof(double),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    fprintf(
        stderr,
        "Failed to copy vector y from host to device yCuda (error code %s)!\n",
        cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  cudaDeviceSynchronize();

  err = cudaMemcpy(T0Cuda, T0, Ndim_tab[0] * Ndim_tab[1] * sizeof(double),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    fprintf(stderr,
            "Failed to copy vector T0 from host to device T0Cuda (error code "
            "%s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  cudaDeviceSynchronize();

  err = cudaMemcpy(dxCuda, dx, 2 * sizeof(double), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    fprintf(stderr,
            "Failed to copy vector dx from host to device dxCuda (error code "
            "%s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  cudaDeviceSynchronize();

  err = cudaMemcpy(UCuda, U, 2 * sizeof(double), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    fprintf(
        stderr,
        "Failed to copy vector U from host to device UCuda (error code %s)!\n",
        cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  cudaDeviceSynchronize();

  err = cudaMemcpy(Ndim_tabCuda, Ndim_tab, 2 * sizeof(int),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    fprintf(stderr,
            "Failed to copy vector Ndim_tabCuda from host to device "
            "Ndim_TabCuda (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  cudaDeviceSynchronize();

  // Dim Grid et block size
  const int blocksize = 32;
  dim3 dimBlock(blocksize, blocksize);
  dim3 dimGrid((Ndim_tab[0] + dimBlock.x - 1) / dimBlock.x,
               (Ndim_tab[1] + dimBlock.y - 1) / dimBlock.y);

  // Boucle en temps
  for (int64_t nit = 0; nit < Nitmax; ++nit) {
    // Boucle Runge-Kutta
    double *Tin;
    double *Tout;
    double *Tbilan;
    for (int64_t step = 0; step < Stepmax; ++step) {
      // mise a jour point courant
      if (step == 0) {
        Tin = T0Cuda;
        Tout = T1Cuda;
        Tbilan = T0Cuda;
      } else {
        Tin = T0Cuda;
        Tout = T0Cuda;
        Tbilan = T1Cuda;
      }
      // Advection
      advection<<<dimGrid, dimBlock>>>(Ndim_tabCuda, Tbilan, bilanCuda, dxCuda,
                                       UCuda, step);
      cudaDeviceSynchronize();

      // diffusion
      diffusion<<<dimGrid, dimBlock>>>(Ndim_tabCuda, Tbilan, bilanCuda, dxCuda,
                                       mu);
      cudaDeviceSynchronize();

      // Mise à jour
      mise_a_jour<<<dimGrid, dimBlock>>>(Ndim_tabCuda, Tin, Tout, bilanCuda,
                                         dt);
      cudaDeviceSynchronize();

      // bord
      bord<<<dimGrid, dimBlock>>>(Ndim_tabCuda, Tout, nfic);
      cudaDeviceSynchronize();

    } // Nstepmax
  }   // Nitmax

  // Recopy GPU ->  CPU
  err = cudaMemcpy(T0, T0Cuda, Ndim_tab[0] * Ndim_tab[1] * sizeof(double),
                   cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  if (err != cudaSuccess) {
    fprintf(
        stderr,
        "Failed to copy vector T0Cuda from device to host (error code%s)!\n",
        cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  for (int64_t j = 0; j < Ndim_tab[1]; ++j) {
    for (int64_t i = 0; i < Ndim_tab[0]; ++i) {

      int l = j * Ndim_tab[0] + i;
      fprintf(out, " Final %f %f %f  \n", x[i], y[j], T0[l]);
    }
    fprintf(out, " Final \n");
  }

  fclose(out);

  free(T0);
  free(T1);
  free(bilan);
  free(x);
  free(y);

  cudaFree(T0Cuda);
  cudaFree(T1Cuda);
  cudaFree(bilanCuda);
  cudaFree(xCuda);
  cudaFree(yCuda);
  cudaFree(dxCuda);
  cudaFree(UCuda);
  cudaFree(Ndim_tabCuda);

  return EXIT_SUCCESS;
}

/* TODO fonction global*/
__global__ void mise_a_jour(int *ndim_tab, double *T0, double *T1,
                            double *bilan, const double dt) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  int l = j * ndim_tab[0] + i;

  if ((j >= 2) && (j < ndim_tab[1]) && (i >= 2) && (i < ndim_tab[0])) {
    T1[l] = T0[l] - dt * bilan[l];
  }
}

__global__ void advection(int *ndim_tab, double *T, double *bilan, double *dx,
                          double *a, int step) {
  double c1 = 7. / 6.;
  double c2 = 1. / 6.;

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  // printf("dx %0.9f %0.9f \n", dx, a*dt );
  // 1er sous pas schema Heun
  if (step == 0) {
    if ((j >= 2) && (j < ndim_tab[1]) && (i >= 2) && (i < ndim_tab[0])) {
      int l = j * ndim_tab[0] + i; // (i  , j  )
      int l1 = l + 1;              // (i+1, j  )
      int l2 = l - 1;              // (i-1, j  )
      int l3 = l - 2;              // (i-2, j  )
      int l4 = l + 2;              // (i+2, j  )

      double fm = (T[l] + T[l2]) * c1 - (T[l1] + T[l3]) * c2;
      double fp = (T[l1] + T[l]) * c1 - (T[l4] + T[l2]) * c2;

      bilan[l] = a[0] * (fp - fm) / (2. * dx[0]);

      l1 = l + ndim_tab[0];     // (i  , j+1)
      l2 = l - ndim_tab[0];     // (i  , j-1)
      l3 = l - 2 * ndim_tab[0]; // (i  , j+2)
      l4 = l + 2 * ndim_tab[0]; // (i  , j-2)

      fm = (T[l] + T[l2]) * c1 - (T[l1] + T[l3]) * c2;
      fp = (T[l1] + T[l]) * c1 - (T[l4] + T[l2]) * c2;

      bilan[l] += a[1] * (fp - fm) / (2. * dx[1]);
    }

  }
  // 2eme sous pas schema Heun
  else {
    if ((j >= 2) && (j < ndim_tab[1]) && (i >= 2) && (i < ndim_tab[0])) {

      int l = j * ndim_tab[0] + i; // (i  , j  )
      int l1 = l + 1;              // (i+1, j  )
      int l2 = l - 1;              // (i-1, j  )
      int l3 = l - 2;              // (i-2, j  )
      int l4 = l + 2;              // (i+2, j  )

      double fm = (T[l] + T[l2]) * c1 - (T[l1] + T[l3]) * c2;
      double fp = (T[l1] + T[l]) * c1 - (T[l4] + T[l2]) * c2;

      bilan[l] = 0.5 * (bilan[l] + a[0] * (fp - fm) / (2. * dx[0]));

      l1 = l + ndim_tab[0];     // (i  , j+1)
      l2 = l - ndim_tab[0];     // (i  , j-1)
      l3 = l - 2 * ndim_tab[0]; // (i  , j+2)
      l4 = l + 2 * ndim_tab[0]; // (i  , j-2)

      fm = (T[l] + T[l2]) * c1 - (T[l1] + T[l3]) * c2;
      fp = (T[l1] + T[l]) * c1 - (T[l4] + T[l2]) * c2;

      bilan[l] += (a[1] * (fp - fm) / (2. * dx[1])) * 0.5;
    }
  }
}

__global__ void diffusion(int *ndim_tab, double *T, double *bilan, double *dx,
                          const double mu) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if ((j >= 2) && (j < ndim_tab[1]) && (i >= 2) && (i < ndim_tab[0])) {

    int l = j * ndim_tab[0] + i; // (i  , j  )
    int l1 = l + 1;              // (i+1, j  )
    int l2 = l - 1;              // (i-1, j  )
    int l3 = l + ndim_tab[0];    // (i  , j+1)
    int l4 = l - ndim_tab[0];    // (i  , j-1)
    bilan[l] = bilan[l] - mu * ((T[l1] + T[l2] - 2 * T[l]) / (dx[0] * dx[0]) +
                                (T[l3] + T[l4] - 2 * T[l]) / (dx[1] * dx[1]));
  }
}

__global__ void bord(int *ndim_tab, double *Tout, int nfic) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  for (int64_t ific = 0; ific < nfic; ific++) {
    if ((i >= 0) && (i < ndim_tab[0])) {
      // Jmin
      int l0 = ific * ndim_tab[0] + i;

      int l1 = ndim_tab[0] * (ndim_tab[1] - 2 * nfic + ific) + i;

      Tout[l0] = Tout[l1];

      // Jmax
      l0 = ndim_tab[0] * (ndim_tab[1] - nfic + ific) + i;
      l1 = ndim_tab[0] * (nfic + ific) + i;

      Tout[l0] = Tout[l1];
    }

    if ((j >= 0) && (j < ndim_tab[1])) {
      // Imin
      int l0 = ific + j * ndim_tab[0];
      int l1 = l0 + ndim_tab[0] - 2 * nfic;

      Tout[l0] = Tout[l1];

      // Imax
      l0 = ific + (j + 1) * ndim_tab[0] - nfic;
      l1 = l0 - ndim_tab[0] + 2 * nfic;

      Tout[l0] = Tout[l1];
    }
  }
}

double *InitGPUVector(long int N) {
  cudaError_t err = cudaSuccess;
  double *d_x;
  err = cudaMalloc((void **)&d_x, N * sizeof(double));
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device vector (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  return d_x;
}

double *InitVector(long int N) {
  double *x;
  x = (double *)malloc(N * sizeof(double));
  assert(x);
  return x;
}