//FILTRU PREWITT PE AXA Y - MILEA DANIEL - CRISTIAN

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#define WIDTH 256
#define HEIGHT 256
#define FILTER_SIZE 3

__global__ void prewittFilterY(int* input, int* output, int width, int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int col = blockIdx.x * blockDim.x + threadIdx.x + 1;

    if (row < height - 1 && col < width - 1) {
        int sum = 0;
        for (int i = -1; i <= 1; ++i) {
            for (int j = -1; j <= 1; ++j) {
                int rowIdx = row + i;
                int colIdx = col + j;

                int filterVal = (i == -1) ? -1 : ((i == 1) ? 1 : 0);

                sum += input[rowIdx * width + colIdx] * filterVal;
            }
        }
        output[row * width + col] = sum;
    }
}

int main() {
    int* hostInput, * hostOutput, * deviceInput, * deviceOutput;
    int size = WIDTH * HEIGHT * sizeof(int);

    // Alocare memorie host
    hostInput = (int*)malloc(size);
    hostOutput = (int*)malloc(size);

    // Initializare matrice
    for (int i = 0; i < HEIGHT; ++i) {
        for (int j = 0; j < WIDTH; ++j) {
            hostInput[i * WIDTH + j] = j % WIDTH;
        }
    }

    // Alocare memorie device
    cudaMalloc((void**)&deviceInput, size);
    cudaMalloc((void**)&deviceOutput, size);

    // copy date de la host la device
    cudaMemcpy(deviceInput, hostInput, size, cudaMemcpyHostToDevice);

    // 32x32 block si grid dimensiuni
    dim3 blockDim(32, 32);
    dim3 gridDim((WIDTH - 2 + blockDim.x - 1) / blockDim.x, (HEIGHT - 2 + blockDim.y - 1) / blockDim.y);

    // lansare kernel si afisare timp executie
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    prewittFilterY << <gridDim, blockDim >> > (deviceInput, deviceOutput, WIDTH, HEIGHT);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // copy rezultate de la device la host
    cudaMemcpy(hostOutput, deviceOutput, size, cudaMemcpyDeviceToHost);

    // afisare
    for (int i = 0; i < HEIGHT; ++i) {
        for (int j = 0; j < WIDTH; ++j) {
            printf("%d ", hostOutput[i * WIDTH + j]);
        }
        printf("\n");
    }

    printf("\nExecution Time: %f ms\n", milliseconds);

    //eliberare memorie
    free(hostInput);
    free(hostOutput);
    cudaFree(deviceInput);
    cudaFree(deviceOutput);

    return 0;
}
