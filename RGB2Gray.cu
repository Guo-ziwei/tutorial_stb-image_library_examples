#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/time.h>

#include <stdio.h>
#include <stdlib.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image/stb_image_write.h"

#define CHANNELS 3

__global__ void colorConvert(unsigned char* grayImage, unsigned char* rgbImage, int width, int height) {
    int x = threadIdx.x;
    int y = blockIdx.x;
    if (x < width && y < height) {
        int grayoffset = y * width + x;
        int rgboffset = grayoffset * CHANNELS;
        unsigned char r = rgbImage[rgboffset];
        unsigned char g = rgbImage[rgboffset + 1];
        unsigned char b = rgbImage[rgboffset + 2];
        grayImage[grayoffset] = 0.21f * r + 0.71f * g + 0.07f * b;
    }
}

void rgb2gray(unsigned char* grayImage, unsigned char* rgbImage, int width, int height) {
    int grayoffset = 0, rgboffset = 0;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            grayoffset = y * width + x;
            rgboffset = grayoffset * CHANNELS;
            unsigned char r = rgbImage[rgboffset];
            unsigned char g = rgbImage[rgboffset + 1];
            unsigned char b = rgbImage[rgboffset + 2];
            grayImage[grayoffset] = 0.21f * r + 0.71f * g + 0.07f * b;
        }
    }
}

int main(int argc, char const* argv[]) {
    unsigned char* rgbimage;
    unsigned char* grayimage;
    int width, height, channels;
    struct timeval start, end;
    unsigned char* img = stbi_load("Shapes.png", &width, &height, &channels, CHANNELS);
    if (img == NULL) {
        printf("Error in loading the image\n");
        exit(1);
    }
    printf("Loaded image with a width of %dpx, a height of %dpx and %d channels\n", width, height, CHANNELS);
    size_t img_size = width * height * CHANNELS;
    int gray_channels = CHANNELS == 4 ? 2 : 1;
    size_t gray_img_size = width * height * gray_channels;
    unsigned char* gray_img = (unsigned char*)malloc(gray_img_size);
    cudaMalloc((void**)&rgbimage, img_size);
    cudaMalloc((void**)&grayimage, gray_img_size);
    // Executing kernel
    dim3 block(width, 1, 1);
    dim3 grid(height, 1, 1);
    gettimeofday(&start, nullptr);
    cudaMemcpy(rgbimage, img, img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(grayimage, gray_img, gray_img_size, cudaMemcpyHostToDevice);
    colorConvert<<<grid, block>>>(grayimage, rgbimage, width, height);
    cudaMemcpy(gray_img, grayimage, gray_img_size, cudaMemcpyDeviceToHost);
    gettimeofday(&end, nullptr);
    double elapsed_seconds = (end.tv_sec - start.tv_sec) * 1e3;
    elapsed_seconds += (end.tv_usec - start.tv_usec) * 1e-3;
    printf("gpu elapsed time: %lf\n", elapsed_seconds);
    // stbi_write_jpg("sky_gray.jpg", width, height, gray_channels, gray_img, 100);
    stbi_write_png("Shapes_gray_cuda.png", width, height, gray_channels, gray_img, width * gray_channels);
    printf("Write image with a width of %dpx, a height of %dpx and %d channels\n", width, height, gray_channels);
    gettimeofday(&start, nullptr);
    rgb2gray(gray_img, img, width, height);
    gettimeofday(&end, nullptr);
    elapsed_seconds = (end.tv_sec - start.tv_sec) * 1e3;
    elapsed_seconds += (end.tv_usec - start.tv_usec) * 1e-3;
    printf("elapsed time: %lf\n", elapsed_seconds);
    stbi_image_free(img);
    free(gray_img);
    cudaFree(grayimage);
    cudaFree(rgbimage);
    return 0;
}
