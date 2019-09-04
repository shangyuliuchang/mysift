#include "main.h"

using namespace cv;
using namespace std;

__device__ float rgbSum(uchar* m, int x, int y, int width, int height) {
	return (float)(m[x*width * 3 + y * 3] + m[x*width * 3 + y * 3 + 1] + m[x*width * 3 + y * 3 + 2]);
}
__device__ void featureVectorDirection(uchar* m, int x, int y, float* direction, float* weight, int width, int height) {
	*direction = (float)atan2(rgbSum(m, x, y + 1, width, height) - rgbSum(m, x, y - 1, width, height), rgbSum(m, x + 1, y, width, height) - rgbSum(m, x - 1, y, width, height));
	*weight = sqrt(pow(rgbSum(m, x, y + 1, width, height) - rgbSum(m, x, y - 1, width, height), 2) + pow(rgbSum(m, x + 1, y, width, height) - rgbSum(m, x - 1, y, width, height), 2));
}
__global__ void test(uchar* m, float* devicex, float* devicey, Feature* f, int* num, float* state, int* width, int* height) {
	float mainDirection, direction, weight;
	float count[37] = { 0.0f };
	float max = 0, maxNum = 0;
	float xx, yy, r, sita;
	int nox, noy, noangle;
	float sum = 0;
	Feature ff;
	float scale = 3.0f;
	float x, y;
	float deltaangle, distance;
	float pi = 3.1415926f;

	if (blockDim.x*blockIdx.x + threadIdx.x < *num) {
		x = devicex[blockDim.x*blockIdx.x + threadIdx.x];
		y = devicey[blockDim.x*blockIdx.x + threadIdx.x];

		f[blockDim.x*blockIdx.x + threadIdx.x].x = (int)(x * 640 / *width);
		f[blockDim.x*blockIdx.x + threadIdx.x].y = (int)(y * 640 / *width);
		f[blockDim.x*blockIdx.x + threadIdx.x].rate = (int)(640 / *width);
		ff.x = 0;
		ff.y = 0;
		ff.rate = 0;
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				for (int k = 0; k < 8; k++) {
					ff.vector[i][j][k] = 0;
				}
			}
		}
		if (x >= scale * 3 + 1 && x < *height - scale * 3 + 1 && y >= scale * 3 + 1 && y < *width - scale * 3 + 1) {
			for (int i = (int)(x - scale * 2); i <= (int)(x + scale * 2); i++) {
				for (int j = (int)(y - scale * 2); j <= (int)(y + scale * 2); j++) {
					featureVectorDirection(m, i, j, &direction, &weight, *width, *height);
					count[(int)((int)(direction * 180 / pi + 180) / 10) % 36] += weight * exp(-1 * ((x - i)*(x - i) + (y - j)*(y - j)) / (scale*scale * 4));

					if (count[(int)((int)(direction * 180 / pi + 180) / 10) % 36] > max) {
						max = count[(int)((int)(direction * 180 / pi + 180) / 10) % 36];
						maxNum = (float)((int)((int)(direction * 180 / pi + 180) / 10) % 36);
					}
				}
			}
			maxNum = maxNum + 0.5f*(count[((int)maxNum + 35) % 36] - count[((int)maxNum + 1) % 36]) / (count[((int)maxNum + 35) % 36] + count[((int)maxNum + 1) % 36] - 2.0f*count[(int)maxNum]);
			mainDirection = ((float)maxNum * 10.0f + 5.0f) * pi / 180.0f;
			if (mainDirection < 0)mainDirection += 2 * pi;
			if (mainDirection >= 2 * pi)mainDirection -= 2 * pi;

			f[blockDim.x*blockIdx.x + threadIdx.x].maindirection = mainDirection;
			for (int i = (int)(x - scale * 3); i <= (int)(x + scale * 3); i++) {
				for (int j = (int)(y - scale * 3); j <= (int)(y + scale * 3); j++) {
					r = sqrt((i - x)*(i - x) + (j - y)*(j - y));
					sita = atan2((j - y), (i - x));
					sita -= mainDirection;
					xx = x - r * cos(sita);
					yy = y + r * sin(sita);

					nox = (int)((xx - (x - scale * 2.0f)) / scale);
					noy = (int)((yy - (y - scale * 2.0f)) / scale);

					if (nox >= 0 && nox < 4 && noy >= 0 && noy < 4) {
						featureVectorDirection(m, i, j, &direction, &weight, *width, *height);
						direction += pi;
						direction -= mainDirection;
						if (direction < 0) {
							direction += 2 * pi;
						}
						noangle = (int)((int)(direction * 180 / pi + 180) / 45) % 8;
						for (int vi = 0; vi < 4; vi++) {
							for (int vj = 0; vj < 4; vj++) {
								for (int vk = 0; vk < 8; vk++) {
									deltaangle = abs(vk - direction * 4.0f / pi) / 2;
									if (deltaangle > 18) {
										deltaangle = 36 - deltaangle;
									}
									distance = (float)(pow((float)(x + (vi - 1.5)*scale) - xx, 2) + pow((float)(y + (vj - 1.5)*scale) - yy, 2) + pow(deltaangle, 2));
									ff.vector[vi][vj][vk] += weight * exp(-1 * distance);
								}
							}
						}
					}
				}
			}
			for (int ii = 0; ii < 4; ii++) {
				for (int jj = 0; jj < 4; jj++) {
					for (int kk = 0; kk < 8; kk++) {
						sum += pow(ff.vector[ii][jj][kk], 2);
					}
				}
			}
			sum = sqrt(sum);
			for (int ii = 0; ii < 4; ii++) {
				for (int jj = 0; jj < 4; jj++) {
					for (int kk = 0; kk < 8; kk++) {
						f[blockDim.x*blockIdx.x + threadIdx.x].vector[ii][jj][kk] = ff.vector[ii][jj][kk] / sum;
					}
				}
			}
		}
		f[blockDim.x*blockIdx.x + threadIdx.x].done = blockDim.x*blockIdx.x + threadIdx.x;
	}
}

extern "C" void gpuFeature(Mat * m, float * x, float * y, int num, Feature* feature) {
	int error = 0;
	float* devicestate;
	float statehost[1] = { 0 };
	error += (int)cudaMalloc(&devicestate, sizeof(float));
	error += (int)cudaMemcpy(devicestate, statehost, sizeof(float), cudaMemcpyHostToDevice);

	uchar* deviceMat, *hostMat;
	hostMat = (uchar*)malloc(m->cols*m->rows * 3);
	for (int i = 0; i < m->rows; i++) {
		for (int j = 0; j < m->cols * 3; j++) {
			hostMat[i*m->cols * 3 + j] = m->ptr(i)[j];
		}
	}
	error += (int)cudaMalloc(&deviceMat, m->cols*m->rows * 3);
	error += (int)cudaMemcpy(deviceMat, hostMat, m->cols*m->rows * 3, cudaMemcpyHostToDevice);

	int *deviceWidth, *deviceHeight;
	int width[1] = { 0 }, height[1] = { 0 };
	error += (int)cudaMalloc(&deviceWidth, sizeof(int));
	error += (int)cudaMalloc(&deviceHeight, sizeof(int));
	*width = m->cols;
	*height = m->rows;
	error += (int)cudaMemcpy(deviceWidth, width, sizeof(int), cudaMemcpyHostToDevice);
	error += (int)cudaMemcpy(deviceHeight, height, sizeof(int), cudaMemcpyHostToDevice);

	float *devicex, *devicey;
	error += (int)cudaMalloc(&devicex, num * sizeof(float));
	error += (int)cudaMalloc(&devicey, num * sizeof(float));
	error += (int)cudaMemcpy(devicex, x, num * sizeof(float), cudaMemcpyHostToDevice);
	error += (int)cudaMemcpy(devicey, y, num * sizeof(float), cudaMemcpyHostToDevice);

	Feature* devicefeature;
	error += (int)cudaMalloc(&devicefeature, num * sizeof(Feature));

	int* deviceNum = 0;
	int hostNum[1] = { 0 };
	hostNum[0] = num;
	error += (int)cudaMalloc(&deviceNum, sizeof(int));
	error += (int)cudaMemcpy(deviceNum, hostNum, sizeof(int), cudaMemcpyHostToDevice);

	dim3 dimBlock(256);
	dim3 dimGrid((num / 256) + 1);

	test << <dimGrid, dimBlock >> > (deviceMat, devicex, devicey, devicefeature, deviceNum, devicestate, deviceWidth, deviceHeight);
	cudaDeviceSynchronize();

	error += (int)cudaMemcpy(statehost, devicestate, sizeof(int), cudaMemcpyDeviceToHost);

	error += (int)cudaMemcpy(feature, devicefeature, num * sizeof(Feature), cudaMemcpyDeviceToHost);

	cudaFree(devicex);
	cudaFree(devicey);
	cudaFree(devicefeature);
	cudaFree(devicestate);
	cudaFree(deviceMat);
	cudaFree(deviceHeight);
	cudaFree(deviceWidth);
}
