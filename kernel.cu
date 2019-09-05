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

					//if (nox >= 0 && nox < 4 && noy >= 0 && noy < 4) {
					if (r <= scale * 2) {
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
									deltaangle = abs(vk + 0.5 - direction * 4 / pi);
									if (deltaangle > 4) {
										deltaangle = 8 - deltaangle;
									}
									distance = (float)(pow((float)(x + (vi - 1.5)*scale) - xx, 2) + pow((float)(y + (vj - 1.5)*scale) - yy, 2) + pow(deltaangle*scale, 2));
									ff.vector[vi][vj][vk] += weight * exp(-1 * distance / 18.0f);
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
	cudaFree(deviceNum);
}

__device__ float subPixelGray(uchar* m, float x, float y, float width, float height) {
	int x1, x2, y1, y2;
	float p1, p2, p3, p4, p5, p6, p7;
	x1 = (int)x;
	x2 = x1 + 1;
	y1 = (int)y;
	y2 = y1 + 1;
	p1 = rgbSum(m, x1, y1, width, height);
	p2 = rgbSum(m, x1, y2, width, height);
	p3 = rgbSum(m, x2, y1, width, height);
	p4 = rgbSum(m, x2, y2, width, height);
	p5 = (p2 - p1)*(y - y1) + p1;
	p6 = (p4 - p3)*(y - y1) + p3;
	p7 = (p6 - p5)*(x - x1) + p5;
	return p7;
}
__global__ void pickoutMax(uchar* m1, uchar* m2, uchar* m3, int* width, int* height, float* outx, float *outy) {
	float dxx, dyy, dxy, dx, dy, x, y;
	float x0 = (blockDim.x*blockIdx.x + threadIdx.x) / (*width);
	float y0 = (blockDim.x*blockIdx.x + threadIdx.x) % (*width);
	int featurescale = 3;
	int use = 1;
	int flag;
	float threshold = 10;

	if (!(x0 >= featurescale * 3 + 1 && x0 < *height - featurescale * 3 + 1 && y0 >= featurescale * 3 + 1 && y0 < *width - featurescale * 3 + 1)) {
		use = 0;
	}
	else {
		flag = 1;
		for (int i = x0 - 1; i < x0 + 1; i++) {
			for (int j = y0 - 1; j < y0 + 1; j++) {
				if (rgbSum(m1, i, j, *width, *height)+threshold >= rgbSum(m2, x0, y0, *width, *height)) {
					flag = 0;
					use = 0;
					break;
				}
			}
		}
		for (int i = x0 - 1; i < x0 + 1; i++) {
			for (int j = y0 - 1; j < y0 + 1; j++) {
				if (rgbSum(m2, i, j, *width, *height) + threshold >= rgbSum(m2, x0, y0, *width, *height) && !(i==x0 && j==y0)) {
					flag = 0;
					use = 0;
					break;
				}
			}
		}
		for (int i = x0 - 1; i < x0 + 1; i++) {
			for (int j = y0 - 1; j < y0 + 1; j++) {
				if (rgbSum(m3, i, j, *width, *height) + threshold >= rgbSum(m2, x0, y0, *width, *height)) {
					flag = 0;
					use = 0;
					break;
				}
			}
		}
		if (flag == 1) {
			for (int l = 0; l < 5; l++) {
				dxx = subPixelGray(m2, x0 - 1, y0,*width,*height) + subPixelGray(m2, x0 + 1, y0, *width, *height) - 2 * subPixelGray(m2, x0, y0, *width, *height);
				dyy = subPixelGray(m2, x0, y0 - 1, *width, *height) + subPixelGray(m2, x0, y0 + 1, *width, *height) - 2 * subPixelGray(m2, x0, y0, *width, *height);
				dxy = ((subPixelGray(m2, x0 + 0.5, y0 + 0.5, *width, *height) - subPixelGray(m2, x0 + 0.5, y0 - 0.5, *width, *height)) - (subPixelGray(m2, x0 - 0.5, y0 + 0.5, *width, *height) - subPixelGray(m2, x0 - 0.5, y0 - 0.5, *width, *height)));
				dx = (subPixelGray(m2, x0 + 0.5, y0, *width, *height) - subPixelGray(m2, x0 - 0.5, y0, *width, *height));
				dy = (subPixelGray(m2, x0, y0 + 0.5, *width, *height) - subPixelGray(m2, x0, y0 - 0.5, *width, *height));
				if ((dxy*dxy - dxx * dyy) != 0 && dxx != 0) {
					y = (dx*dxy - dy * dxx) / (dxx*dyy - dxy * dxy);
					x = (dy*dxy - dx * dyy) / (dxx*dyy - dxy * dxy);
				}
				else {
					use = 0;
					break;
				}
				if (abs(x) > 0.5 || abs(y) > 0.5) {
					use = 0;
					break;
				}
				x0 += x;
				y0 += y;
				if (!(x0 >= featurescale * 3 + 1 && x0 < *height - featurescale * 3 + 1 && y0 >= featurescale * 3 + 1 && y0 < *width - featurescale * 3 + 1)) {
					x0 -= x;
					y0 -= y;
					use = 0;
					break;
				}
			}
			//if (dxy < 10) {
			//	use = 0;
			//}
			if (use == 1) {
				outx[(blockDim.x*blockIdx.x + threadIdx.x)] = x0;
				outy[(blockDim.x*blockIdx.x + threadIdx.x)] = y0;
			}
		}
	}
	if (use == 0) {
		outx[(blockDim.x*blockIdx.x + threadIdx.x)] = 0;
		outy[(blockDim.x*blockIdx.x + threadIdx.x)] = 0;
	}
}

extern "C" void gpuMax(Mat* m1, Mat* m2, Mat* m3, float* xout, float* yout) {
	int error = 0;
	uchar* deviceMat1, *deviceMat2, *deviceMat3, *hostMat1, *hostMat2, *hostMat3;
	error+=(int)cudaMalloc(&deviceMat1, m1->cols*m1->rows * 3);
	error += (int)cudaMalloc(&deviceMat2, m1->cols*m1->rows * 3);
	error += (int)cudaMalloc(&deviceMat3, m1->cols*m1->rows * 3);
	hostMat1 = (uchar*)malloc(m1->cols*m1->rows * 3);
	hostMat2 = (uchar*)malloc(m1->cols*m1->rows * 3);
	hostMat3 = (uchar*)malloc(m1->cols*m1->rows * 3);
	for (int i = 0; i < m1->rows; i++) {
		for (int j = 0; j < m1->cols * 3; j++) {
			hostMat1[i*m1->cols * 3 + j] = m1->ptr(i)[j];
		}
	}
	for (int i = 0; i < m1->rows; i++) {
		for (int j = 0; j < m1->cols * 3; j++) {
			hostMat2[i*m1->cols * 3 + j] = m2->ptr(i)[j];
		}
	}
	for (int i = 0; i < m1->rows; i++) {
		for (int j = 0; j < m1->cols * 3; j++) {
			hostMat2[i*m1->cols * 3 + j] = m2->ptr(i)[j];
		}
	}
	error += (int)cudaMemcpy(deviceMat1, hostMat1, m1->cols*m1->rows * 3, cudaMemcpyHostToDevice);
	error += (int)cudaMemcpy(deviceMat2, hostMat2, m1->cols*m1->rows * 3, cudaMemcpyHostToDevice);
	error += (int)cudaMemcpy(deviceMat3, hostMat3, m1->cols*m1->rows * 3, cudaMemcpyHostToDevice);

	float* deviceoutx, *deviceouty;
	error += (int)cudaMalloc(&deviceoutx, m1->cols*m1->rows * sizeof(float));
	error += (int)cudaMalloc(&deviceouty, m1->cols*m1->rows * sizeof(float));

	dim3 blockDim = 512;
	dim3 gridDim = (m1->cols*m1->rows) / 512 + 1;

	int width[1], height[1];
	width[0] = m1->cols;
	height[0] = m1->rows;
	int *deviceWidth, *deviceHeight;
	error += (int)cudaMalloc(&deviceWidth, sizeof(int));
	error += (int)cudaMalloc(&deviceHeight, sizeof(int));
	error += (int)cudaMemcpy(deviceWidth, width, sizeof(int), cudaMemcpyHostToDevice);
	error += (int)cudaMemcpy(deviceHeight, height, sizeof(int), cudaMemcpyHostToDevice);

	pickoutMax << <gridDim, blockDim >> > (deviceMat1, deviceMat2, deviceMat3, deviceWidth, deviceHeight, deviceoutx, deviceouty);
	cudaDeviceSynchronize();

	error += (int)cudaMemcpy(xout, deviceoutx, m1->cols*m1->rows * sizeof(float), cudaMemcpyDeviceToHost);
	error += (int)cudaMemcpy(yout, deviceouty, m1->cols*m1->rows * sizeof(float), cudaMemcpyDeviceToHost);

	//cout << error << endl;

	cudaFree(deviceMat1);
	cudaFree(deviceMat2);
	cudaFree(deviceMat3);
	free(hostMat1);
	free(hostMat2);
	free(hostMat3);
	cudaFree(deviceoutx);
	cudaFree(deviceouty);
	cudaFree(deviceWidth);
	cudaFree(deviceHeight);
}

__global__ void gpuMatch(Feature* f, Feature* target, int* num, int *ans, int * sourcenum) {
	int index = blockDim.x*blockIdx.x + threadIdx.x;
	if (index < *sourcenum) {
		float matchk, matchmax = 0, matchNo;
		for (int i = 0; i < *num; i++) {
			matchk = f[index].gpumatch(&target[i]);
			if (matchk > matchmax) {
				matchmax = matchk;
				matchNo = i;
			}
		}
		if (matchmax > 0.99f) {
			ans[index] = matchNo;
		}
		else {
			ans[index] = -1;
		}
	}
}
extern "C" void gpuFeatureMatch(Feature* source, Feature* target, int sourcenum, int targetnum, int* ans) {
	Feature *devicesource, *devicetarget;
	cudaMalloc(&devicesource, sourcenum * sizeof(Feature));
	cudaMalloc(&devicetarget, targetnum * sizeof(Feature));
	cudaMemcpy(devicesource, source, sourcenum * sizeof(Feature),cudaMemcpyHostToDevice);
	cudaMemcpy(devicetarget, target, targetnum * sizeof(Feature),cudaMemcpyHostToDevice);

	int* deviceans;
	cudaMalloc(&deviceans, sourcenum * sizeof(int));

	int *devicesourcenum, *devicetargetnum;
	cudaMalloc(&devicesourcenum, sizeof(int));
	cudaMalloc(&devicetargetnum, sizeof(int));
	int snum[1], tnum[1];
	snum[0] = sourcenum;
	tnum[0] = targetnum;
	cudaMemcpy(devicesourcenum, snum, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(devicetargetnum, tnum, sizeof(int), cudaMemcpyHostToDevice);
	
	dim3 blockDim(512);
	dim3 gridDim(sourcenum / 512 + 1);

	gpuMatch << <gridDim, blockDim >> > (devicesource, devicetarget, devicetargetnum, deviceans, devicesourcenum);
	cudaDeviceSynchronize();

	cudaMemcpy(ans, deviceans, sourcenum * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(devicesource);
	cudaFree(devicetarget);
	cudaFree(deviceans);
	cudaFree(devicesourcenum);
	cudaFree(devicetargetnum);
}

