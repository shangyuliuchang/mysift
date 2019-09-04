#pragma once
#include<iostream>
#include<GL/glew.h>
#include<GLFW/glfw3.h>
#include<GL/glut.h>
#include<opencv2/highgui.hpp>
#include<opencv2/opencv.hpp>
#include<opencv2/xfeatures2d/nonfree.hpp>
#include"cuda_runtime.h"
#include "device_launch_parameters.h"
extern cv::Mat *devicemat;
extern float matchThreashold;
class Feature {
public:
	int x, y, rate, done;
	float maindirection;
	float vector[4][4][8];
	bool match(Feature* f) {
		float sum = 0, sumA = 0, sumB = 0, ans;
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				for (int kk = 0; kk < 8; kk++) {
					sum += f->vector[i][j][kk] * this->vector[i][j][kk];
					sumA += pow(f->vector[i][j][kk], 2);
					sumB += pow(this->vector[i][j][kk], 2);
				}
			}
		}
		sumA = sqrt(sumA);
		sumB = sqrt(sumB);

		ans = sum / (sumA*sumB);
		if (ans > matchThreashold) {
			return true;
		}
		else {
			return false;
		}
	}
	void clear() {
		x = 0;
		y = 0;
		rate = 0;
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				for (int k = 0; k < 8; k++) {
					vector[i][j][k] = 0;
				}
			}
		}
	}
	void cloneto(Feature* cloning) {
		cloning->x = this->x;
		cloning->y = this->y;
		cloning->rate = this->rate;
		cloning->maindirection = this->maindirection;
		cloning->done = this->done;
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				for (int k = 0; k < 8; k++) {
					cloning->vector[i][j][k] = this->vector[i][j][k];
				}
			}
		}
	}
};
extern Feature hostfeature[10000];