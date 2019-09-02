#include"main.h"

using namespace cv;
using namespace std;

void subtraction(Mat*, Mat*, Mat*);
void normalize(Mat*);
float rgbSum(uchar* p, int j);
float subPixelGray(Mat* m, float x, float y);

int keypoints[640 * 480][5], pos;
int edgeTest = 1;
int targetNum = 0;
int resetTarget = 0;
int match = 1;
int point = 0;
VideoCapture cap;
Mat img, pic, descriptor, descriptor2, raw, preprocess;
GLfloat image[480 * 640 * 3 * 2];
GLFWwindow *window;
GLint success;
GLuint vertexShader, fragmentShader, shaderProgram, vbo, vao, lists;
char str[] = { 'r',':','x','x','x',' ','g',':','x','x','x',' ','b',':','x','x','x' };
float sigma = 1.6f, k = 0.5f;
int threashold = 30;
float edgeThreashold = 5.0f;
float preBlur = 0.5f;
const float pi = 3.14159f;
float edgeWidth = 3.0f;
float matchThreashold = 0.9f;


class GaussianMat {
public:
	Mat gaussianmat;
	void update(Mat* m, double sigma) {
		GaussianBlur(*m, gaussianmat, Size(0,0), sigma, sigma);
	}
};
class Feature {
public:
	int x, y, rate, maindirection;
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
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				for (int k = 0; k < 8; k++) {
					cloning->vector[i][j][k] = this->vector[i][j][k];
				}
			}
		}
	}
};
Feature feature[10000], matchTarget[10000];
class DoG {
public:
	Mat g[10];
	Mat dog[9];
	void update(Mat* m, int num, double sigma0, double k) {
		number = num;
		double sigma = sigma0;
		g[0] = m->clone();
		for (int i = 1; i < num; i++) {
			sigma *= k;
			GaussianBlur(g[i - 1], g[i], Size(0, 0), sigma, sigma);
			subtraction(&g[i], &g[i - 1], &dog[i - 1]);
			normalize(&dog[i - 1]);
		}
	}
	void compute() {
		float x, y;
		GLubyte* p[9];

		for (int kk = 1; kk < number - 2; kk++) {
			for (int i = 1; i < g[0].rows - 1; i++) {

				p[0] = dog[kk - 1].ptr(i - 1);
				p[1] = dog[kk - 1].ptr(i);
				p[2] = dog[kk - 1].ptr(i + 1);
				p[3] = dog[kk].ptr(i - 1);
				p[4] = dog[kk].ptr(i);
				p[5] = dog[kk].ptr(i + 1);
				p[6] = dog[kk + 1].ptr(i - 1);
				p[7] = dog[kk + 1].ptr(i);
				p[8] = dog[kk + 1].ptr(i + 1);

				for (int j = 1; j < g[0].cols - 1; j++) {
					if (max(p, j)) {
						if (isnotEdge2(&dog[kk], dog[kk].cols, dog[kk].rows, i, j)||!edgeTest) {
							/*a = 0.5f*(rgbSum(p[5], j) + rgbSum(p[3], j)) - rgbSum(p[4], j);
							d = 0.5f*(rgbSum(p[5], j) - rgbSum(p[3], j));
							b = 0.5f*(rgbSum(p[4], j + 1) + rgbSum(p[4], j - 1)) - rgbSum(p[4], j);
							e = 0.5f*(rgbSum(p[4], j + 1) - rgbSum(p[4], j - 1));
							c = 0.25f*((rgbSum(p[5], j + 1) - rgbSum(p[3], j - 1)) - (rgbSum(p[3], j + 1) - rgbSum(p[5], j - 1)));

							y = (c*d - 2 * a*e) / (4 * a*b - c * c);
							x = -1 * (c*d + c * c*y) / (2 * a*c);

							x = 0; y = 0;*/
							x = -0.5f*(rgbSum(p[5], j) - rgbSum(p[3], j)) / (rgbSum(p[3], j) + rgbSum(p[5], j) - 2 * rgbSum(p[4], j));
							y = -0.5f*(rgbSum(p[4], j + 1) - rgbSum(p[4], j - 1)) / (rgbSum(p[4], j + 1) + rgbSum(p[4], j - 1) - 2 * rgbSum(p[4], j));

							feature[pos].clear();
							feature[pos].x = (int)((i + x) * 640 / g[0].cols);
							feature[pos].y = (int)((j + y) * 640 / g[0].cols);
							feature[pos].rate = 640 / g[0].cols;
							featureCompute(&g[kk], (float)i + x, (float)j + y, &feature[pos], 3.0f);

							if (pos < 9999) {
								pos++;
							}
						}
					}
				}
			}
		}
	}
private:
	int number;
	bool max(uchar** p, int j) {
		for (int i = 0; i < 9; i++) {
			for (int k = j - 1; k <= j + 1; k++) {
				if (rgbSum(p[i], k) + threashold > rgbSum(p[4], j) && (i != 4 || k != j)) {
					return false;
				}
			}
		}
		return true;
	}
	/*bool isnotEdge(Mat* m, int mWidth, int mHeight, int iSrc, int jSrc) {
		uchar* p[5];
		float fux, fuy, fdx, fdy, flx, fly, frx, fry;
		float sxy, sxx, syy, syx;
		float tr, det;
		float key;
		float threashold = edgeThreashold;
		int scale = 2;
		int i = iSrc * m->cols / mWidth;
		int j = jSrc * m->cols / mWidth;


		if (edgeTest == 0) {
			return true;
		}
		else {
			if (i < scale * 2 || i >= m->rows - scale * 2 || j < scale * 2 || j >= m->cols - scale * 2) {
				return false;
			}
			else {
				p[0] = m->ptr(i - scale * 2);
				p[1] = m->ptr(i - scale);
				p[2] = m->ptr(i);
				p[3] = m->ptr(i + scale);
				p[4] = m->ptr(i + scale * 2);


				fux = (rgbSum(p[1], j + scale) - rgbSum(p[1], j - scale)) / 2 / scale;
				fdx = (rgbSum(p[3], j + scale) - rgbSum(p[3], j - scale)) / 2 / scale;
				flx = (rgbSum(p[2], j + scale * 2) - rgbSum(p[2], j)) / 2 / scale;
				frx = (rgbSum(p[2], j) - rgbSum(p[2], j - scale * 2)) / 2 / scale;
				fuy = (rgbSum(p[2], j) - rgbSum(p[0], j)) / 2 / scale;
				fdy = (rgbSum(p[4], j) - rgbSum(p[2], j)) / 2 / scale;
				fly = (rgbSum(p[3], j - scale) - rgbSum(p[1], j - scale)) / 2 / scale;
				fry = (rgbSum(p[3], j + scale) - rgbSum(p[1], j + scale)) / 2 / scale;

				sxx = (frx - flx) / scale / 2;
				syy = (fdy - fuy) / scale / 2;
				sxy = (fry - fly) / scale / 2;
				syx = (fdx - fux) / scale / 2;

				tr = sxx + syy;
				det = sxx * syy - sxy * syx;

				key = tr * tr / abs(det);
				if (key > threashold) {
					return false;
				}
				else {
					return true;
				}
			}
		}
	}*/
	bool isnotEdge2(Mat* m, int mWidth, int mHeight, int iSrc, int jSrc) {
		float i = (float)(iSrc * m->cols) / (float)(mWidth);
		float j = (float)(jSrc * m->cols) / (float)(mWidth);
		float deltaSita = 5 * pi / 180;
		float sita = 0;
		float fux, fuy, fdx, fdy, flx, fly, frx, fry, sxx, syy, sxy, syx;
		float x[13], y[13];
		float key, tr, det;

		if (i<edgeWidth*2 + 2 || i>=m->rows - edgeWidth*2 - 2 || j<edgeWidth*2 + 2 || j>=m->cols - edgeWidth*2 - 2) {
			return false;
		}

		for (int k = 0; k < (int)(2 * pi / deltaSita); k++) {
			x[1] = i - edgeWidth * cos(sita); x[0] = x[1] + edgeWidth * sin(sita); x[2] = x[1] - edgeWidth * sin(sita);
			x[4] = i; x[3] = x[4] + edgeWidth * sin(sita); x[5] = x[4] - edgeWidth * sin(sita);
			x[7] = i + edgeWidth * cos(sita); x[6] = x[7] + edgeWidth * sin(sita); x[8] = x[7] - edgeWidth * sin(sita);
			x[9] = i - edgeWidth * cos(sita) * 2; x[10] = x[4] + edgeWidth * sin(sita) * 2;
			x[11] = x[4] - edgeWidth * sin(sita) * 2; x[12] = i + edgeWidth * cos(sita) * 2;

			y[1] = j - edgeWidth * sin(sita); y[0] = y[1] - edgeWidth * cos(sita); y[2] = y[1] + edgeWidth * cos(sita);
			y[4] = j; y[3] = y[4] - edgeWidth * cos(sita); y[5] = y[4] + edgeWidth * cos(sita);
			y[7] = j + edgeWidth * sin(sita); y[6] = y[7] - edgeWidth * cos(sita); y[8] = y[7] + edgeWidth * cos(sita);
			y[9] = j - edgeWidth * sin(sita) * 2; y[10] = y[4] - edgeWidth * cos(sita) * 2;
			y[11] = y[4] + edgeWidth * cos(sita) * 2; y[12] = j + edgeWidth * sin(sita) * 2;

			fux = (subPixelGray(m, x[2], y[2]) - subPixelGray(m, x[0], y[0])) / (edgeWidth * 2);
			fuy = (subPixelGray(m, x[4], y[4]) - subPixelGray(m, x[9], y[9])) / (edgeWidth * 2);
			flx = (subPixelGray(m, x[4], y[4]) - subPixelGray(m, x[10], y[10])) / (edgeWidth * 2);
			fly = (subPixelGray(m, x[6], y[6]) - subPixelGray(m, x[0], y[0])) / (edgeWidth * 2);
			frx = (subPixelGray(m, x[11], y[11]) - subPixelGray(m, x[4], y[4])) / (edgeWidth * 2);
			fry = (subPixelGray(m, x[8], y[8]) - subPixelGray(m, x[2], y[2])) / (edgeWidth * 2);
			fdx = (subPixelGray(m, x[8], y[8]) - subPixelGray(m, x[6], y[6])) / (edgeWidth * 2);
			fdy = (subPixelGray(m, x[12], y[12]) - subPixelGray(m, x[4], y[4])) / (edgeWidth * 2);

			sxx = (frx - flx) / (edgeWidth * 2);
			syy = (fdy - fuy) / (edgeWidth * 2);
			sxy = (fry - fly) / (edgeWidth * 2);
			syx = (fdx - fux) / (edgeWidth * 2);

			tr = sxx + syy;
			det = sxx * syy - sxy * syx;

			key = tr * tr / abs(det);

			if (key > edgeThreashold) {
				return false;
			}

			sita += deltaSita;
		}
		return true;
	}
	void featureCompute(Mat* m, float x, float y, Feature* featurePointer, float scale) {
		float mainDirection, direction, weight;
		float count[37] = { 0.0f };
		float max = 0, maxNum = 0;
		float xx, yy, r, sita, addition;
		int nox, noy, noangle;
		float sum = 0;
		Feature ff;
		static int first = 1;

		ff.clear();

		if (x >= scale*3+1 && x < m->rows - scale * 3 + 1 && y >= scale * 3 + 1 && y < m->cols - scale * 3 + 1) {
			for (int i = (int)(x - scale * 2); i <= (int)(x + scale * 2); i++) {
				for (int j = (int)(y - scale * 2); j <= (int)(y + scale * 2); j++) {
					featureVectorDirection(m, i, j, &direction, &weight);
					count[(int)((int)(direction * 180 / pi + 180) / 10) % 36] += weight * exp(-1 * ((x - i)*(x - i) + (y - j)*(y - j)) / (scale*scale));

					if (count[(int)((int)(direction * 180 / pi + 180) / 10) % 36] > max) {
						max = count[(int)((int)(direction * 180 / pi + 180) / 10) % 36];
						maxNum = (int)((int)(direction * 180 / pi + 180) / 10) % 36;
					}
				}
			}

			maxNum = maxNum + 0.5f*(count[((int)maxNum + 35) % 36] - count[((int)maxNum + 1) % 36]) / (count[((int)maxNum + 35) % 36] + count[((int)maxNum + 1) % 36] - 2.0f*count[(int)maxNum]);
			mainDirection = ((float)maxNum * 10.0f + 5.0f) * pi / 180.0f;
			if (mainDirection < 0)mainDirection += 2 * pi;
			if (mainDirection >= 2 * pi)mainDirection -= 2 * pi;
			featurePointer->maindirection = mainDirection;

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
						featureVectorDirection(m, i, j, &direction, &weight);
						direction += pi;
						direction -= mainDirection;
						noangle = (int)((int)(direction * 180 / pi + 180) / 45) % 8;
						for (int vi = 0; vi < 4; vi++) {
							for (int vj = 0; vj < 4; vj++) {
								for (int vk = 0; vk < 8; vk++) {
									ff.vector[vi][vj][vk] += weight * exp(-1 * distance((float)(x + (vi - 1.5)*scale), (float)(y + (vj - 1.5)*scale), (float)vk, xx, yy, direction*4.0f / pi) / 2);
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
						featurePointer->vector[ii][jj][kk] = ff.vector[ii][jj][kk] / sum;
					}
				}
			}
		}
	}
	void featureVectorDirection(Mat* m, int x, int y, float* direction, float* weight) {
		if (x > 0 && x < m->rows - 1 && y>0 && y < m->cols - 1) {
			*direction = (float)atan2(rgbSum(m->ptr(x), y + 1) - rgbSum(m->ptr(x), y - 1), rgbSum(m->ptr(x + 1), y) - rgbSum(m->ptr(x - 1), y));
			*weight = sqrt(pow(rgbSum(m->ptr(x), y + 1) - rgbSum(m->ptr(x), y - 1), 2) + pow(rgbSum(m->ptr(x + 1), y) - rgbSum(m->ptr(x - 1), y), 2));
		}
	}
	float distance(float xsrc, float ysrc, float anglesrc, float xtrg, float ytrg, float angletrg) {
		float deltaangle = abs(anglesrc - angletrg);
		if (deltaangle > 18) {
			deltaangle = 36 - deltaangle;
		}
		return (float)(pow(xsrc - xtrg, 2) + pow(ysrc - ytrg, 2) + pow(deltaangle, 2));
	}
};
class DoGPyramid {
public:
	DoG layers[10];
	void update(Mat* m,int num) {
		layers[0].update(m, 5, sigma, k);
		layers[0].compute();
		for (int i = 1; i < num; i++) {
			Mat middle;
			minimize(&layers[i - 1].g[2], &middle);
			layers[i].update(&middle, 5, sigma, k);
			layers[i].compute();
		}
	}
	void minimize(Mat* src, Mat* out) {
		uchar *pSrc, *pOut;
		out->create(src->rows / 2, src->cols / 2, src->type());
		for (int i = 0; i < out->rows; i++) {
			pSrc = src->ptr(i * 2);
			pOut = out->ptr(i);
			for (int j = 0; j < out->cols; j++) {
				pOut[j * 3] = pSrc[j * 6];
				pOut[j * 3 + 1] = pSrc[j * 6 + 1];
				pOut[j * 3 + 2] = pSrc[j * 6 + 2];
			}
		}
	}
};

DoG dog;
DoGPyramid pyramid;

float rgbSum(uchar* p, int j) {
	return float(p[j * 3] + p[j * 3 + 1] + p[j * 3 + 2]);
}
float subPixelGray(Mat* m, float x, float y) {
	int xO = (int)x, yO = (int)y;
	float a, b, c, d, e, f;
	float xx = x - xO, yy = y - yO;
	uchar* p[3];

	p[0] = m->ptr(xO - 1);
	p[1] = m->ptr(xO);
	p[2] = m->ptr(xO + 1);
	
	f = rgbSum(p[1], yO);
	a = 0.5f*(rgbSum(p[2], yO) + rgbSum(p[0], yO)) - rgbSum(p[1], yO);
	b = 0.5f*(rgbSum(p[1], yO + 1) + rgbSum(p[1], yO - 1)) - rgbSum(p[1], yO);
	d = 0.5f*(rgbSum(p[2], yO) - rgbSum(p[0], yO));
	e = 0.5f*(rgbSum(p[1], yO + 1) - rgbSum(p[1], yO - 1));
	c = 0.25f*((rgbSum(p[2], yO + 1) + rgbSum(p[0], yO - 1)) - (rgbSum(p[0], yO + 1) + rgbSum(p[2], yO - 1)));

	return a * xx*xx + b * yy*yy + c * xx*yy + d * xx + e * yy + f;
}
void normalize(Mat* m) {
	uchar* p;
	uchar max = 0;
	float k;
	for (int i = 0; i < m->rows; i++) {
		p = m->ptr(i);
		for (int j = 0; j < m->cols; j++) {
			if ((p[j * 3] + p[j * 3 + 1] + p[j * 3 + 2]) > max) {
				max = (p[j * 3] + p[j * 3 + 1] + p[j * 3 + 2]);
			}
		}
	}
	k = 255.0f*3.0f / (float)max;
	for (int i = 0; i < m->rows; i++) {
		p = m->ptr(i);
		for (int j = 0; j < m->cols * 3; j++) {
			p[j] = (uchar)(k*p[j]);
		}
	}
}
void keyCallback(GLFWwindow* window, int a, int b, int c , int d) {
	if (a == 32) {
		glfwSetWindowShouldClose(window, true);
	}
	if (a == 48 && c==1) {
		edgeTest = 1 - edgeTest;
		if (edgeTest) {
			cout << "on" << endl;
		}
		else {
			cout << "off" << endl;
		}
	}
	if (a == 49 && c == 1) {
		resetTarget = 1;
		cout << "reset target" << endl;
	}
	if (a == 50 && c == 1) {
		match = 1 - match;
		if (match) {
			cout << "match enabled" << endl;
		}
		else {
			cout << "match disabled" << endl;
		}
	}
}
void cursorCallback(GLFWwindow* window, double x, double y) {
	GLubyte r, g, b;
	uchar* p;
	int xTrue, yTrue;
	xTrue = (int)(((y - 240) / 0.9 + 240));
	yTrue = (int)(((x - 320) / 0.9 + 320));
	if (xTrue < 0 || xTrue>=480 || yTrue < 0 || yTrue>=640) {
		str[2] = '0'; str[3] = '0'; str[4] = '0';
		str[8] = '0'; str[9] = '0'; str[10] = '0';
		str[14] = '0'; str[15] = '0'; str[16] = '0';
	}
	else {
		p = img.ptr(xTrue*img.cols / 640);
		b = p[yTrue*img.cols / 640 * 3];
		g = p[yTrue*img.cols / 640 * 3 + 1];
		r = p[yTrue*img.cols / 640 * 3 + 2];
		str[2] = (r / 100) + 48; str[3] = ((r / 10) % 10) + 48; str[4] = (r % 10) + 48;
		str[8] = (g / 100) + 48; str[9] = ((g / 10) % 10) + 48; str[10] = (g % 10) + 48;
		str[14] = (b / 100) + 48; str[15] = ((b / 10) % 10) + 48; str[16] = (b % 10) + 48;
	}

}
void glfwWindowInit(void) {
	glfwInit();
	window = glfwCreateWindow(640, 480, "camera", NULL, NULL);
	glfwMakeContextCurrent(window);
	glViewport(0, 0, 640, 480);
	glfwSetKeyCallback(window, keyCallback);
	glfwSetCursorPosCallback(window, cursorCallback);
}
void shaderInit(void) {
	glewInit();

	vertexShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
	glCompileShader(vertexShader);
	glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
	if (!success) {
		cout << "vertexShader compile failed" << endl;
	}

	fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
	glCompileShader(fragmentShader);
	glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
	if (!success) {
		cout << "fragmentShader compile failed" << endl;
	}

	shaderProgram = glCreateProgram();
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);
	glLinkProgram(shaderProgram);

	glUseProgram(shaderProgram);

	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);
}
void drawInit(void) {
	glGenBuffers(1, &vbo);
	glGenVertexArrays(1, &vao);
	glUniform3f(glGetUniformLocation(shaderProgram, "target"), 131.0f / 225.0f, 214.0f / 225.0f, 255.0f / 225.0f);
}
void clear(void) {
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);
}
void draw(Mat* m) {
	uchar*p;
	int mWidth, mHeight, mHalfWidth, mHalfHeight;
	mWidth = m->cols;
	mHeight = m->rows;
	mHalfWidth = mWidth / 2;
	mHalfHeight = mHeight / 2;
	bool flag = true;
	int matchNo;

	clear();

	for (int i = 0; i < mHeight; i++) {
		p = m->ptr(i);
		for (int j = 0; j < mWidth; j++) {
			image[i * mWidth * 6 + j * 6 + 0] = ((float)j / mHalfWidth - 1)*0.9f;
			image[i * mWidth * 6 + j * 6 + 1] = (1 - (float)i / mHalfHeight)*0.9f;
			image[i * mWidth * 6 + j * 6 + 2] = 0;
			image[i * mWidth * 6 + j * 6 + 3] = (float)p[j * 3 + 2] / 255;
			image[i * mWidth * 6 + j * 6 + 4] = (float)p[j * 3 + 1] / 255;
			image[i * mWidth * 6 + j * 6 + 5] = (float)p[j * 3] / 255;
		}
	}
	for (int i = 0; i < pos; i++) {
		matchNo = 0;
		if (!resetTarget && match) {
			flag = false;
			for (int num = 0; num < targetNum; num++) {
				if (feature[i].match(&matchTarget[num])) {
					flag = true;
					matchNo = num;
					break;
				}
			}
		}
		for (int x = feature[i].x - 2; x <= feature[i].x + 2; x++) {
			for (int y = feature[i].y - 2; y <= feature[i].y + 2; y++) {
				if (x >= 0 && x < mHeight && y >= 0 && y < mWidth) {
					if (flag) {
						image[x * mWidth * 6 + y * 6 + 3] = 1.0f - (float)((matchNo * 10) % 256) / 255.0f;
						image[x * mWidth * 6 + y * 6 + 4] = (float)((matchNo * 10) % 256) / 255.0f;
						image[x * mWidth * 6 + y * 6 + 5] = 1.0f - (float)((matchNo * 10) % 256) / 255.0f;
					}
				}
			}
		}
	}


	glBindVertexArray(vao);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(image), &image, GL_STATIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GL_FLOAT), (GLvoid*)0);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GL_FLOAT), (GLvoid*)(3 * sizeof(GL_FLOAT)));
	glEnableVertexAttribArray(1);
	glDrawArrays(GL_POINTS, 0, mHeight * mWidth);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	if (resetTarget == 1) {
		resetTarget = 0;
		for (int i = 0; i < pos; i++) {
			feature[i].cloneto(&matchTarget[i]);
		}
		targetNum = pos;
	}
	pos = 0;
}
void displayWordsInit(void) {
	lists = glGenLists(128);
	wglUseFontBitmaps(wglGetCurrentDC(), 0, 128, lists);
}
void displayWords(char* cha) {
	glRasterPos2f(-0.95f, -0.96f);
	while (*cha != '\0') {
		glCallList(lists + *cha);
		cha++;
	}
}
void exit(void) {
	glDeleteVertexArrays(1, &vao);
	glDeleteBuffers(1, &vbo);
	cap.release();
	glfwTerminate();
}
void subtraction(Mat* a, Mat* b, Mat* out) {
	uchar *pa,*pb,*po;
	int mWidth, mHeight;
	mWidth = a->cols;
	mHeight = a->rows;
	out->create(a->rows, a->cols, a->type());

	for (int i = 0; i < mHeight; i++) {
		pa= a->ptr(i);
		pb = b->ptr(i);
		po = out->ptr(i);
		for (int j = 0; j < mWidth * 3; j++) {
			po[j] = pa[j] > pb[j] ? pa[j] - pb[j] : 0;
		}
	}
}
void display(void) {
	draw(&img);
	//displayWords(str);
	glfwSwapBuffers(window);
}
int main(int argc,char** argv) {
	glfwWindowInit();
	shaderInit();
	drawInit();
	displayWordsInit();
	cap.open(CAP_DSHOW+1);
	if (!cap.isOpened()) {
		cout << "camera is not opened" << endl;
		system("pause");
		return 0;
	}
	/*pic = imread("a.jpg");
	cv::Ptr<xfeatures2d::SIFT> detector = xfeatures2d::SIFT::create();
	vector<KeyPoint> keypoint, keypoint2;
	Mat imgSIFT, image, result;

	cap >> raw;
	detector->detectAndCompute(raw, noArray(), keypoint, descriptor);
	drawKeypoints(raw, keypoint, image);
	imshow("image", image);
	waitKey(10);
	BFMatcher matcher;
	vector<DMatch> match;*/

	while (!glfwWindowShouldClose(window)) {
		cap.read(raw);
		img = raw.clone();
		GaussianBlur(raw, preprocess, Size(0, 0), preBlur);

		/*detector->detectAndCompute(raw, noArray(), keypoint2, descriptor2);

		if (keypoint.size() > 0) {
			drawKeypoints(raw, keypoint2, imgSIFT);
			//imshow("SIFT", imgSIFT);
			//waitKey(10);

			matcher.match(descriptor, descriptor2, match);
			nth_element(match.begin(), match.begin() + 9, match.end());
			match.erase(match.begin() + 10, match.end());
			drawMatches(image, keypoint, raw, keypoint2, match, result);
			imshow("result", result);
			waitKey(10);
		}*/

		pyramid.update(&preprocess, 6);

		glfwPollEvents();
		display();
	}
	exit();
	return cap.isOpened();
}
