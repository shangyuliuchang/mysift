#include<iostream>
#include<GL/glew.h>
#include<GL/glut.h>
#include<GLFW/glfw3.h>
#include<glm/glm.hpp>
#include<glm/gtc/matrix_transform.hpp>
#include<glm/gtc/type_ptr.hpp>
#include<SOIL.h>

#pragma comment(lib,"OpenGL32.lib")

using namespace std; 

GLfloat vertices[] = {
	0.0f,0.0f,0.0f,1.0f,1.0f,1.0f,0.0f,1.0f,
	0.5f,0.0f,0.0f,1.0f,1.0f,1.0f,1.0f,1.0f,
	0.5f,0.5f,0.0f,1.0f,1.0f,1.0f,1.0f,0.0f,
	0.0f,0.5f,0.0f,1.0f,1.0f,1.0f,0.0f,0.0f,

	0.0f,0.0f,0.5f,1.0f,1.0f,1.0f,1.0f,1.0f,
	0.5f,0.0f,0.5f,1.0f,1.0f,1.0f,0.0f,1.0f,
	0.5f,0.5f,0.5f,1.0f,1.0f,1.0f,0.0f,0.0f,
	0.0f,0.5f,0.5f,1.0f,1.0f,1.0f,1.0f,0.0f
};
GLuint element[] = {
	0,1,2,
	2,3,0,
	4,5,6,
	6,7,4
	/*0,1,5,
	0,4,5,
	1,5,6,
	1,2,6,
	0,3,7,
	0,4,7,
	2,6,7,
	2,3,7*/
};
glm::vec3 mymove;
const GLchar* vertexShaderSource =
"#version 330 core\n"
"layout (location=0) in vec3 position;\n"
"layout (location=1) in vec3 acolor;\n"
"layout (location=2) in vec2 texcoord;\n"
"out vec3 color;\n"
"out vec3 ourcolor;\n"
"out vec2 ourtexcoord;\n"
"uniform mat4 matrix;\n"
"uniform vec3 move;\n"
"uniform float scale;\n"
"void main(){\n"
"ourtexcoord=texcoord;\n"
"gl_Position=matrix*vec4(position.x+move.x,position.y+move.y,position.z+move.z,1.0f);\n"
"gl_Position*=vec4(scale,scale,scale,1.0f);\n"
//"color=vec3(position.x*2,position.y*2,position.z*2);\n"
"color=acolor;\n"
"//color=vec3(1.0f,1.0f,1.0f);\n"
"}\0";
const GLchar* fragementShaderSource =
"#version 330 core\n"
"in vec3 color;\n"
"in vec2 ourtexcoord;\n"
"out vec4 FragColor;\n"
"uniform sampler2D mytexture;\n"
"vec4 mycolor;\n"
"void main(){\n"
"mycolor=texture(mytexture,ourtexcoord);\n"
"FragColor=vec4(mycolor.r*color.r,mycolor.g*color.g,mycolor.b*color.b,0.5f);}\0";
//"FragColor=vec4(color,1.0f);}\0";
GLuint vbo, vao, vertexShader, fragmentShader, shaderProgram, ebo, texture;
GLint success;
glm::mat4 trans, back;

int height, width;
float m[16] = { 0 };
double pi = 3.1415926535;
double angle = 0;
double sita = angle * pi / 180;
int mouse_down = 0;
int windowWidth = 500, windowHeight = 500;
double xp, yp, zp;
int first=1;
double r = sqrt(windowHeight*windowHeight+windowWidth*windowWidth)/2;
float scale = 1;
int scrollSum = 0;
FILE* file;
GLint imgWidth, imgHeight, pixelLength;
//GLubyte* imgdata;
//GLubyte mydata[512][512][3];


void callback(GLFWwindow* window, int key, int scancode, int action, int mode) {
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, GL_TRUE);
	}
	if (key == GLFW_KEY_LEFT && action == GLFW_PRESS) {

	}
	if (key == GLFW_KEY_RIGHT && action == GLFW_PRESS) {
		
	}
	sita = angle * pi / 180;
}
void mouse(GLFWwindow* window,double xx, double yy) {
	double x, y, z, a, b, c, d, alpha, xtrue, ytrue, ztrue;
	glm::vec3 xaxis, yaxis, zaxis;
	x = xx - windowWidth / 2;
	y = -yy + windowHeight / 2;
	if (x*x + y * y <= r*r) {
		z = -sqrt(r*r - x * x - y * y);
		if (mouse_down) {
			if (first) {
				first = 0;
				xp = x;
				yp = y;
				zp = z;
			}
			else {
				d = sqrt((x - xp)*(x - xp) + (y - yp)*(y - yp) + (z - zp)*(z - zp));
				a = yp * z - zp * y;
				b = zp * x - xp * z;
				c = xp * y - yp * x;
				alpha = asin(d / 2 / r) * 4;
				xaxis = glm::vec3((trans*glm::vec4(1.0f, 0.0f, 0.0f, 1.0f)).x, (trans*glm::vec4(1.0f, 0.0f, 0.0f, 1.0f)).y, (trans*glm::vec4(1.0f, 0.0f, 0.0f, 1.0f)).z);
				yaxis = glm::vec3((trans*glm::vec4(0.0f, 1.0f, 0.0f, 1.0f)).x, (trans*glm::vec4(0.0f, 1.0f, 0.0f, 1.0f)).y, (trans*glm::vec4(0.0f, 1.0f, 0.0f, 1.0f)).z);
				zaxis = glm::vec3((trans*glm::vec4(0.0f, 0.0f, 1.0f, 1.0f)).x, (trans*glm::vec4(0.0f, 0.0f, 1.0f, 1.0f)).y, (trans*glm::vec4(0.0f, 0.0f, 1.0f, 1.0f)).z);
				xtrue = a * xaxis.x + b * xaxis.y + c * xaxis.z;
				ytrue = a * yaxis.x + b * yaxis.y + c * yaxis.z;
				ztrue = a * zaxis.x + b * zaxis.y + c * zaxis.z;
				trans = glm::rotate(trans, (float)(alpha), glm::vec3(xtrue, ytrue, ztrue));
				xp = x;
				yp = y;
				zp = z;
			}
		}
	}
}
void mouse_pressed(GLFWwindow* window,int state,int p,int y) {
	mouse_down = p;
	if (p == 0) {
		first = 1;
	}
}
void scroll(GLFWwindow* window,double a,double b) {
	scrollSum += (int)b;
	scale = (float)pow(1.1, (double)scrollSum);
}
void draw(float x, float y, float z) {
	GLint move_l = glGetUniformLocation(shaderProgram, "move");
	mymove = glm::vec3(-0.25f + x, -0.25f + y, -0.25f + z);
	glUniform3f(move_l, mymove.x, mymove.y, mymove.z);
	glBindVertexArray(vao);
	glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, (GLvoid*)0);
	glBindVertexArray(0);
}
void drawFace(int num) {
	GLint mymove = glGetUniformLocation(shaderProgram, "move");
	glUniform3f(mymove, -0.25f, -0.25f, -0.25f);
	glBindVertexArray(vao);
	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, (GLvoid*)(num * 6 * sizeof(unsigned int)));
	glBindVertexArray(0);
}
float getZ(int i) {
	float x, y, z;
	glm::vec4 in, out;
	x = vertices[i * 32];
	y = vertices[i * 32 + 1];
	z = vertices[i * 3 + 2];
	in = glm::vec4(x, y, z, 1.0f);
	out = trans * in;
	return out.z;
}
int main(void) {
	/*file = fopen("a.bmp", "rb");
	fseek(file, 0x0012,SEEK_SET);
	fread(&imgWidth, sizeof(imgWidth), 1, file);
	fread(&imgHeight, sizeof(imgHeight), 1, file);
	pixelLength = imgWidth * 3;
	while (pixelLength % 4 != 0)pixelLength++;
	pixelLength *= imgHeight;
	imgdata = (GLubyte*)malloc(pixelLength);
	fseek(file, 54, SEEK_SET);
	fread(imgdata, pixelLength, 1, file);
	for (int i = 0; i < imgHeight; i++) {
		for (int j = 0; j < imgWidth; j++) {
			mydata[i][j][0] = imgdata[j * 3 + i * imgWidth * 3];
			mydata[i][j][1] = imgdata[j * 3 + i * imgWidth * 3 + 1];
			mydata[i][j][2] = imgdata[j * 3 + i * imgWidth * 3 + 2];
		}
	}
	fclose(file);*/

	glfwInit();
	GLFWwindow* window = glfwCreateWindow(windowWidth, windowHeight, "first", nullptr, nullptr);
	glfwMakeContextCurrent(window);
	glewInit();
	glfwGetWindowSize(window, &width, &height);
	glViewport(0, 0, width, height);
	glfwSetKeyCallback(window, callback);
	glfwSetCursorPosCallback(window, mouse);
	glfwSetMouseButtonCallback(window, mouse_pressed);
	glfwSetScrollCallback(window, scroll);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_DST_ALPHA);

	vertexShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
	glCompileShader(vertexShader);
	glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
	if (!success) {
		cout << "vsfailed" << endl;
	}
	fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragmentShader, 1, &fragementShaderSource, NULL);
	glCompileShader(fragmentShader);
	glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
	if (!success) {
		cout << "fsofailed" << endl;
	}
	shaderProgram = glCreateProgram();
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);
	glLinkProgram(shaderProgram);
	glUseProgram(shaderProgram);
	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);

	glGenVertexArrays(1, &vao);
	glGenBuffers(1, &vbo);
	glGenBuffers(1, &ebo);
	glBindVertexArray(vao);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_DYNAMIC_DRAW);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(element), element, GL_DYNAMIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(GL_FLOAT), (GLvoid*)0);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(GL_FLOAT), (GLvoid*)(3 * sizeof(GL_FLOAT)));
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(GL_FLOAT), (GLvoid*)(6 * sizeof(GL_FLOAT)));
	glEnableVertexAttribArray(2);
	glBindVertexArray(0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	unsigned char* image = SOIL_load_image("a.bmp", &imgWidth, &imgHeight, 0, SOIL_LOAD_RGB);

	glGenTextures(1, &texture);
	glBindTexture(GL_TEXTURE_2D, texture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, imgWidth, imgHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, image);
	glGenerateMipmap(GL_TEXTURE_2D);

	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();
		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);
		glClear(GL_DEPTH_BUFFER_BIT);

		GLint scale_l = glGetUniformLocation(shaderProgram, "scale");
		GLint m_l = glGetUniformLocation(shaderProgram, "matrix");
		glUniformMatrix4fv(m_l, 1, GL_FALSE, glm::value_ptr(trans));
		glUniform1f(scale_l, scale);

		//draw(0, 0, 0);
		if (getZ(0) > getZ(1)) {
			drawFace(0);
			drawFace(1);
		}
		else {
			drawFace(1);
			drawFace(0);
		}

		/*for (float x = -7.5; x <= 7.5; x += 0.75) {
			for (float y = -7.5; y <= 7.5; y += 0.75) {
				for (float z = -7.5; z <= 7.5; z += 0.75) {
					draw(x, y, z);
				}
			}
			
		}*/
		
		glfwSwapBuffers(window);
	}
	glDeleteVertexArrays(1, &vao);
	glDeleteBuffers(1, &vbo);
	glfwTerminate();
	return 0;
}