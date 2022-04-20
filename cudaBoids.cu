// cudaBoids.cu - Devon McKee, 2022

#include <glad.h>
#include <GLFW/glfw3.h>
#include "cuda_runtime.h"
#include "cuda_gl_interop.h"
#include "device_launch_parameters.h"
#include <vector>
#include "VecMat.h"
#include "Camera.h"
#include "CameraControls.h"
#include "Misc.h"
#include "GLXtras.h"
#include "GeomUtils.h"
#include "dCube.h"

#define cudaCheck(error) if (error != cudaSuccess) { printf("CUDA error: %s at %s:%d\n", cudaGetErrorString(error), __FILE__, __LINE__); exit(1); }

using std::vector;
using std::string;

GLuint renderProgram = 0;
GLuint boidBuffer = 0;
cudaGraphicsResource* boidBufferGraphRes;

// strange issue running more than 1259 boids, needs to be investigated further
const int STARTING_BOIDS = 1259;
const int POINT_SIZE = 2;
const int NUM_GPU_THREADS = 256;

// ----- Simulation constants -----
#define BOID_SPEED 0.005f
#define BOID_PERCEPTION 0.1f
#define WALL_RANGE 0.05f
#define ALIGNMENT_WEIGHT 1.0f
#define COHESION_WEIGHT 1.0f
#define SEPARATION_WEIGHT 1.0f

int win_width = 800, win_height = 800;
Camera camera((float)win_width / win_height, vec3(0, 0, 0), vec3(0, 0, -5));
GLFWwindow* window;
vec3 lightPos = vec3(1, 1, 0);
dCube cube;
cudaError_t c_stat;
const char* render_glsl_version = "#version 430";

float bgColor[4] = { 0.5f, 0.5f, 0.5f, 1.0f };
float boidColor[4] = { 0.0f, 0.0f, 0.0f, 1.0f };

// ----- Float3 operators -----
__host__ __device__ float3 operator-(const float3 & v) { return float3{ -v.x, -v.y, -v.z }; }
__host__ __device__ float3 operator+(const float3 & l, const float3 & r) { return float3{ l.x + r.x, l.y + r.y, l.z + r.z }; }
__host__ __device__ float3 operator-(const float3 & l, const float3 & r) { return float3{ l.x - r.x, l.y - r.y, l.z - r.z }; }
__host__ __device__ float3 operator*(const float3 & l, const float3 & r) { return float3{ l.x * r.x, l.y * r.y, l.z * r.z }; }
__host__ __device__ float3 operator*(const float3 & l, float r) { return float3{ l.x * r, l.y * r, l.z * r }; }
__host__ __device__ float3 operator/(const float3 & l, float r) { float _d = 1.f / r; return l * _d; }
__host__ __device__ float b_dist(float3 p1, float3 p2) { return (float)sqrt(pow(p2.x - p1.x, 2) + pow(p2.y - p1.y, 2) + pow(p2.z - p1.z, 2)); }
__host__ __device__ float b_dot(const float3 & a, const float3 & b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__host__ __device__ float b_length(const float3 & v) { return sqrt(b_dot(v, v)); }
__host__ __device__ float3 b_normalize(const float3 & v) { return v / b_length(v); }

struct Boid;

__global__ void boidKernel(Boid* b, size_t n_boids);

struct Boid {
	float3 pos, vel, col;
	Boid() {
		pos = float3{ rand_float(-1.0f, 1.0f), rand_float(-1.0f, 1.0f), rand_float(-1.0f, 1.0f) };
		vel = b_normalize(float3{ rand_float(-1.0f, 1.0f), rand_float(-1.0f, 1.0f), rand_float(-1.0f, 1.0f) }) * BOID_SPEED;
		col = float3{ boidColor[0], boidColor[1], boidColor[2] };
	}
};

Boid* boids;
Boid* boids_dp;
size_t n_boids = STARTING_BOIDS;

void openGLErrorCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* userParam) {
	fprintf(stderr, "GL CALLBACK: %s type = 0x%x, severity = 0x%x, message = %s\n", (type == GL_DEBUG_TYPE_ERROR ? "** GL ERROR **" : ""), type, severity, message);
}

void compileShaders() {
	renderProgram = LinkProgramViaFile("shaders/render.vert", "shaders/render.frag");
	if (!renderProgram) {
		fprintf(stderr, "SHADER: Error linking render shader! Exiting...\n");
		exit(1);
	}
}

void b_initialize() {
	cube.loadBuffer();
	boids = new Boid[n_boids];
	size_t b_size = sizeof(Boid) * n_boids;
	glGenBuffers(1, &boidBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, boidBuffer);
	glBufferData(GL_ARRAY_BUFFER, b_size, boids, GL_STATIC_COPY);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	cudaCheck(cudaGraphicsGLRegisterBuffer(&boidBufferGraphRes, boidBuffer, cudaGraphicsRegisterFlagsNone));
	cudaCheck(cudaGraphicsMapResources(1, &boidBufferGraphRes, 0));
	cudaCheck(cudaGraphicsResourceGetMappedPointer((void**)&boids_dp, &b_size, boidBufferGraphRes));
}

void b_terminate() {
	cube.unloadBuffer();
	cudaCheck(cudaGraphicsUnmapResources(1, &boidBufferGraphRes, 0));
	cudaCheck(cudaGraphicsUnregisterResource(boidBufferGraphRes));
	glDeleteBuffers(1, &boidBuffer);
	delete boids;
}

void compute() {
	// Dispatch kernel
	int num_blocks = (int)floor(n_boids / NUM_GPU_THREADS) + (n_boids % NUM_GPU_THREADS == 0 ? 0 : 1);
	//printf("Dispatching CUDA with %d threads and %d blocks\n", NUM_GPU_THREADS, num_blocks);
	boidKernel<<<num_blocks, NUM_GPU_THREADS>>>(boids_dp, n_boids);
	cudaCheck(cudaGetLastError());
	cudaCheck(cudaDeviceSynchronize());
}

void display() {
	glClearColor(bgColor[0], bgColor[1], bgColor[2], bgColor[3]);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	cube.display(camera);
	// Render boids
	glUseProgram(renderProgram);
	glPointSize(POINT_SIZE);
	glBindBuffer(GL_ARRAY_BUFFER, boidBuffer);
	VertexAttribPointer(renderProgram, "point", 3, sizeof(Boid), (GLvoid*)offsetof(Boid, pos));
	VertexAttribPointer(renderProgram, "color", 3, sizeof(Boid), (GLvoid*)offsetof(Boid, col));
	glUniform4f(0, boidColor[0], boidColor[1], boidColor[2], boidColor[3]);
	SetUniform(renderProgram, "persp", camera.persp);
	SetUniform(renderProgram, "modelview", camera.modelview);
	glDrawArrays(GL_POINTS, 0, (int)n_boids);
	glFlush();
}

int main() {
	srand((int)time(NULL));
	c_stat = cudaSetDevice(0);
	if (c_stat != cudaSuccess) { printf("No CUDA-capable GPU found! Exiting...\n"); return 1; }
	if (!glfwInit()) return 1;
	window = glfwCreateWindow(win_width, win_height, "Cuda Boids", NULL, NULL);
	if (!window) { glfwTerminate(); return 1; }
	glfwSetWindowPos(window, 100, 100);
	glfwMakeContextCurrent(window);
	gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
	PrintGLErrors();
	compileShaders();
	glfwWindowHint(GLFW_SAMPLES, 4);
	glfwSwapInterval(1);
	InitializeCallbacks(window);
	//glEnable(GL_DEBUG_OUTPUT);
	glDebugMessageCallback(openGLErrorCallback, 0);
	b_initialize();
	double lastFrame = 0, lastSim = 0;
	while (!glfwWindowShouldClose(window)) {
		double now = glfwGetTime();
		double deltaTime = now - lastFrame;
		if ((now - lastSim) >= (1.0 / 60)) {
			compute();
		}
		display();
		glfwPollEvents();
		glfwSwapBuffers(window);
	}
	b_terminate();
	glfwDestroyWindow(window);
	glfwTerminate();
}

// ----- DEVICE CODE -----

__device__ void b_findNeighbors(Boid b, Boid* boids, size_t n_boids, int* nb, size_t &n_nb) {
	for (size_t i = 0; i < n_boids; i++) {
		float d = b_dist(b.pos, boids[i].pos);
		if (&boids[i] != &b && d < BOID_PERCEPTION && d > 0)
			nb[n_nb++] = i;
	}
}

__device__ float3 b_alignment(Boid b, Boid* boids, int* nb, size_t n_nb) {
	float3 cv = { 0.0f };
	int nc = 0;
	for (size_t i = 0; i < n_nb; i++) {
		size_t n = nb[i];
		cv = cv + boids[n].vel;
		nc++;
	}
	if (nc > 0) {
		cv = cv / (float)nc;
		cv = b_normalize(cv);
		return cv;
	} else {
		return float3{ 0.0f };
	}
}

__device__ float3 b_cohesion(Boid b, Boid* boids, int* nb, size_t n_nb) {
	float3 cv{ 0.0f };
	int nc = 0;
	for (size_t i = 0; i < n_nb; i++) {
		size_t n = nb[i];
		cv = cv + boids[n].pos;
		nc++;
	}
	if (nc > 0) {
		cv = cv / (float)nc;
		cv = cv - b.pos;
		cv = b_normalize(cv);
		return cv;
	} else {
		return float3{ 0.0f };
	}
}

__device__ float3 b_separation(Boid b, Boid* boids, int* nb, size_t n_nb) {
	float3 cv{ 0.0f };
	float nc = 0;
	for (size_t i = 0; i < n_nb; i++) {
		size_t n = nb[i];
		float3 iv = b.pos - boids[n].pos;
		iv = b_normalize(iv);
		iv = iv / b_dist(b.pos, boids[n].pos);
		cv = cv + iv;
		nc++;
	}
	if (nc > 0) {
		cv = cv / nc;
		cv = b_normalize(cv);
		return cv;
	} else {
		return float3{ 0.0f };
	}
}

__device__ float3 b_avoidance(Boid &b) {
	if (b.pos.x > 1.0f) b.pos.x = -1.0f;
	if (b.pos.x < -1.0f) b.pos.x = 1.0f;
	if (b.pos.y > 1.0f) b.pos.y = -1.0f;
	if (b.pos.y < -1.0f) b.pos.y = 1.0f;
	if (b.pos.z > 1.0f) b.pos.z = -1.0f;
	if (b.pos.z < -1.0f) b.pos.z = 1.0f;
	if (b.pos.y > 1 - WALL_RANGE) return float3{ 0.0f, 1 / (-1 - b.pos.y), 0.0f }; // top wall
	if (b.pos.y < -1 + WALL_RANGE) return float3{ 0.0f, 1 / (1 - b.pos.y), 0.0f }; // bottom wall
	return float3{ 0.0f };
}

__global__ void boidKernel(Boid* boids, size_t n_boids) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (idx < n_boids) {
		Boid b = boids[idx];
		// Find neighbors of boid
		int* nb = new int[n_boids];
		size_t n_nb = 0;
		b_findNeighbors(b, boids, n_boids, nb, n_nb);
		// Calculate vectors of influence on boid
		float3 a_vec = b_alignment(b, boids, nb, n_nb) * ALIGNMENT_WEIGHT;
		float3 c_vec = b_cohesion(b, boids, nb, n_nb) * COHESION_WEIGHT;
		float3 s_vec = b_separation(b, boids, nb, n_nb) * SEPARATION_WEIGHT;
		float3 w_vec = b_avoidance(b);
		b.vel = b.vel + a_vec + c_vec + s_vec + w_vec;
		b.vel = b_normalize(b.vel);
		b.vel = b.vel * BOID_SPEED;
		b.pos = b.pos + b.vel;
		delete nb;
		float mp = 1 / BOID_SPEED;
		b.col = float3{ (b.vel.x * mp + 1) / 2, (b.vel.y * mp + 1) / 2, (b.vel.z * mp + 1) / 2 };
		boids[idx] = b;
	}
}