/*
Copyright (C) <2023>  <Dezeming>  <feimos@mail.ustc.edu.cn>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or any
later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Github site: <https://github.com/feimos32/Crystal>
*/

#ifndef __Common_cuh__
#define __Common_cuh__

#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <limits>
#include <iostream>

namespace CrystalAlgrithm {

#define HOST_AND_DEVICE __host__ __device__
#define HOST __host__
#define DEVICE __device__
#define GPU_CALL __global__

#define Float float



// Some Numerical definations

#define M_PI 3.1415926f
#define OneOver4PI 0.07957747f
#define INV_M_PI 0.3183099f
#define ANGLE(angle) (angle * M_PI / 180.0f)

#define MaxFloat std::numeric_limits<Float>::max()
#define Infinity std::numeric_limits<Float>::infinity()


// Some Utility functions

HOST_AND_DEVICE inline Float Clamp(Float val, Float low, Float high) {
	if (val < low)
		return low;
	else if (val > high)
		return high;
	else
		return val;
}
HOST_AND_DEVICE inline Float Lerp(Float t, Float v1, Float v2) { return (1 - t) * v1 + t * v2; }

const Float largeValue = 9999999.0;


// Print CUDA Runtime Information

std::string getCudaDebug_NULL(const char* file, int line);

std::string getCudaDebug_ERROR(const char* error, const char* file, int line);

bool getCudaError(cudaError_t err, const char* file, int line);

bool getCudaError(cudaError_t err);

#define Get_CUDA_ERROR( err ) (getCudaError( err, __FILE__, __LINE__ ))


// Print Debug Information

inline void PrintError_Std(std::string err, const char* file, int line) {
	std::cout << "[Error]: " << std::string(err) + " in " + std::string(file) +
		" at line " + std::to_string(line) << std::endl;
}

#define PrintError( err ) (PrintError_Std( err, __FILE__, __LINE__ ))

inline void PrintValue_Std(std::string info, float s) {
	std::cout << "[Debug]: " << info + ": [" + std::to_string(s) + "]" << std::endl;
}
inline void PrintValue_Std(std::string info, size_t s) {
	std::cout << "[Debug]: " << info + ": [" + std::to_string(s) + "]" << std::endl;
}
inline void PrintValue_Std(std::string info, int s) {
	std::cout << "[Debug]: " << info + ": [" + std::to_string(s) + "]" << std::endl;
}

#define PrintValue(info, val) PrintValue_Std(info, val);

inline void PrintValue_Std(float s) {
	std::cout << "[Debug]: " << "[" + std::to_string(s) + "]" << std::endl;
}
inline void PrintValue_Std(int s) {
	std::cout << "[Debug]: " << "[" + std::to_string(s) + "]" << std::endl;
}
inline void PrintValue_Std(std::string s) {
	std::cout << "[Debug]: " << "[" + s + "]" << std::endl;
}

// GPU Information

struct GpuInfo {
	std::string GpuName;
	std::string ComputeCapability;
	int ClockRate;
	size_t TotalGlobalMemory; // GB
	size_t TotalConstantMemory; //KB
	size_t MultiprocessorCount;
};

class GpuDeviceInfos {
public:
	int GpuCount;
	GpuInfo gpu[6];
};



}





#endif






