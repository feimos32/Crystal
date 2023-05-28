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

#ifndef __Common_h__
#define __Common_h__

#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <limits>

namespace CrystalAlgrithm {

#define HOST_AND_DEVICE __host__ __device__
#define HOST __host__
#define DEVICE __device__
#define GPU_CALL __global__

#define Float float

#define M_PI 3.1415926f
#define OneOver4PI 0.07957747f
#define INV_M_PI 0.3183099f
#define ANGLE(angle) (angle*M_PI/180.0f)

#define MaxFloat std::numeric_limits<Float>::max()
#define Infinity std::numeric_limits<Float>::infinity()

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

std::string getCudaDebug_NULL(const char* file, int line);

std::string getCudaDebug_ERROR(const char* error, const char* file, int line);

bool getCudaError(cudaError_t err, const char* file, int line);

bool getCudaError(cudaError_t err);

#define Get_CUDA_ERROR( err ) (getCudaError( err, __FILE__, __LINE__ ))

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
	GpuInfo gpu[4];
};



}





#endif






