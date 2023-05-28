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

#include <iostream>

#include "Common.cuh"
#include "Export_dll.cuh"

namespace CrystalAlgrithm {

void printCudaDevice() {
	std::cout << "Cuda Support Devices" << std::endl;

	cudaDeviceProp prop;
	int count;
	bool flag = Get_CUDA_ERROR(cudaGetDeviceCount(&count));

	if (flag)
		for (int i = 0; i < count; i++) {
			bool flag1 = Get_CUDA_ERROR(cudaGetDeviceProperties(&prop, i));
			if (flag1) {
				std::cout << "GPU sequence " + std::to_string(i) << std::endl;
				std::cout << "  GPU name " << prop.name << std::endl;
				std::cout << "  Compute capability " << prop.major << "." << prop.minor << std::endl;
				std::cout << "  Clock rate " << prop.clockRate << std::endl;
				std::cout << "  Total global memory " << prop.totalGlobalMem / 1024 / 1024 / 1024 << "GB" << std::endl;
				std::cout << "  Total constant memory " << prop.totalConstMem / 1024 << "KB" << std::endl;
				std::cout << "  Multiprocessor count " << prop.multiProcessorCount << std::endl;
			}
		}
}

void cudaDevicesInit(GpuDeviceInfos& info) {
	cudaDeviceProp prop;
	int count;
	bool flag = Get_CUDA_ERROR(cudaGetDeviceCount(&count));
	if (0 == count) {
		info.GpuCount = 0;
		return;
	}
	int validCount = 0;
	for (int i = 0; i < count; i++) {
		bool flag1 = Get_CUDA_ERROR(cudaGetDeviceProperties(&prop, i));
		if (flag1) {

			info.gpu[validCount].GpuName = prop.name;
			info.gpu[validCount].ComputeCapability = std::to_string(prop.major) + "."
				+ std::to_string(prop.minor);
			info.gpu[validCount].ClockRate = prop.clockRate;
			info.gpu[validCount].TotalGlobalMemory = prop.totalGlobalMem / 1024 / 1024 / 1024;
			info.gpu[validCount].TotalConstantMemory = prop.totalConstMem / 1024;
			info.gpu[validCount].MultiprocessorCount = prop.multiProcessorCount;

			validCount++;
		}

	}
	info.GpuCount = validCount;

}

bool getCudaError(cudaError_t err, const char* file, int line) {
	if (err != cudaSuccess) {
		std::cout << getCudaDebug_ERROR(cudaGetErrorString(err), file, line) << std::endl;
		return false;
	}
	else {
		return true;
	}
}

bool getCudaError(cudaError_t err) {
	if (err != cudaSuccess) {
		std::cout << cudaGetErrorString(err) << std::endl;
		return false;
	}
	else {
		return true;
	}
}

std::string getCudaDebug_ERROR(const char* error, const char* file, int line) {
	return (std::string(error) + " in " + std::string(file) + " at line " + std::to_string(line));
	//exit(EXIT_FAILURE);
}

std::string getCudaDebug_NULL(const char* file, int line) {
	return ("Host memory failed in " + std::string(file) + " at line " + std::to_string(line));
	//exit(EXIT_FAILURE);
}





}

