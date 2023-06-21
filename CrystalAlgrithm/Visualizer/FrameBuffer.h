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

#ifndef __FrameBuffer_h__
#define __FrameBuffer_h__

#include "CrystalAlgrithm/Basic/Export_dll.cuh"
#include "CrystalAlgrithm/Basic/Common.cuh"

namespace CrystalAlgrithm {

class EXPORT_DLL FrameBuffer {

public:

	FrameBuffer();
	~FrameBuffer();

	void ReleaseAll();
	void ResetAll(size_t w, size_t h);

	size_t width, height;

	size_t allRenderCount;
	size_t renderCount_stableCamera;
	size_t renderCount_stableTfFunc;
	size_t renderCount_stableLight;
	void FrameCountPlus() {
		allRenderCount++;
		renderCount_stableCamera++;
		renderCount_stableTfFunc++;
		renderCount_stableLight++;
	}

	// Buffers in CPU
	uchar3* displayBufferU;
	uchar3* get_displayBufferU() {
		return displayBufferU;
	}
	bool obtainOutputFromGPU() {
		if (width == 0 || height == 0) {
			PrintError("(width == 0 || height == 0) in FrameBuffer");
			return false;
		}
		bool flag = Get_CUDA_ERROR(cudaMemcpy(displayBufferU, GPU_displayBufferU, sizeof(uchar3) * width * height, cudaMemcpyDeviceToHost));
		return flag;
	}

	// Buffers in GPU
	uchar3* GPU_displayBufferU;
	uchar3* get_GPU_displayBufferU() {
		return GPU_displayBufferU;
	}
	float4* GPU_displayBufferF;
	float4* get_GPU_displayBufferF() {
		return GPU_displayBufferF;
	}



};

}

#endif



