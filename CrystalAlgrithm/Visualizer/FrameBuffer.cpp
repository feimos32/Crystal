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

#include "FrameBuffer.h"
#include "CrystalAlgrithm/Basic/Common.cuh"

#define FrameBuffer_Debug true

namespace CrystalAlgrithm {

FrameBuffer::FrameBuffer() {

	if (FrameBuffer_Debug) {
		PrintValue_Std("FrameBuffer::FrameBuffer()");
	}

	width = height = 0;
	allRenderCount = 0;
	renderCount_stableCamera = 0;
	renderCount_stableTsFunc = 0;
	renderCount_stableLight = 0;

	displayBufferU = nullptr;
	GPU_displayBufferU = nullptr;
	GPU_displayBufferF = nullptr;
}

FrameBuffer::~FrameBuffer()
{
	if (FrameBuffer_Debug) {
		PrintValue_Std("FrameBuffer::~FrameBuffer()");
	}
	ReleaseAll();
}

void FrameBuffer::ReleaseAll() {

	// CPU Buffers
	if (displayBufferU) delete displayBufferU;
	displayBufferU = nullptr;

	// GPU Buffers
	if (GPU_displayBufferU) Get_CUDA_ERROR(cudaFree(displayBufferU));
	GPU_displayBufferU = nullptr;
	if (GPU_displayBufferF) Get_CUDA_ERROR(cudaFree(GPU_displayBufferF));
	GPU_displayBufferF = nullptr;


	width = height = 0;

	allRenderCount = 0;
	renderCount_stableCamera = 0;
	renderCount_stableTsFunc = 0;
	renderCount_stableLight = 0;
}

void FrameBuffer::ResetAll(size_t w, size_t h) {

	if (FrameBuffer_Debug) {
		PrintValue_Std("FrameBuffer::ResetAll(...)");
	}

	ReleaseAll();

	width = w;
	height = h;

	// CPU Buffers
	displayBufferU = new uchar3[width * height];

	// GPU Buffers
	Get_CUDA_ERROR(cudaMalloc(&GPU_displayBufferU, width * height * sizeof(uchar3)));
	Get_CUDA_ERROR(cudaMalloc(&GPU_displayBufferF, width * height * sizeof(float4)));

}






}





