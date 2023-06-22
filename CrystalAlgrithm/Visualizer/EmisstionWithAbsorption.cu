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

#include "EmisstionWithAbsorption.h"
#include "CrystalAlgrithm/Basic/Common.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>

#define ExposureRender_Debug true

namespace CrystalAlgrithm {



__global__ void EmisstionWithAbsorption_visualize_core(FrameBuffer* fb_GPU);
void EmisstionWithAbsorption::visualize() {
	//if (ExposureRender_Debug)
	//	PrintValue_Std("ExposureRender::visualize(...)");

	if (framebuffer_GPU == nullptr) {
		PrintError("framebuffer_GPU == nullptr");
		return;
	}

	dim3    blocks((width + 15) / 16, (height + 15) / 16);
	dim3    threads(16, 16);
	EmisstionWithAbsorption_visualize_core << <blocks, threads >> > (framebuffer_GPU);

	/*framebuffer->FrameCountPlus();
	size_t Frame = framebuffer->allRenderCount;
	uchar3* buffer = framebuffer->get_displayBufferU();

	for (int i = 0; i < framebuffer->width; i++) {
		for (int j = 0; j < framebuffer->height; j++) {
			
			rsize_t offset = (i + j * framebuffer->width);

			unsigned char color = Frame % 250;

			buffer[offset] = make_uchar3(color, color, color);
			
		}
	}*/

}


__global__ void EmisstionWithAbsorption_visualize_core(FrameBuffer* fb_GPU)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * fb_GPU->width;

	unsigned char color = fb_GPU->allRenderCount % 250;
	fb_GPU->GPU_displayBufferU[offset] = make_uchar3(0, color, color);
}




}









