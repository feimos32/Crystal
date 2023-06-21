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

#include "Visualizer.h"

#define Visualizer_Debug true

namespace CrystalAlgrithm {

Visualizer::Visualizer() {
	
	if (Visualizer_Debug) {
		PrintValue_Std("Visualizer::Visualizer()");
	}

	framebuffer_GPU = nullptr;
	width = height = 0;
}
	
Visualizer::~Visualizer() {

	if (Visualizer_Debug) {
		PrintValue_Std("Visualizer::~Visualizer()");
	}
	if (framebuffer_GPU) {
		bool flag = Get_CUDA_ERROR(cudaFree(framebuffer_GPU));
		if (!flag) PrintError("Find error when free framebuffer_GPU");
	}
}

void Visualizer::visualize() {
	PrintError("Visualizer::visualize(...) cannot be execuated");
}

bool Visualizer::resetFrameBuffer(FrameBuffer* fb) {
	bool flag;
	//if (framebuffer_GPU) {
	//	flag = Get_CUDA_ERROR(cudaFree(framebuffer_GPU));
	//	if (!flag) return false;
	//}
	
	if (!framebuffer_GPU) {
		flag = Get_CUDA_ERROR(cudaMalloc(&framebuffer_GPU, sizeof(FrameBuffer)));
		if (!flag) return false;
	}

	flag = Get_CUDA_ERROR(cudaMemcpy(framebuffer_GPU, fb, sizeof(FrameBuffer), cudaMemcpyHostToDevice));
	if (!flag) return false;

	width = fb->width;
	height = fb->height;

	return true;
}





}










