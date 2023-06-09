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

#ifndef __Visualizer_h__
#define __Visualizer_h__

#include "CrystalAlgrithm/Basic/Export_dll.cuh"
#include "CrystalAlgrithm/Basic/Common.cuh"

#include "CrystalAlgrithm/Visualizer/FrameBuffer.h"

namespace CrystalAlgrithm {

class EXPORT_DLL Visualizer {

public:
	Visualizer();
	~Visualizer();

	virtual void visualize();
	bool resetFrameBuffer(FrameBuffer* fb);

public:
	FrameBuffer* framebuffer_GPU;
	int width, height;

};




}


#endif



