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

#ifndef __ExposureRender_h__
#define __ExposureRender_h__

#include "Visualizer.h"

#include "CrystalAlgrithm/Basic/Export_dll.cuh"
#include "CrystalAlgrithm/Basic/Common.cuh"

namespace CrystalAlgrithm {

class EXPORT_DLL ExposureRender: public Visualizer {

public:
	ExposureRender();
	~ExposureRender();

	virtual void visualize(FrameBuffer* framebuffer);

};




}


#endif



