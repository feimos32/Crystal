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

#ifndef __TF_1D_Trapezoidal_h__
#define __TF_1D_Trapezoidal_h__

#include "TransferFunction.h"
#include "CrystalAlgrithm/Basic/Geometry.cuh"
#include "CrystalAlgrithm/Basic/Spectrum.cuh"

#include <vector>

namespace CrystalAlgrithm {

struct TF_1D_Trapezoidal_Node {
	float normalizedIntensity;
	Spectrum3 emit;
	Spectrum3 diffuse;
	Spectrum3 specular;
	float opacity;
	float roughness;
	float metallic;
};

class TF_1D_Trapezoidal : public TransferFunction {
public:

	TF_1D_Trapezoidal() {
		clear();
	}

	void clear() { 
		hasEmit = hasDiffuse = hasSpecular =
			hasOpacity = hasRoughness = hasMetallic = false;
		nodes.clear();
	}

	std::vector<TF_1D_Trapezoidal_Node> nodes;

	bool hasEmit, hasDiffuse, hasSpecular, 
		hasOpacity, hasRoughness, hasMetallic;


};


}



#endif



