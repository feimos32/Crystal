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

#include "CrystalAlgrithm/Basic/Export_dll.cuh"

#include <vector>

namespace CrystalAlgrithm {

struct TF_1D_Trapezoidal_Node {
	float normalizedIntensity;
	float opacity;
	float roughness;
	float metallic;
	// cannot defined as 'emit', because 'emit' is a basic function in Qt
	float3 emission;
	float3 diffuse;
	float3 specular;
};

class EXPORT_DLL TF_1D_Trapezoidal : public TransferFunction {
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
	float DensityScale;

	bool hasEmit, hasDiffuse, hasSpecular, 
		hasOpacity, hasRoughness, hasMetallic;


};


}



#endif



