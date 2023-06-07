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

#ifndef __TF_2D_Trapezoidal_GF_h__
#define __TF_2D_Trapezoidal_GF_h__

#include "TransferFunction.h"
#include "CrystalAlgrithm/Basic/Geometry.cuh"
#include "CrystalAlgrithm/Basic/Spectrum.cuh"

#include <vector>

namespace CrystalAlgrithm {

struct TF_2D_Trapezoidal_GF_Node_1 {
	float normalizedIntensity;
	Spectrum3 emit;
	Spectrum3 diffuse;
	Spectrum3 specular;
	float opacity;
	float roughness;
	float metallic;
};

struct TF_2D_Trapezoidal_GF_Node_2 {
	float normalizedIntensity;
	float opacity;
};

// TF_2D_Trapezoidal Gradient factor
class TF_2D_Trapezoidal_GF : public TransferFunction {
public:

	std::vector<TF_2D_Trapezoidal_GF_Node_1> nodes1;
	std::vector<TF_2D_Trapezoidal_GF_Node_2> nodes2;


};




}


#endif



