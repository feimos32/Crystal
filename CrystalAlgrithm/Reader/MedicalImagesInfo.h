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

#ifndef __MedicalImagesInfo_h__
#define __MedicalImagesInfo_h__

#include "CrystalAlgrithm/Basic/Common.cuh"

namespace CrystalAlgrithm {


class MedicalImagesInfo {
public:
	MedicalImagesInfo() {
		data = nullptr;
		reset();
	} 
	
	void reset() {
		Dim = 0;
		X_Dim = Y_Dim = Z_Dim = 0;
		X_Spacing = Y_Spacing = Z_Spacing = 0.0;
		Slope = Intercept = 0.0;
		dataRange[0] = dataRange[1] = 0;
		gradientRange[0] = gradientRange[1] = 0;

		if (data) {
			delete[] data;
			data = nullptr;
		}
	}
	~MedicalImagesInfo() {
		reset();
	}

	int Dim;
	size_t X_Dim, Y_Dim, Z_Dim;
	Float X_Spacing, Y_Spacing, Z_Spacing;
	Float Slope, Intercept;
	short dataRange[2];
	short gradientRange[2];

	void* data;
};



}

#endif




