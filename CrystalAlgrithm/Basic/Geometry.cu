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

/*
The following code is partially rewritten from: 
PBRT v3: <https://pbrt.org/>
*/

#include "Geometry.cuh"

#include <cuda.h>
#include <cuda_runtime.h>

// It seems that math_functions.h may not be included into.cpp file.
#include <math_functions.h>

#include <cmath>
#include <algorithm>


// Vector3f/Point3f/Normal3f
namespace CrystalAlgrithm {

HOST_Ctl inline Point3f Floor(const Point3f& p) {
	return Point3f(std::floor(p.x), std::floor(p.y), std::floor(p.z));
}

HOST_Ctl inline Point3f Ceil(const Point3f& p) {
	return Point3f(std::ceil(p.x), std::ceil(p.y), std::ceil(p.z));
}

}

// Vector2f/Point2f/Normal2f
namespace CrystalAlgrithm {

}








