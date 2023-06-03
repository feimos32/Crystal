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

#ifndef __PresetCamera_h__
#define __PresetCamera_h__

#include <string>
#include <iostream>

namespace CrystalAlgrithm {

class PresetCamera {
public:
	PresetCamera() {
		CameraType = "";
	}
	std::string CameraType;

	void PrintCameraPreset() {
		std::cout << "PrintCameraPreset: " << std::endl;
		std::cout << "  CameraType: " << CameraType << std::endl;
	}
};

}

#endif


