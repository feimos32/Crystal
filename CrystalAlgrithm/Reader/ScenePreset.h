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
along with this program. If not, see <https://www.gnu.org/licenses/>.

Github site: <https://github.com/feimos32/Crystal>
*/

#include "CrystalAlgrithm/VisInteractor/CameraPreset.h"
#include "CrystalAlgrithm/Light/LightPreset.h"
#include "CrystalAlgrithm/DataOrganiz/DataPreset.h"
#include "CrystalAlgrithm/DataMapper/DataMapperPreset.h"

#ifndef __ScenePreset_h__
#define __ScenePreset_h__

#include <string>

namespace CrystalAlgrithm{

class ScenePreset {
public:
	ScenePreset():
		m_CameraPreset(),
		m_LightPreset(),
		m_DataPreset(),
		m_DataMapperPreset()
	{
		SceneFilePath = "";
		SceneFileName = "";
		SceneFileDir = "";
	}

	CameraPreset m_CameraPreset;
	LightPreset m_LightPreset;
	DataPreset m_DataPreset;
	DataMapperPreset m_DataMapperPreset;

	std::string SceneFilePath;
	std::string SceneFileName;
	std::string SceneFileDir;
};

}



#endif




