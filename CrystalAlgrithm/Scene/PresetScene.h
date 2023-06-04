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


#ifndef __PresetScene_h__
#define __PresetScene_h__

#include <string>
#include <iostream>

namespace CrystalAlgrithm{

class PresetData {
public:
	PresetData() {
		DataFileType = "";
		DataType = "";
		DataFilePath = "";
	}
	std::string DataFileType;
	std::string DataType;
	std::string DataFilePath;

	void PrintDataPreset() {
		std::cout << "PrintDataPreset: " << std::endl;
		std::cout << "  DataFilePath: " << DataFilePath << std::endl;
		std::cout << "  DataFileType: " << DataFileType << std::endl;
		std::cout << "  DataType: " << DataType << std::endl;
	}
};

class PresetDataMapper {
public:
	PresetDataMapper() {
		TsFuncType = "";
		TsFuncFileName = "";
	}
	std::string TsFuncType;
	std::string TsFuncFileName;

	void PrintDataMapperPreset() {
		std::cout << "PrintDataMapperPreset: " << std::endl;
		std::cout << "  TsFuncType: " << TsFuncType << std::endl;
		std::cout << "  TsFuncFileName: " << TsFuncFileName << std::endl;
	}
};

class PresetSceneGeometry {
public:
	PresetSceneGeometry() {
		SceneGeometryFile = "";
	}

	std::string SceneGeometryFile;

	void PrintLightPreset() {
		std::cout << "PrintSceneGeometryPreset: " << std::endl;
		std::cout << "  SceneGeometryFile: " << SceneGeometryFile << std::endl;
	}
};

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

class PresetScene {
public:
	PresetScene():
		m_CameraPreset(),
		m_SceneGeometryPreset(),
		m_DataPreset(),
		m_DataMapperPreset()
	{
		SceneFilePath = "";
		SceneFileName = "";
		SceneFileDir = "";
	}

	PresetCamera m_CameraPreset;
	PresetSceneGeometry m_SceneGeometryPreset;
	PresetData m_DataPreset;
	PresetDataMapper m_DataMapperPreset;

	std::string SceneFilePath;
	std::string SceneFileName;
	std::string SceneFileDir;
};

}



#endif




