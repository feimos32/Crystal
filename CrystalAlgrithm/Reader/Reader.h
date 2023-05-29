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

#ifndef __Reader_h__
#define __Reader_h__

#include <string>
#include <iostream>

namespace CrystalAlgrithm {


inline std::string obtainDirFromFilePath(std::string filepath){
	size_t last_slash_idx = filepath.rfind('\\');
	if (std::string::npos != last_slash_idx) {
		std::string fileDir = filepath.substr(0, last_slash_idx);
		return fileDir;
	}
	last_slash_idx = filepath.rfind('/');
	if (std::string::npos != last_slash_idx) {
		std::string fileDir = filepath.substr(0, last_slash_idx);
		return fileDir;
	}
	return "";
}

class Reader {
public:

	Reader() {
		filePath = "";
		fileDir = "";
	}

	virtual bool readFromDirectory(std::string dirpath) {
		std::cout << "This class does not support reading data from the directory." << std::endl;
		return false;
	}

	virtual bool readFromFilePath(std::string filepath) {
		std::cout << "This class does not support reading data based on file name." << std::endl;
		return false;
	}

	std::string filePath;
	std::string fileDir;
};


}



#endif





