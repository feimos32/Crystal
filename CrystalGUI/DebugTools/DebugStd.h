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

#ifndef __DebugStd_h__
#define __DebugStd_h__

#include "CrystalGUI/Utility/Common.h"

#include <iostream>
#include <string>
#include <QString>

namespace CrystalGUI {

inline void PrintError_Std(std::string err, const char* file, int line) {
	std::cout << "[Error]: " <<  std::string(err) + " in " + std::string(file) +
		" at line " + std::to_string(line) << std::endl;
}

#define PrintError( err ) (PrintError_Std( err, __FILE__, __LINE__ ))

inline void PrintValue_Std(std::string info, float s) {
	std::cout <<"[Debug]: " <<  info + ": [" + std::to_string(s) + "]" << std::endl;
}
inline void PrintValue_Std(std::string info, size_t s) {
	std::cout << "[Debug]: " << info + ": [" + std::to_string(s) + "]" << std::endl;
}
inline void PrintValue_Std(std::string info, int s) {
	std::cout << "[Debug]: " << info + ": [" + std::to_string(s) + "]" << std::endl;
}
inline void PrintValue_Std(std::string info, QString s) {
	std::cout << "[Debug]: " << info + ": [" + s.toStdString() + "]" << std::endl;
}

#define PrintValue(info, val) PrintValue_Std(info, val);



inline void PrintValue_Std(float s) {
	std::cout << "[Debug]: " << "[" + std::to_string(s) + "]" << std::endl;
}
inline void PrintValue_Std(int s) {
	std::cout << "[Debug]: " << "[" + std::to_string(s) + "]" << std::endl;
}
inline void PrintValue_Std(QString s) {
	std::cout << "[Debug]: " << "[" + s.toStdString() + "]" << std::endl;
}




}






#endif



