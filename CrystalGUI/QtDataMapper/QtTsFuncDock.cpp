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

#include "CrystalGUI/DebugTools/DebugStd.h"
#include "QtTsFuncDock.h"

#define QtTsFuncDockDebug true

namespace CrystalGUI {

QtTsFuncDock::QtTsFuncDock(QWidget* pParent) {

	if (QtTsFuncDockDebug) {
		PrintValue_Std("QtTsFuncDock::QtTsFuncDock(...)");
	}

}

QtTsFuncDock::~QtTsFuncDock() {

	if (QtTsFuncDockDebug) {
		PrintValue_Std("QtTsFuncDock::~QtTsFuncDock()");
	}

}

void QtTsFuncDock::Initialize(
	CrystalAlgrithm::PresetDataMapper& m_DataMapperPreset,
	CrystalAlgrithm::PresetVisualizer& m_VisualizerPreset) {

	if (QtTsFuncDockDebug) {
		PrintValue_Std("QtTsFuncDock::Initialize(...)");
	}

}



}




