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

#include "QtVisualizer.h"

#include "CrystalGUI/DebugTools/DebugStd.h"
#define QtVisualizerDebug true

namespace CrystalGUI {

QtVisualizer::QtVisualizer(QObject* parent){
	m_Visualizer = nullptr;
	m_FrameBuffer = nullptr;

}

QtVisualizer::~QtVisualizer() {

	if (QtVisualizerDebug) {
		PrintValue_Std("QtVisualizer::~QtVisualizer()");
	}

	if (!m_Visualizer) m_Visualizer.release();
	if (!m_FrameBuffer) m_FrameBuffer.release();
}

void QtVisualizer::Initialization(const CrystalAlgrithm::PresetVisualizer& visualizerPreset) 
{
	if (QtVisualizerDebug) {
		PrintValue_Std("QtVisualizer::Initialization(...)");
	}

	visualizerPreset.VisualizerType;

	visualizerPreset.width;
	visualizerPreset.height;



}




}












