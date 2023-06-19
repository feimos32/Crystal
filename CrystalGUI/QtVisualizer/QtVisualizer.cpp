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

#include "CrystalAlgrithm/Visualizer/ExposureRender.h"

namespace CrystalGUI {

QtVisualizer::QtVisualizer(QObject* parent){
	if (QtVisualizerDebug) {
		PrintValue_Std("QtVisualizer::QtVisualizer()");
	}

	m_Visualizer = nullptr;
	m_FrameBuffer = nullptr;

}

QtVisualizer::~QtVisualizer() {

	if (QtVisualizerDebug) {
		PrintValue_Std("QtVisualizer::~QtVisualizer()");
	}
}

void QtVisualizer::Initialization(const CrystalAlgrithm::PresetVisualizer& visualizerPreset) 
{
	if (QtVisualizerDebug) {
		PrintValue_Std("QtVisualizer::Initialization(...)");
	}

	if ("ExposureRender" == visualizerPreset.VisualizerType) {
		m_Visualizer = 
			std::shared_ptr<CrystalAlgrithm::Visualizer>(new CrystalAlgrithm::ExposureRender());
	}
	else {
		PrintError("Unknown Visualizer name");
		return;
	}

	m_FrameBuffer = std::shared_ptr<CrystalAlgrithm::FrameBuffer>(new CrystalAlgrithm::FrameBuffer());
	m_FrameBuffer->ResetAll(visualizerPreset.width, visualizerPreset.height);

}







}












