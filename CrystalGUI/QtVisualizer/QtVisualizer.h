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

#ifndef __QtVisualizer_h__
#define __QtVisualizer_h__

#include "CrystalGUI/Utility/Common.h"

#include <QObject>

#include "CrystalAlgrithm/Visualizer/FrameBuffer.h"

#include "CrystalAlgrithm/Visualizer/Visualizer.h"
#include "CrystalAlgrithm/Visualizer/ExposureRender.h"

#include "CrystalAlgrithm/Scene/PresetScene.h"


#include <memory>

namespace CrystalGUI {


class QtVisualizer : public QObject {
	Q_OBJECT
public:
	QtVisualizer(QObject* parent = Q_NULLPTR);
	~QtVisualizer();

	void Initialization(const CrystalAlgrithm::PresetVisualizer& visualizerPreset);

	std::unique_ptr<CrystalAlgrithm::Visualizer> m_Visualizer;
	std::unique_ptr<CrystalAlgrithm::FrameBuffer> m_FrameBuffer;



};





}

#endif



