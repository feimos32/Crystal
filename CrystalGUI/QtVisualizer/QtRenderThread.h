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

#ifndef __QtRenderThread_h__
#define __QtRenderThread_h__

#include "CrystalGUI/Utility/Common.h"

#include <QThread>

#include "QtVisualizer.h"

namespace CrystalGUI {


class QtRenderThread : public QThread {
	Q_OBJECT
public:
	QtRenderThread(QObject* pParent = NULL);
	~QtRenderThread();

	void run();
	void renderBegin();
	void setStopFlag(bool s) {
		stopFlag = s;
	}

	std::shared_ptr<CrystalAlgrithm::Visualizer> m_Visualizer;
	void setVisualizer(std::shared_ptr<CrystalAlgrithm::Visualizer> vis) {
		m_Visualizer = vis;
	}
	std::shared_ptr<CrystalAlgrithm::FrameBuffer> m_FrameBuffer;
	void setFrameBuffer(std::shared_ptr<CrystalAlgrithm::FrameBuffer> framebuffer) {
		m_FrameBuffer = framebuffer;
	}
	void visualize();

private:
	bool stopFlag;
signals:
	void generateNewFrame();

};


}



#endif



