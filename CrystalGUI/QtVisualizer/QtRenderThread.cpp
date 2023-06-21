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

#include <QElapsedTimer>

#include "QtRenderThread.h"

#include "CrystalGUI/DebugTools/DebugStd.h"

#define QtRenderThread_Debug true

namespace CrystalGUI {


QtRenderThread::QtRenderThread(QObject* pParent) {

	if (QtRenderThread_Debug) {
		PrintValue_Std("QtRenderThread::QtRenderThread(...)");
	}

	stopFlag = false;

	m_Visualizer = nullptr;
	m_FrameBuffer = nullptr;
}

QtRenderThread::~QtRenderThread() {
	if (QtRenderThread_Debug) {
		PrintValue_Std("QtRenderThread::~QtRenderThread()");
	}
}

void QtRenderThread::renderBegin() {
	if (m_Visualizer && m_FrameBuffer)
		m_Visualizer->resetFrameBuffer(m_FrameBuffer.get());
	else {
		PrintError("m_Visualizer or m_FrameBuffer if nullptr");
	}


}

void QtRenderThread::run() {

	while (!stopFlag) {
		QElapsedTimer t;
		t.start();

		visualize();


		emit generateNewFrame();
		while (t.elapsed() < 10);
	}
}

void QtRenderThread::visualize() {

	if (m_Visualizer && m_FrameBuffer) {
		m_FrameBuffer->FrameCountPlus();

		m_Visualizer->resetFrameBuffer(m_FrameBuffer.get());
		m_Visualizer->visualize();
	}
		
	else {
		PrintError("m_Visualizer or m_FrameBuffer if nullptr");
		return;
	}

	m_FrameBuffer->obtainOutputFromGPU();

}




}



