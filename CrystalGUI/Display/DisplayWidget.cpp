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

#include "DisplayWidget.h"
#include "CrystalGUI/DebugTools/DebugStd.h"

#define DisplayWidget_Debug true


#include <vtkRenderWindow.h>
#include <vtkGenericOpenGLRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkImageImport.h>
#include <vtkImageActor.h>
#include <vtkCamera.h>

#include <vtkInteractorStyleImage.h>
#include <vtkCallbackCommand.h>
#include <vtkInteractorStyleUser.h>

namespace CrystalGUI {

DisplayWidget::DisplayWidget(QWidget* parent) { 
    if (DisplayWidget_Debug) {
        PrintValue_Std("DisplayWidget::DisplayWidget(...)");
    }

    m_FrameBuffer = NULL;

    m_SceneRenderer = vtkRenderer::New();

    m_SceneRenderer->SetBackground(0.25, 0.25, 0.25);
    m_SceneRenderer->SetBackground2(0.25, 0.25, 0.25);
    m_SceneRenderer->SetGradientBackground(true);
    
    m_RenderWindow = vtkSmartPointer<vtkGenericOpenGLRenderWindow>::New();
    m_RenderWindow->AddRenderer(m_SceneRenderer);
    setRenderWindow(m_RenderWindow);

    m_ImageImport = vtkImageImport::New();
    m_ImageActor = vtkImageActor::New();

}

DisplayWidget::~DisplayWidget() {
    if (DisplayWidget_Debug) {
        PrintValue_Std("DisplayWidget::~DisplayWidget()");
    }

}

void DisplayWidget::displayNewFrame() {
    if (!m_FrameBuffer) {
        PrintError("Pointer m_FrameBuffer is nullptr when display new frame");
        return;
    }

    //if (DisplayWidget_Debug)
    //    PrintValue_Std("DisplayWidget::displayNewFrame()");

    m_ImageImport->SetImportVoidPointer(NULL);
    m_ImageImport->SetImportVoidPointer(m_FrameBuffer->get_displayBufferU());

    m_ImageImport->SetDataOrigin(
        -0.5 * (double)m_FrameBuffer->width, -0.5 * (double)m_FrameBuffer->height, 0);
    m_ImageImport->SetWholeExtent(0, m_FrameBuffer->width - 1, 0, m_FrameBuffer->height - 1, 0, 0);
    m_ImageImport->UpdateWholeExtent();
    m_ImageImport->SetDataExtentToWholeExtent();
    m_ImageImport->Update();

    m_ImageActor->SetInputData(m_ImageImport->GetOutput());
    m_RenderWindow->GetInteractor()->Render();
}

void DisplayWidget::initializeBuffer() {
    if (!m_FrameBuffer) {
        PrintError("Pointer m_FrameBuffer is nullptr when initialize DisplayWidget");
        return;
    }

    m_SceneRenderer->GetActiveCamera()->SetParallelScale(100.0f);

    m_ImageImport->SetDataSpacing(1, 1, 1);
    m_ImageImport->SetDataOrigin(
        -0.5 * (double)m_FrameBuffer->width, -0.5 * (double)m_FrameBuffer->height, 0);

    void* m_pPixels = (m_FrameBuffer->get_displayBufferU());

    m_ImageImport->SetImportVoidPointer(m_pPixels, 1);
    m_ImageImport->SetWholeExtent(0, m_FrameBuffer->width - 1, 0, m_FrameBuffer->height - 1, 0, 0);
    m_ImageImport->SetDataExtentToWholeExtent();
    m_ImageImport->SetDataScalarTypeToUnsignedChar();
    m_ImageImport->SetNumberOfScalarComponents(3);
    m_ImageImport->Update();

    m_ImageActor->SetInterpolate(1);
    m_ImageActor->SetInputData(m_ImageImport->GetOutput());
    m_ImageActor->SetScale(1, -1, -1);
    m_ImageActor->VisibilityOn();

    // Add the image actor
    m_SceneRenderer->AddActor(m_ImageActor);
    m_RenderWindow->GetInteractor()->Render();
}

void DisplayWidget::resizeBuffer() {

    if (!m_FrameBuffer) {
        PrintError("Pointer m_FrameBuffer is nullptr when resize the Buffer");
        return;
    }




}

}



