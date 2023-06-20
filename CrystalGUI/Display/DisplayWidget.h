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

#ifndef __DisplayWidget_h__
#define __DisplayWidget_h__

#include "CrystalGUI/Utility/Common.h"

#include "vtkAutoInit.h"
VTK_MODULE_INIT(vtkRenderingOpenGL2);
VTK_MODULE_INIT(vtkInteractionStyle);
VTK_MODULE_INIT(vtkRenderingVolumeOpenGL2);
VTK_MODULE_INIT(vtkRenderingFreeType);


#include "CrystalGUI/QtVisInteractor/QtVisInteractor.h"
#include "CrystalGUI/QtVisualizer/QtVisualizer.h"

#include <QVTKOpenGLNativewidget.h>
#include <vtkSmartPointer.h>

class vtkRenderer;
class vtkGenericOpenGLRenderWindow;
class vtkImageActor;
class vtkImageImport;

class vtkCallbackCommand;
class vtkInteractorStyleImage;
class vtkInteractorStyleUser;


namespace CrystalGUI {

class DisplayWidget : public QVTKOpenGLNativeWidget {
    Q_OBJECT
public:
    DisplayWidget(QWidget* parent = nullptr);
    ~DisplayWidget();

private slots:
    void displayNewFrame();

public:
    void setFrameBuffer(std::shared_ptr<CrystalAlgrithm::FrameBuffer> framebuffer) {
        m_FrameBuffer = framebuffer;
    }
    std::shared_ptr<CrystalAlgrithm::FrameBuffer> m_FrameBuffer;
    void initializeBuffer();
    void resizeBuffer();

private: 
    

    vtkSmartPointer<vtkRenderer>				    m_SceneRenderer;
    vtkSmartPointer<vtkGenericOpenGLRenderWindow>   m_RenderWindow;
    vtkSmartPointer<vtkImageActor>				m_ImageActor;
    vtkSmartPointer<vtkImageImport>				m_ImageImport;

    vtkSmartPointer<vtkCallbackCommand>			m_KeyPressCallback;
    vtkSmartPointer<vtkCallbackCommand>			m_KeyReleaseCallback;
    vtkSmartPointer<vtkInteractorStyleImage>	m_InteractorStyleImage;
    vtkSmartPointer<vtkInteractorStyleUser>	    m_InteractorStyleUser;

};




}


#endif


