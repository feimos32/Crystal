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

#include "MainWindow.h"
#include "CrystalGUI/DebugTools/DebugStd.h"

#include "CrystalAlgrithm/Basic/Export_dll.cuh"


#include "CrystalAlgrithm/Basic/Transform.cuh"

#include "../../CrystalGUI/QtDataMapper/QtTF_1D_Trapezoidal.h"
#include "../../CrystalGUI/QtDataMapper/QtTF_2D_Trapezoidal_GF.h"

#include <QFile>

#define MainWindowDebug true

namespace CrystalGUI{


  
InitialMainWindow::InitialMainWindow(QWidget* parent)
    : QMainWindow(parent) {

    if (MainWindowDebug) {
        PrintValue_Std("InitialMainWindow::InitialMainWindow(...)");
    }

    // Appearance
    setWindowIcon(QIcon("Resources/Icons/sIcon.png"));
    
    QFile qssfile("Resources/qss/InitialMainWindow.qss");
    qssfile.open(QFile::ReadOnly);
    QString styleSheet = QString::fromLatin1(qssfile.readAll());
    this->setStyleSheet(styleSheet);

    setAttribute(Qt::WA_DeleteOnClose, true);

    setFixedSize(500, 483);

    setMenuBar(&menuBar);

    setupMenu();

    addToolBar(Qt::TopToolBarArea, &mainToolBar);

    setupTool();

    isDisplayMainWindowExist = false;
}

InitialMainWindow::~InitialMainWindow() {
    if (MainWindowDebug)
        PrintValue_Std("InitialMainWindow::~InitialMainWindow()");

    if (m_DisplayMainWindow && isDisplayMainWindowExist) {
        // Disconnect signals and slots
        disconnect(&RunExampleAction, SIGNAL(triggered()), this, SLOT(RunExample()));

        // delete DisplayMainWindow
        m_DisplayMainWindow->~DisplayMainWindow();
        
    }

    

    // Disconnect all signals and slots

}


void InitialMainWindow::setupMenu() {
    OpenSceneAction.setIcon(QIcon("Resources/Icons/OpenScene.png"));
    OpenSceneAction.setText(tr("Open Scene"));

    RunExampleAction.setIcon(QIcon("Resources/Icons/RunExample.png"));
    RunExampleAction.setText(tr("Run Example"));

    connect(&RunExampleAction, SIGNAL(triggered()), this, SLOT(RunExample()));

    menuBar.addMenu(&fileMenu);
    fileMenu.setTitle("File");

    fileMenu.addAction(&OpenSceneAction);
    fileMenu.addSeparator();
    fileMenu.addAction(&RunExampleAction);
}

void InitialMainWindow::setupTool() {
    mainToolBar.addAction(&OpenSceneAction);
    mainToolBar.addSeparator();
    mainToolBar.addAction(&RunExampleAction);
    mainToolBar.addSeparator();
}


void InitialMainWindow::closeEvent(QCloseEvent* e) {
    int ret = QMessageBox::question(this, "close", "Really close Crystal?");
    if (ret == QMessageBox::Yes) {
        e->accept();
    }
    else {
        e->ignore();
    }
}

void InitialMainWindow::RunExample() {
    if (m_DisplayMainWindow && isDisplayMainWindowExist) {
        
        return;
    }
    m_DisplayMainWindow = new DisplayMainWindow("Examples/Example1/Scene.xml", nullptr);
    m_DisplayMainWindow->show();
    // Connect signals and slots
    connect(m_DisplayMainWindow, SIGNAL(windowClosed()), this, SLOT(DisplayMainWindowClosed()));

    isDisplayMainWindowExist = true;
}

void InitialMainWindow::DisplayMainWindowClosed() {
    isDisplayMainWindowExist = false;
}



DisplayMainWindow::DisplayMainWindow(QString sceneFile, QWidget* parent) {
    setMinimumSize(350, 200);

    m_QtRenderThread = NULL;

    setWindowIcon(QIcon("Resources/Icons/sIcon.png"));
    QFile qssfile("Resources/qss/DisplayMainWindow.qss");
    qssfile.open(QFile::ReadOnly);
    QString styleSheet = QString::fromLatin1(qssfile.readAll());
    this->setStyleSheet(styleSheet);

    centralWidget = new QWidget;
    setCentralWidget(centralWidget);
    
    mainLayout = new QHBoxLayout;
    centralWidget->setLayout(mainLayout);

    // parse XML File
    sp.setFilePath(sceneFile);
    sp.readSceneXML();

    // Initialize TransferFunction DockWidget
    setQtTsFuncDock(sp);

    // Initialize QtVisualizer
    m_QtVisualizer = new QtVisualizer;
    m_QtVisualizer->Initialization(sp.m_ScenePreset.m_VisualizerPreset);

    // Initialize DisplayWidget
    displayWidget = new DisplayWidget;
    // thread return 0x1, maybe not a mistake
    mainLayout->addWidget(displayWidget);
    displayWidget->setFrameBuffer(m_QtVisualizer->m_FrameBuffer);
    displayWidget->initializeBuffer();


    // start rendering thread
    m_QtRenderThread = new QtRenderThread();
    m_QtRenderThread->setVisualizer(m_QtVisualizer->m_Visualizer);
    m_QtRenderThread->setFrameBuffer(m_QtVisualizer->m_FrameBuffer);

    m_QtRenderThread->renderBegin();
    m_QtRenderThread->start();
    connect(m_QtRenderThread, SIGNAL(generateNewFrame()), displayWidget, SLOT(displayNewFrame()));

    // Test
    CrystalAlgrithm::printCudaDevice();
    CrystalAlgrithm::SpectrumTest();



}


DisplayMainWindow::~DisplayMainWindow() {
    if (MainWindowDebug)
        PrintValue_Std("DisplayMainWindow::~DisplayMainWindow()");

    disconnect(m_QtRenderThread, SIGNAL(generateNewFrame()), displayWidget, SLOT(displayNewFrame()));

    if (m_QtRenderThread) {
        m_QtRenderThread->setStopFlag(true);

        // Kill the render thread
        m_QtRenderThread->quit();
        // Wait for thread to end
        m_QtRenderThread->wait();

        delete m_QtRenderThread;
        m_QtRenderThread = NULL;
    }

    if (m_QtVisualizer) {
        delete m_QtVisualizer;
    }

}

void DisplayMainWindow::closeEvent(QCloseEvent* e) {

    int ret = QMessageBox::question(this, "close", "Really close Display Window?");
    if (ret == QMessageBox::Yes) {
        emit windowClosed();
        e->accept();
        delete this;
    }
    else {
        e->ignore();
    }
}



void DisplayMainWindow::setQtTsFuncDock(ParserScene& sp) {

    if ("TF_1D_Trapezoidal" == sp.getTsFuncType()) {
        m_QtTsFuncDock = new QtTF_1D_Trapezoidal();
    }
    else if ("TF_2D_Trapezoidal_GF" == sp.getTsFuncType()) {
        m_QtTsFuncDock = new QtTF_2D_Trapezoidal_GF();
    }
    else {
        PrintError("No matching transfer function name");
        return;
    }
    m_QtTsFuncDock->setTsFuncDirPath(sp.getFileDir().toStdString());
    m_QtTsFuncDock->Initialize(
        sp.m_ScenePreset.m_DataMapperPreset,
        sp.m_ScenePreset.m_VisualizerPreset);

    addDockWidget(Qt::LeftDockWidgetArea, m_QtTsFuncDock);

    //tabifyDockWidget(mytfDockWidget, myLightSetDockWidget);
    m_QtTsFuncDock->raise();

}







}


