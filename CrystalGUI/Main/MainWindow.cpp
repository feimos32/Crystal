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

#include <QFile>

#include "CrystalAlgrithm/Basic/Export_dll.cuh"
#include "CrystalGUI/QtReader/ParserScene.h"

#include "CrystalAlgrithm/Basic/Transform.cuh"

#define MainWindowDebug true

namespace CrystalGUI{

  
InitialMainWindow::InitialMainWindow(QWidget* parent)
    : QMainWindow(parent) {

    isDisplayMainWindowExist = false;
    m_DisplayMainWindow = nullptr;

    // Appearance
    setWindowIcon(QIcon("Resources/Icons/sIcon.png"));
    
    QFile qssfile("Resources/qss/InitialMainWindow.qss");
    qssfile.open(QFile::ReadOnly);
    QString styleSheet = QString::fromLatin1(qssfile.readAll());
    this->setStyleSheet(styleSheet);

    setFixedSize(500, 483);

    setMenuBar(&menuBar);

    setupMenu();

    addToolBar(Qt::TopToolBarArea, &mainToolBar);

    setupTool();

}

InitialMainWindow::~InitialMainWindow() {

    if (m_DisplayMainWindow && isDisplayMainWindowExist) {
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
    int ret = QMessageBox::question(this, "question", "Really close Crystal?");
    if (ret == QMessageBox::Yes) {
        e->accept();
    }
    else {
        e->ignore();
    }
}

void InitialMainWindow::RunExample() {
    m_DisplayMainWindow = new DisplayMainWindow("Examples/Example1/Scene.xml");
    isDisplayMainWindowExist = true;

    connect(m_DisplayMainWindow, SIGNAL(windowClosed()), this, SLOT(DisplayMainWindowClosed()));

    m_DisplayMainWindow->show();


}

void InitialMainWindow::DisplayMainWindowClosed() {
    isDisplayMainWindowExist = false;
}




DisplayMainWindow::DisplayMainWindow(QString sceneFile, QWidget* parent) {
    setMinimumSize(350, 200);

    setCentralWidget(&centralWidget);

    CrystalAlgrithm::printCudaDevice();
    CrystalAlgrithm::SpectrumTest();

    CrystalGUI::ParserScene sp;
    sp.setFilePath(sceneFile);
    sp.readSceneXML();

}

DisplayMainWindow::~DisplayMainWindow() {
    if (MainWindowDebug)
        PrintValue_Std("DisplayMainWindow::~DisplayMainWindow()");
    

}

void DisplayMainWindow::closeEvent(QCloseEvent* e) {
    int ret = QMessageBox::question(this, "question", "Really close Crystal?");
    if (ret == QMessageBox::Yes) {
        e->accept();
        emit windowClosed();
        delete this;
    }
    else {
        e->ignore();
    }
}








}


