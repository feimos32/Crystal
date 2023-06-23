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

#ifndef __MainWindow_h__
#define __MainWindow_h__

#include "CrystalGUI/Utility/Common.h"

#include <QtWidgets>

#include <QMainWindow>
#include <QWidget>
#include <QCloseEvent>

#include <QHBoxLayout>

#include "CrystalGUI/Display/DisplayWidget.h"
#include "CrystalGUI/QtDataMapper/QtTsFuncDock.h"
#include "CrystalGUI/QtVisualizer/QtVisualizer.h"
#include "CrystalGUI/QtVisualizer/QtRenderThread.h"
#include "CrystalGUI/QtVisInteractor/QtVisInteractor.h"

#include "CrystalGUI/QtReader/ParserScene.h"

namespace CrystalGUI{

class DisplayMainWindow : public QMainWindow
{
    Q_OBJECT

public:

    DisplayMainWindow(QString sceneFile, QWidget* parent = 0);
    ~DisplayMainWindow();

    void setQtTsFuncDock(ParserScene& sp);

signals:
    void windowClosed();

protected:

    QHBoxLayout * mainLayout;

    DisplayWidget * displayWidget;

    QtTsFuncDock* m_QtTsFuncDock;

    QWidget* centralWidget;

    void closeEvent(QCloseEvent* e);

    ParserScene sp;

    QtVisualizer* m_QtVisualizer;
    QtRenderThread* m_QtRenderThread;



};

class InitialMainWindow : public QMainWindow
{
    Q_OBJECT

public:
    InitialMainWindow(QWidget* parent = Q_NULLPTR);
    ~InitialMainWindow();

protected:
    QToolBar mainToolBar;

    QMenuBar menuBar;
    QMenu fileMenu;

    QAction OpenSceneAction;
    QAction RunExampleAction;

    DisplayMainWindow* m_DisplayMainWindow;

    bool isDisplayMainWindowExist;

    void setupMenu();
    void setupTool();

    void closeEvent(QCloseEvent* e);
private:
    QWidget centralWidget;


private slots:
    void RunExample();
    void DisplayMainWindowClosed();

};







}



#endif



