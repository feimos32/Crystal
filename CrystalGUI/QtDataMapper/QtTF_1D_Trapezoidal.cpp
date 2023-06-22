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


#include "QtTF_1D_Trapezoidal.h"

#include <QFile>


namespace CrystalGUI {




QtTF_1D_Trapezoidal::QtTF_1D_Trapezoidal(QWidget* pParent) : QtTfFuncDock(pParent){

	setWindowTitle("TF_1D_Trapezoidal");

    QFile qssfile("Resources/qss/QtTF_1D_Trapezoidal.qss");
    qssfile.open(QFile::ReadOnly);
    QString styleSheet = QString::fromLatin1(qssfile.readAll());
    this->setStyleSheet(styleSheet);

    setMinimumWidth(200);

    setWidget(&m_QScrollArea);
    m_QScrollArea.setWidget(&centerWidget);
    centerWidget.setLayout(&centerLayout);
    
}


QtTF_1D_Trapezoidal::~QtTF_1D_Trapezoidal() {

}

void QtTF_1D_Trapezoidal::Initialize(
    CrystalAlgrithm::PresetDataMapper& m_DataMapperPreset,
    CrystalAlgrithm::PresetVisualizer& m_VisualizerPreset) {


    centerLayout.addWidget(m_QtTfFuncGraphicsView_1D);

}







}




