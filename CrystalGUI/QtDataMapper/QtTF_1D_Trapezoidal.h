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

#ifndef __QtTF_1D_Trapezoidal_h__
#define __QtTF_1D_Trapezoidal_h__

#include "CrystalGUI/Utility/Common.h"

#include "QtTsFuncDock.h"

#include "CrystalGUI/QtDataMapper/QtTsFuncGraphicsView_1D.h"
#include "CrystalGUI/QtDataMapper/QtNodePropertiesWidget.h"


#include <QScrollArea>
#include <QHBoxLayout>


namespace CrystalGUI {


class QtTF_1D_Trapezoidal : public QtTsFuncDock
{
	Q_OBJECT

public:
	QtTF_1D_Trapezoidal(QWidget* pParent = NULL);
	~QtTF_1D_Trapezoidal();

	virtual void Initialize(
		CrystalAlgrithm::PresetDataMapper& m_DataMapperPreset,
		CrystalAlgrithm::PresetVisualizer& m_VisualizerPreset);

protected:
	CrystalAlgrithm::TF_1D_Trapezoidal m_TF_1D_Trapezoidal;
	ParserTransferFunction m_ParserTransferFunction;

protected:
	QScrollArea m_QScrollArea;
	QWidget centerWidget;
	QHBoxLayout centerLayout;

	QtTsFuncGraphicsView_1D *m_QtTsFuncGraphicsView_1D;

	NodePropertiesWidget *m_NodePropertiesWidget;



};




}


#endif



