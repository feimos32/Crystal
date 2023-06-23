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

#ifndef __QtTsFuncDock_h__
#define __QtTsFuncDock_h__

#include "CrystalGUI/Utility/Common.h"

#include "CrystalAlgrithm/DataMapper/TransferFunction.h"
#include "CrystalAlgrithm/DataMapper/TF_1D_Trapezoidal.h"
#include "CrystalAlgrithm/DataMapper/TF_2D_Trapezoidal_GF.h"

#include <QDockWidget>

namespace CrystalGUI {

class QtTsFuncDock : public QDockWidget
{
	Q_OBJECT

public:
	QtTsFuncDock(QWidget* pParent = NULL);
	~QtTsFuncDock();

protected:
	std::string TsFuncType;
	CrystalAlgrithm::TF_1D_Trapezoidal m_TF_1D_Trapezoidal;
	CrystalAlgrithm::TF_2D_Trapezoidal_GF m_TF_2D_Trapezoidal_GF;



};




}


#endif



