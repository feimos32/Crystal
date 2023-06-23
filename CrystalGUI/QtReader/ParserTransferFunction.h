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

#ifndef __ParserTransferFunction_h__
#define __ParserTransferFunction_h__

#include "CrystalGUI/Utility/Common.h"

#include <QtXml\QtXml>
#include <QtXml\QDomDocument>

#include "CrystalGUI/DebugTools/DebugStd.h"

#include "CrystalAlgrithm/DataMapper/TransferFunction.h"
#include "CrystalAlgrithm/DataMapper/TF_1D_Trapezoidal.h"
#include "CrystalAlgrithm/DataMapper/TF_2D_Trapezoidal_GF.h"

namespace CrystalGUI {

class ParserTransferFunction : public QObject {
	Q_OBJECT

public:
	ParserTransferFunction(QObject* parent = Q_NULLPTR);
	~ParserTransferFunction();
	void setTsFuncPath(std::string path) {
		TsFuncPath = path;
	}
	bool Parse_TF_1D_Trapezoidal(CrystalAlgrithm::TF_1D_Trapezoidal& tf);
	bool Parse_TF_2D_Trapezoidal_GF(CrystalAlgrithm::TF_2D_Trapezoidal_GF& tf);

private:
	std::string TsFuncPath;

	QDomDocument reader;
	QDomDocument writer;




};






}


#endif








