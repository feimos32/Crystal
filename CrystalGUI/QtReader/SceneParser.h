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

#ifndef __SceneParser_h__
#define __SceneParser_h__

#include <QObject>
#include <QString>

#include <QtXml\QtXml>
#include <QtXml\QDomDocument>

#include "CrystalGUI/DebugTools/DebugStd.h"

#include "CrystalAlgrithm/VisInteractor/CameraPreset.h"
#include "CrystalAlgrithm/Light/LightPreset.h"
#include "CrystalAlgrithm/DataOrganiz/DataPreset.h"
#include "CrystalAlgrithm/DataMapper/DataMapperPreset.h"
#include "CrystalAlgrithm/Reader/ScenePreset.h"

namespace CrystalGUI {

inline QString obtainDirFromFilePath(QString filepath) {
	int last_slash_idx = filepath.lastIndexOf('\\');
	int last_slash_idx_2 = filepath.lastIndexOf('/');
	
	if (last_slash_idx > -1 || last_slash_idx_2 > -1) {
		int index = std::max(last_slash_idx, last_slash_idx_2);
		QString fileDir = filepath.mid(0, index);
		return fileDir;
	}
	return "";
}

inline QString obtainFileNameFromFilePath(QString filepath) {
	int last_slash_idx = filepath.lastIndexOf('\\');
	int last_slash_idx_2 = filepath.lastIndexOf('/');

	int index = std::max(last_slash_idx, last_slash_idx_2);
	index = std::max(index, 0);
	QString fileDir = filepath.mid(index + 1, filepath.size() - index - 1);
	return fileDir;
}

class SceneParser : public QObject {
	Q_OBJECT

public:
	SceneParser(QObject* parent = Q_NULLPTR);
	~SceneParser();

	void setFilePath(QString fp) {
		filePath = fp;
	}
	bool readSceneXML();

	bool readCameraXML();
	bool readLightXML();
	bool readDataMapperXML();
	bool readDataXML();

private:
	CrystalAlgrithm::ScenePreset m_ScenePreset;

	QDomDocument reader;
	QDomDocument writer;

	QString filePath;
	QString fileDir;
	QString fileName;

	QString rootName;
};



}

#endif




