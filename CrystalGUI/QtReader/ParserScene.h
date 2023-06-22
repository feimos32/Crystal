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

#ifndef __ParserScene_h__
#define __ParserScene_h__

#include "CrystalGUI/Utility/Common.h"

#include <QObject>
#include <QString>

#include <QtXml\QtXml>
#include <QtXml\QDomDocument>

#include "CrystalGUI/DebugTools/DebugStd.h"

#include "CrystalAlgrithm/Scene/PresetScene.h"

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
	QString fileDir = filepath.mid((size_t)index + 1, filepath.size() - index - 1);
	return fileDir;
}


class ParserScene : public QObject {
	Q_OBJECT

public:
	ParserScene(QObject* parent = Q_NULLPTR);
	~ParserScene();

	void setFilePath(QString fp) {
		filePath = fp;
	}
	bool readSceneXML();
	bool readSceneCameraXML(const QDomNodeList& nodes);
	bool readSceneSceneGeometryXML(const QDomNodeList& nodes);
	bool readSceneDataMapperXML(const QDomNodeList& nodes);
	bool readSceneMedicalDataXML(const QDomNodeList& nodes);
	bool readSceneVisualizerXML(const QDomNodeList& nodes);

	bool writeSceneXML();

	std::string getTfFuncType() {
		return m_ScenePreset.m_DataMapperPreset.TsFuncType;
	}

	CrystalAlgrithm::PresetScene m_ScenePreset;

private:
	

	QDomDocument reader;
	QDomDocument writer;

	QString filePath;
	QString fileDir;
	QString fileName;

	QString rootName;
};



}

#endif




