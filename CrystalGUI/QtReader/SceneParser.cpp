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


#include <QObject>
#include <QString>
#include <QFileInfo>

#include <QtXml\QtXml>
#include <QtXml\QDomDocument>

#include <iostream>

#include "SceneParser.h"

#define SceneParserDebug true

namespace CrystalGUI {

SceneParser::SceneParser(QObject* parent) {

}

SceneParser::~SceneParser() {

}

bool SceneParser::readSceneXML() {

	if (filePath == "") {
		if (SceneParserDebug) {
			PrintError("(filePath == \"\")");
		}
		return false;
	}

	QFile file(filePath);
	if (!file.open(QIODevice::ReadOnly)) {
		if (SceneParserDebug) {
			PrintError( "(!file.open(QIODevice::ReadOnly))" );
		}
		return false;
	}

	if (!reader.setContent(&file)) {
		file.close();
		if (SceneParserDebug) {
			PrintError("(!reader.setContent(&file))");
		}
		return false;
	}
	file.close();

	fileName = obtainFileNameFromFilePath(filePath);
	QFileInfo fileInfo(filePath);
	fileDir = fileInfo.absolutePath();
	filePath = fileDir + "/" + fileName;
	
	m_ScenePreset.SceneFilePath = filePath.toStdString();
	m_ScenePreset.SceneFileDir = fileDir.toStdString();
	m_ScenePreset.SceneFileName = fileName.toStdString();

	if (SceneParserDebug) {
		PrintValue("filePath", filePath);
		PrintValue("fileDir", fileDir);
		PrintValue("fileName", fileName);
	}

	// version
	QDomNode firstChild = reader.firstChild();
	if (firstChild.nodeName() == "xml") {
		if(SceneParserDebug) {
			PrintValue("firstChild.nodeName()", firstChild.nodeName());
			PrintValue("firstChild.nodeValue()", firstChild.nodeValue());
		}
		//DebugText::getDebugText()->addContents(firstChild.nodeName());
		//DebugText::getDebugText()->addContents(firstChild.nodeValue());
	}
	else {
		// "No version , No Format"
		if (SceneParserDebug) {
			PrintValue_Std("No version , No Format");
		}
	}

	QDomElement root = reader.documentElement();
	rootName = root.tagName();
	if (SceneParserDebug) {
		PrintValue("rootName", rootName);
	}

	QDomNode data = root.firstChild();
	while (!data.isNull())
	{

		QDomElement e = data.toElement(); // try to convert the node to an element.  
		PrintValue("e.tagName()", e.tagName());

		//find next node
		data = data.nextSiblingElement();
	}

}

bool SceneParser::readCameraXML() {}

bool SceneParser::readLightXML() {}

bool SceneParser::readDataMapperXML() {}

bool SceneParser::readDataXML() {}






}






