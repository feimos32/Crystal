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

	QDomNode child = root.firstChild();
	while (!child.isNull())
	{
		QDomElement e = child.toElement(); // try to convert the node to an element. 
		if (SceneParserDebug) {
			PrintValue("e.tagName()", e.tagName());
		}
		
		if ("Data" == e.tagName()) {
			readSceneDataXML(child.childNodes());
		}else if ("Camera" == e.tagName()) {
			readSceneCameraXML(child.childNodes());
		}
		else if ("Light" == e.tagName()) {
			readSceneLightXML(child.childNodes());
		}
		else if ("DataMapper" == e.tagName()) {
			readSceneDataMapperXML(child.childNodes());
		}
		//find next node
		child = child.nextSiblingElement();
	}
	return true;
}

bool SceneParser::readSceneDataXML(const QDomNodeList nodes) {
	for (int i = 0; i < nodes.count(); i++) {
		QDomNode childNode = nodes.at(i);
		QString tag = childNode.toElement().tagName();
		if ("DataFileType" == tag) {
			m_ScenePreset.m_DataPreset.DataFileType = 
				childNode.toElement().attribute("type").toStdString();

		} else if ("DataFilePath" == tag) {
			m_ScenePreset.m_DataPreset.DataFilePath =
				childNode.toElement().attribute("path").toStdString();

		} else if ("DataType" == tag) {
			m_ScenePreset.m_DataPreset.DataType =
				childNode.toElement().attribute("type").toStdString();
		} 
	}
	if (SceneParserDebug) {
		m_ScenePreset.m_DataPreset.PrintDataPreset();
	}
	return true;
}

bool SceneParser::readSceneCameraXML(const QDomNodeList& nodes) {
	for (int i = 0; i < nodes.count(); i++) {
		QDomNode childNode = nodes.at(i);
		QString tag = childNode.toElement().tagName();
		if ("CameraType" == tag) {
			m_ScenePreset.m_CameraPreset.CameraType =
				childNode.toElement().attribute("type").toStdString();
		}
	}
	if (SceneParserDebug) {
		m_ScenePreset.m_CameraPreset.PrintCameraPreset();
	}
	return true;
}

bool SceneParser::readSceneLightXML(const QDomNodeList& nodes) {
	for (int i = 0; i < nodes.count(); i++) {
		QDomNode childNode = nodes.at(i);
		QString tag = childNode.toElement().tagName();
		if ("LightFile" == tag) {
			m_ScenePreset.m_LightPreset.LightFile =
				childNode.toElement().attribute("file").toStdString();
		}
	}
	if (SceneParserDebug) {
		m_ScenePreset.m_LightPreset.PrintLightPreset();
	}
	return true;
}

bool SceneParser::readSceneDataMapperXML(const QDomNodeList& nodes) {
	for (int i = 0; i < nodes.count(); i++) {
		QDomNode childNode = nodes.at(i);
		QString tag = childNode.toElement().tagName();
		if ("TsFuncType" == tag) {
			m_ScenePreset.m_DataMapperPreset.TsFuncType =
				childNode.toElement().attribute("type").toStdString();
		}
		else if ("TsFuncFileName" == tag) {
			m_ScenePreset.m_DataMapperPreset.TsFuncFileName =
				childNode.toElement().attribute("file").toStdString();
		}
	}
	if (SceneParserDebug) {
		m_ScenePreset.m_DataMapperPreset.PrintDataMapperPreset();
	}

	return true;
}








}






