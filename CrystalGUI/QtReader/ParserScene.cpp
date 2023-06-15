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

#include "ParserScene.h"

#define ParserSceneDebug true

namespace CrystalGUI {

ParserScene::ParserScene(QObject* parent) {

}

ParserScene::~ParserScene() {

}

bool ParserScene::readSceneXML() {

	if (filePath == "") {
		if (ParserSceneDebug) {
			PrintError("(filePath == \"\")");
		}
		return false;
	}

	QFile file(filePath);
	if (!file.open(QIODevice::ReadOnly)) {
		if (ParserSceneDebug) {
			PrintError( "(!file.open(QIODevice::ReadOnly))" );
		}
		return false;
	}

	if (!reader.setContent(&file)) {
		file.close();
		if (ParserSceneDebug) {
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

	if (ParserSceneDebug) {
		PrintValue("filePath", filePath);
		PrintValue("fileDir", fileDir);
		PrintValue("fileName", fileName);
	}

	// version
	QDomNode firstChild = reader.firstChild();
	if (firstChild.nodeName() == "xml") {
		if(ParserSceneDebug) {
			PrintValue("firstChild.nodeName()", firstChild.nodeName());
			PrintValue("firstChild.nodeValue()", firstChild.nodeValue());
		}
	}
	else {
		// "No version , No Format"
		if (ParserSceneDebug) {
			PrintValue_Std("No version , No Format");
		}
	}

	QDomElement root = reader.documentElement();
	rootName = root.tagName();
	if (ParserSceneDebug) {
		PrintValue("rootName", rootName);
	}

	bool readFlag = true;
	QDomNode child = root.firstChild();
	while (!child.isNull())
	{
		QDomElement e = child.toElement(); // try to convert the node to an element. 
		if (ParserSceneDebug) {
			PrintValue("e.tagName()", e.tagName());
		}
		
		if ("MedicalData" == e.tagName()) {
			readSceneMedicalDataXML(child.childNodes());
		}else if ("Camera" == e.tagName()) {
			readSceneCameraXML(child.childNodes());
		}
		else if ("SceneGeometry" == e.tagName()) {
			readSceneSceneGeometryXML(child.childNodes());
		}
		else if ("DataMapper" == e.tagName()) {
			readSceneDataMapperXML(child.childNodes());
		}
		else if ("Visualizer" == e.tagName()) {
			readSceneVisualizerXML(child.childNodes());
		}
		else {
			readFlag = false;
			PrintError("Error: Unwanted tag name while parsering Scene Xml: " + e.tagName().toStdString());
		}
		//find next node
		child = child.nextSiblingElement();
	}
	return readFlag;
}

bool ParserScene::readSceneMedicalDataXML(const QDomNodeList& nodes) {

	bool readFlag = true;
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
		else {
			readFlag = false;
			PrintError("Error: Unwanted tag name while parsering Scene Xml: " + tag.toStdString());
		}
	}
	if (ParserSceneDebug) {
		m_ScenePreset.m_DataPreset.PrintDataPreset();
	}
	return readFlag;
}

bool ParserScene::readSceneCameraXML(const QDomNodeList& nodes) {
	bool readFlag = true;
	for (int i = 0; i < nodes.count(); i++) {
		QDomNode childNode = nodes.at(i);
		QString tag = childNode.toElement().tagName();
		if ("CameraType" == tag) {
			m_ScenePreset.m_CameraPreset.CameraType =
				childNode.toElement().attribute("type").toStdString();
		}
		else {
			readFlag = false;
			PrintError("Error: Unwanted tag name while parsering Scene Xml: " + tag.toStdString());
		}
	}
	if (ParserSceneDebug) {
		m_ScenePreset.m_CameraPreset.PrintCameraPreset();
	}
	return readFlag;
}

bool ParserScene::readSceneSceneGeometryXML(const QDomNodeList& nodes) {
	bool readFlag = true;
	for (int i = 0; i < nodes.count(); i++) {
		QDomNode childNode = nodes.at(i);
		QString tag = childNode.toElement().tagName();
		if ("SceneGeometryFile" == tag) {
			m_ScenePreset.m_SceneGeometryPreset.SceneGeometryFile =
				childNode.toElement().attribute("file").toStdString();
		}
		else {
			readFlag = false;
			PrintError("Error: Unwanted tag name while parsering Scene Xml: " + tag.toStdString());
		}
	}
	if (ParserSceneDebug) {
		m_ScenePreset.m_SceneGeometryPreset.PrintLightPreset();
	}
	return readFlag;
}

bool ParserScene::readSceneDataMapperXML(const QDomNodeList& nodes) {

	bool readFlag = true;
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
		else {
			readFlag = false;
			PrintError("Error: Unwanted tag name while parsering Scene Xml: " + tag.toStdString());
		}
	}
	if (ParserSceneDebug) {
		m_ScenePreset.m_DataMapperPreset.PrintDataMapperPreset();
	}

	return readFlag;
}

bool ParserScene::readSceneVisualizerXML(const QDomNodeList& nodes) {
	bool readFlag = true;
	for (int i = 0; i < nodes.count(); i++) {
		QDomNode childNode = nodes.at(i);
		QString tag = childNode.toElement().tagName();
		if ("VisualizerType" == tag) {
			m_ScenePreset.m_VisualizerPreset.VisualizerType =
				childNode.toElement().attribute("type").toStdString();
		}
		else if ("FrameBufferSize" == tag) {
			m_ScenePreset.m_VisualizerPreset.width =
				childNode.toElement().attribute("width").toULongLong();
			m_ScenePreset.m_VisualizerPreset.height =
				childNode.toElement().attribute("height").toULongLong();
		}
		else {
			readFlag = false;
			PrintError("Error: Unwanted tag name while parsering Scene Xml: " + tag.toStdString());
		}
	}
	if (ParserSceneDebug) {
		m_ScenePreset.m_VisualizerPreset.PrintVisualizerPreset();
	}

	return readFlag;
}






}






