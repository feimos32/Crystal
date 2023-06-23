cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 11)

project(Crystal)


########################################################################
################### Third party integration library ####################
########################################################################

########## Qt Lib ##########

# set Qt moc rcc uic
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Qt6Config.cmake
set(Qt6_DIR "D:/DevTools/Qt6/6.2.4/msvc2019_64/" CACHE PATH "qt6 cmake dir")
set(CMAKE_PREFIX_PATH ${Qt6_DIR})
find_package(Qt6 COMPONENTS 
	Widgets 
	Gui
	Core
	Xml
	OpenGLWidgets
REQUIRED)

if(Qt6_FOUND)
	message(STATUS "found Qt6") 
else()
   	message(FATAL_ERROR "Qt6 not found!")
endif()

# set(CMAKE_CXX_FLAGS "-mwindows")


############ VTK Lib ###########

set(VTK_DIR "D:/DevTools/VTK9-Install")
set(VTK_Debug_Lib_DIR ${VTK_DIR}/lib-Debug)
set(VTK_Debug_Dll_DIR ${VTK_DIR}/bin-Debug)
set(VTK_Release_Lib_DIR ${VTK_DIR}/lib-Release)
set(VTK_Release_Dll_DIR ${VTK_DIR}/bin-Release)
set(VTK_Include_DIR ${VTK_DIR}/include)
set(VTK_Include_DIR_SubDir ${VTK_Include_DIR}/vtk-9.2/)

if(EXISTS ${VTK_DIR})
  message(STATUS "VTK_DIR exists")
else()
  message(FATAL_ERROR "VTK_DIR not found!")
endif()

########## OpenMP Lib ##########

find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
	message(STATUS "found openmp") 
	set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS}, ${OPENMP_C_FLAGS})
   	set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS}, ${OPENMP_CXX_FLAGS})
else()
   	message(FATAL_ERROR "openmp not found!")
endif()

########## Nvidia CUDA Lib ##########

find_package(CUDA REQUIRED)

# Set Nvidia CUDA
set(CUDA_NVCC_FLAGS "-g -G")
# build option
set(GENCODE -gencode=arch=compute_35,code=sm_35)
set(GENCODE ${GENCODE} -gencode=arch=compute_30,code=sm_30)
set(GENCODE ${GENCODE} -gencode=arch=compute_20,code=sm_20)
set(GENCODE ${GENCODE} -gencode=arch=compute_10,code=sm_10)

enable_language(CUDA)

# include files
INCLUDE_DIRECTORIES(
	${CMAKE_CURRENT_BINARY_DIR}
	${CMAKE_CURRENT_SOURCE_DIR}
	${QT_INCLUDE_DIR}
	${CUDA_INCLUDE_DIRS}
	${VTK_Include_DIR}
	${VTK_Include_DIR_SubDir}
)

########################################################################
###################### Crystal Algrithm Project ########################
########################################################################

# macros predefined 
add_compile_definitions(_EXPORTING)

# 程序输出目录
set(publish_debug ${CMAKE_CURRENT_BINARY_DIR}/lib_out/debug) 
set(publish_release ${CMAKE_CURRENT_BINARY_DIR}/lib_out/release)
set(publish_relwithdebuginfo ${CMAKE_CURRENT_BINARY_DIR}/lib_out/relwithdebuginfo)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${publish_release})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${publish_release})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${publish_debug})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${publish_debug})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO ${publish_relwithdebuginfo})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${publish_relwithdebuginfo})

if(NOT DEFINED CMAKE_DEBUG_POSTFIX)
set(CMAKE_DEBUG_POSTFIX "_d")
endif()
if(NOT DEFINED CMAKE_RELEASE_POSTFIX)
set(CMAKE_RELEASE_POSTFIX "_r")
endif()
if(NOT DEFINED CMAKE_RELWITHDEBINFO_POSTFIX)
set(CMAKE_RELWITHDEBINFO_POSTFIX "_rd")
endif()

# Third Party Library files
set(3rdLib_stb
	CrystalAlgrithm/3rdLib_stb/stb_image.h
	CrystalAlgrithm/3rdLib_stb/stb_image_write.h
	CrystalAlgrithm/3rdLib_stb/stb_image_resize.h
)
# Make the 3rdLib_stb group
SOURCE_GROUP("3rdLib_stb" FILES ${3rdLib_stb})

# Basic
set(Basic
	CrystalAlgrithm/Basic/Common.cuh
	CrystalAlgrithm/Basic/Common.cu
	CrystalAlgrithm/Basic/Geometry.cuh
	CrystalAlgrithm/Basic/Geometry.cu
	CrystalAlgrithm/Basic/Transform.cuh
	CrystalAlgrithm/Basic/Transform.cu
	CrystalAlgrithm/Basic/Export_dll.cuh
	CrystalAlgrithm/Basic/Spectrum.cuh
	CrystalAlgrithm/Basic/Spectrum.cu
	CrystalAlgrithm/Basic/BasicTest.cu
	CrystalAlgrithm/Basic/cudaTextureTest.cu
)
# Make the Basic group
SOURCE_GROUP("Basic" FILES ${Basic})

# DataAttribute
set(DataAttribute
)
# Make the DataAttribute group
SOURCE_GROUP("DataAttribute" FILES ${DataAttribute})

# DataMapper
set(DataMapper
	CrystalAlgrithm/DataMapper/DataMapper.h
	CrystalAlgrithm/DataMapper/TransferFunction.h
	CrystalAlgrithm/DataMapper/TransferFunction.cpp
	CrystalAlgrithm/DataMapper/TF_1D_Trapezoidal.h
	CrystalAlgrithm/DataMapper/TF_1D_Trapezoidal.cpp
	CrystalAlgrithm/DataMapper/TF_2D_Trapezoidal_GF.h
	CrystalAlgrithm/DataMapper/TF_2D_Trapezoidal_GF.cpp
)
# Make the DataMapper group
SOURCE_GROUP("DataMapper" FILES ${DataMapper})

# DataOrganiz
set(DataOrganiz
	CrystalAlgrithm/DataOrganiz/DataOrganiz.h
	CrystalAlgrithm/DataOrganiz/DataOrganiz.cpp
)
# Make the DataOrganiz group
SOURCE_GROUP("DataOrganiz" FILES ${DataOrganiz})

# Light
set(Light
	CrystalAlgrithm/Light/Light.h
	CrystalAlgrithm/Light/Light.cpp
)
# Make the Light group
SOURCE_GROUP("Light" FILES ${Light})

# Postprocessor
set(Postprocessor
	CrystalAlgrithm/Postprocessor/Postprocessor.h
	CrystalAlgrithm/Postprocessor/Postprocessor.cpp
)
# Make the Postprocessor group
SOURCE_GROUP("Postprocessor" FILES ${Postprocessor})

# Preprocessor
set(Preprocessor
	CrystalAlgrithm/Preprocessor/Preprocessor.h
	CrystalAlgrithm/Preprocessor/Preprocessor.cpp
)
# Make the Preprocessor group
SOURCE_GROUP("Preprocessor" FILES ${Preprocessor})

# Reader
set(Reader
	CrystalAlgrithm/Reader/MedicalDataReader.h
	CrystalAlgrithm/Reader/MedicalImagesInfo.h
	CrystalAlgrithm/Reader/MhdFileReader.cpp
	CrystalAlgrithm/Reader/MhdFileReader.h
)
# Make the Reader group
SOURCE_GROUP("Reader" FILES ${Reader})

# Scene
set(Scene
	CrystalAlgrithm/Scene/PresetScene.h
	CrystalAlgrithm/Scene/PresetScene.cpp
	CrystalAlgrithm/Scene/Scene.h
	CrystalAlgrithm/Scene/Scene.cpp
)
# Make the Scene group
SOURCE_GROUP("Scene" FILES ${Scene})

# VisInteractor
set(VisInteractor
	CrystalAlgrithm/VisInteractor/Camera.h
	CrystalAlgrithm/VisInteractor/Camera.cpp
	CrystalAlgrithm/VisInteractor/CameraArcball.h
	CrystalAlgrithm/VisInteractor/CameraArcball.cpp
	CrystalAlgrithm/VisInteractor/CameraFps.h
	CrystalAlgrithm/VisInteractor/CameraFps.cpp
)
# Make the VisInteractor group
SOURCE_GROUP("VisInteractor" FILES ${VisInteractor})

# Visualizer
set(Visualizer
	CrystalAlgrithm/Visualizer/Visualizer.h
	CrystalAlgrithm/Visualizer/Visualizer.cpp
	CrystalAlgrithm/Visualizer/ExposureRender.h
	CrystalAlgrithm/Visualizer/ExposureRender.cpp
	CrystalAlgrithm/Visualizer/ExposureRender.cu
	CrystalAlgrithm/Visualizer/DeltaTracing.h
	CrystalAlgrithm/Visualizer/DeltaTracing.cpp
	CrystalAlgrithm/Visualizer/DeltaTracing.cu
	CrystalAlgrithm/Visualizer/RayCastingWithPhongShading.h
	CrystalAlgrithm/Visualizer/RayCastingWithPhongShading.cpp
	CrystalAlgrithm/Visualizer/RayCastingWithPhongShading.cu
	CrystalAlgrithm/Visualizer/EmisstionWithAbsorption.h
	CrystalAlgrithm/Visualizer/EmisstionWithAbsorption.cpp
	CrystalAlgrithm/Visualizer/EmisstionWithAbsorption.cu
	CrystalAlgrithm/Visualizer/FrameBuffer.h
	CrystalAlgrithm/Visualizer/FrameBuffer.cpp
)
# Make the Visualizer group
SOURCE_GROUP("Visualizer" FILES ${Visualizer})

#set(CMAKE_BUILD_TYPE "debug")

# cuda_add_library(CrystalAlgrithm STATIC ${3rdLib_stb})
cuda_add_library(CrystalAlgrithm SHARED 
	${3rdLib_stb}
	${Basic}
	${Reader}
	${Scene}
	${DataOrganiz}
	${DataMapper}
	${DataAttribute}
	${Visualizer}
	${Preprocessor}
	${Postprocessor}
	${VisInteractor}
	${Light}
)

SET(CrystalAlgrithm_LIBRARY_debug 
	# VTK Lib
	${VTK_Debug_Lib_DIR}/*.lib
)

SET(CrystalAlgrithm_LIBRARY_release 
	# VTK Lib
	${VTK_Release_Lib_DIR}/*.lib
)

target_link_libraries (CrystalAlgrithm 
						debug ${CrystalAlgrithm_LIBRARY_debug}
						optimized ${CrystalAlgrithm_LIBRARY_release} )

set_target_properties(CrystalAlgrithm PROPERTIES CUDA_ARCHITECTURES "35;50;72")
target_compile_features(CrystalAlgrithm PUBLIC cxx_std_11)
target_compile_features(CrystalAlgrithm PUBLIC cuda_std_11)

########################################################################
######################### Crystal GUI Project ##########################
########################################################################

# Main
set(Main
	CrystalGUI/Main/main.cpp
	CrystalGUI/Main/MainWindow.h
	CrystalGUI/Main/MainWindow.cpp
)
# Make the Main group
SOURCE_GROUP("Main" FILES ${Main})

# Status
set(Status
)
# Make the Status group
SOURCE_GROUP("Status" FILES ${Status})

# Display
set(Display
	CrystalGUI/Display/DisplayWidget.h
	CrystalGUI/Display/DisplayWidget.cpp
)
# Make the Display group
SOURCE_GROUP("Display" FILES ${Display})

# DebugTools
set(DebugTools
	CrystalGUI/DebugTools/DebugStd.h
	CrystalGUI/DebugTools/DebugStd.cpp
)
# Make the DebugTools group
SOURCE_GROUP("DebugTools" FILES ${DebugTools})

# Utility
set(Utility
	CrystalGUI/Utility/Utility.h
	CrystalGUI/Utility/Utility.cpp
	CrystalGUI/Utility/Common.h
)
# Make the Utility group
SOURCE_GROUP("Utility" FILES ${Utility})

# QtReader
set(QtReader
	CrystalGUI/QtReader/ParserScene.h
	CrystalGUI/QtReader/ParserScene.cpp
	CrystalGUI/QtReader/ParserLight.h
	CrystalGUI/QtReader/ParserTransferFunction.h
	CrystalGUI/QtReader/ParserTransferFunction.cpp
	CrystalGUI/QtReader/QtMedicalDataReader.h
	CrystalGUI/QtReader/QtMedicalDataReader.cpp
)
# Make the QtReader group
SOURCE_GROUP("QtReader" FILES ${QtReader})

# QtDataOrganiz
set(QtDataOrganiz
	CrystalGUI/QtDataOrganiz/QtDataOrganiz.h
	CrystalGUI/QtDataOrganiz/QtDataOrganiz.cpp
)
# Make the QtDataOrganiz group
SOURCE_GROUP("QtDataOrganiz" FILES ${QtDataOrganiz})

# QtDataMapper
set(QtDataMapper
	CrystalGUI/QtDataMapper/QtDataMapper.h
	CrystalGUI/QtDataMapper/QtDataMapper.cpp
	
	CrystalGUI/QtDataMapper/QtTsFuncDock.h
	CrystalGUI/QtDataMapper/QtTsFuncDock.cpp
	CrystalGUI/QtDataMapper/QtTsFuncGraphicsView_1D.h
	CrystalGUI/QtDataMapper/QtTsFuncGraphicsView_1D.cpp
	CrystalGUI/QtDataMapper/QtNodePropertiesWidget.h
	CrystalGUI/QtDataMapper/QtNodePropertiesWidget.cpp
	
	CrystalGUI/QtDataMapper/QtTF_1D_Trapezoidal.h
	CrystalGUI/QtDataMapper/QtTF_1D_Trapezoidal.cpp
	CrystalGUI/QtDataMapper/QtTF_2D_Trapezoidal_GF.h
	CrystalGUI/QtDataMapper/QtTF_2D_Trapezoidal_GF.cpp
)
# Make the QtDataMapper group
SOURCE_GROUP("QtDataMapper" FILES ${QtDataMapper})

# QtVisualizer
set(QtVisualizer
	CrystalGUI/QtVisualizer/QtVisualizer.h
	CrystalGUI/QtVisualizer/QtVisualizer.cpp
	CrystalGUI/QtVisualizer/QtRenderThread.h
	CrystalGUI/QtVisualizer/QtRenderThread.cpp
)
# Make the QtVisualizer group
SOURCE_GROUP("QtVisualizer" FILES ${QtVisualizer})

# QtPreprocessor
set(QtPreprocessor
	CrystalGUI/QtPreprocessor/QtPreprocessor.h
	CrystalGUI/QtPreprocessor/QtPreprocessor.cpp
)
# Make the QtPreprocessor group
SOURCE_GROUP("QtPreprocessor" FILES ${QtPreprocessor})

# QtPostprocessor
set(QtPostprocessor
	CrystalGUI/QtPostprocessor/QtPostprocessor.h
	CrystalGUI/QtPostprocessor/QtPostprocessor.cpp
)
# Make the QtPostprocessor group
SOURCE_GROUP("QtPostprocessor" FILES ${QtPostprocessor})

# QtVisInteractor
set(QtVisInteractor
	CrystalGUI/QtVisInteractor/QtVisInteractor.h
	CrystalGUI/QtVisInteractor/QtVisInteractor.cpp
)
# Make the QtVisInteractor group
SOURCE_GROUP("QtVisInteractor" FILES ${QtVisInteractor})

# QtLight
set(QtLight
	CrystalGUI/QtLight/QtLight.h
	CrystalGUI/QtLight/QtLight.cpp
)
# Make the QtLight group
SOURCE_GROUP("QtLight" FILES ${QtLight})

# QtDataAttribute
set(QtDataAttribute
)
# Make the QtDataAttribute group
SOURCE_GROUP("QtDataAttribute" FILES ${QtDataAttribute})

# Create executable
add_executable(CrystalGUI
	WIN32
	${Main}
	${Status}
	${Display}
	${DebugTools}
	${Utility}
	${QtReader}
	${QtDataOrganiz}
	${QtDataMapper}
	${QtVisualizer}
	${QtPreprocessor}
	${QtPostprocessor}
	${QtVisInteractor}
	${QtLight}
	${QtDataAttribute}
)

SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR})

SET(LIBRARY_CrystalGUI 
	# Qt Lib
	Qt6::Widgets
	Qt6::Gui
	Qt6::Core
	Qt6::Xml
	Qt6::OpenGLWidgets
)

SET(LIBRARY_debug 
	# Qt Lib
	${LIBRARY_CrystalGUI}
	# VTK Lib
	${VTK_Debug_Lib_DIR}/*.lib
	# Crystal Algrithm Lib
	${publish_debug}/*.lib
)

SET(LIBRARY_release 
	# Qt Lib
	${LIBRARY_CrystalGUI}
	# VTK Lib
	${VTK_Release_Lib_DIR}/*.lib
	# Crystal Algrithm Lib
	${publish_release}/*.lib
)

target_link_libraries (CrystalGUI PRIVATE
						debug ${LIBRARY_debug}
						# optimized ${LIBRARY_release} 
						)

# set CrystalGUI as start up project
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT CrystalGUI)

# Set to console mode to output content to the cmd
set_target_properties(CrystalGUI PROPERTIES LINK_FLAGS_RELEASE "/SUBSYSTEM:CONSOLE")
set_target_properties(CrystalGUI PROPERTIES COMPILE_DEFINITIONS_RELEASE "_CONSOLE")
set_target_properties(CrystalGUI PROPERTIES LINK_FLAGS_RELWITHDEBINFO "/SUBSYSTEM:CONSOLE")
set_target_properties(CrystalGUI PROPERTIES COMPILE_DEFINITIONS_RELWITHDEBINFO "_CONSOLE")
set_target_properties(CrystalGUI PROPERTIES LINK_FLAGS_DEBUG "/SUBSYSTEM:CONSOLE")
set_target_properties(CrystalGUI PROPERTIES LINK_FLAGS_MINSIZEREL "/SUBSYSTEM:CONSOLE")


file(GLOB Resources "${CMAKE_CURRENT_SOURCE_DIR}/Resources/*")
file(COPY ${Resources} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/Resources/)

file(GLOB Examples "${CMAKE_CURRENT_SOURCE_DIR}/Examples/*")
file(COPY ${Examples} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/Examples/)




