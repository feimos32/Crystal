#ifndef __Header_Export_dll_cuh__
#define __Header_Export_dll_cuh__

namespace CrystalAlgrithm {

#define EXPORT_DLL __declspec(dllexport)

// Init functions before conduct the program

class GpuDeviceInfos;
extern "C" EXPORT_DLL
void cudaDevicesInit(GpuDeviceInfos& info);


extern "C" EXPORT_DLL
void printCudaDevice();






}

#endif


