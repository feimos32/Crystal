
# <img src="Resources/Icons/logo.png" width="200" >

> <font size=5>Crystal - A `modern`, `fashionable`, `high-quality` 3D visualization tool for medical images</font>

<img src="Resources/Images/cVRT1.png" width="390" >

---
# <img src="Resources/Images/crystal.png" width="40" > Build this project

## Dependent third-party libraries


> <img src="Resources/Icons/logo-qt.png" width="50" >
>
> **Qt** version 6.2   
> https://www.qt.io/product/qt6


> <img src="Resources/Icons/logo-cuda.png" width="90" >
>
> **Nvidia Cuda** version 11.0   
> https://developer.nvidia.com/cuda-toolkit


> <img src="Resources/Icons/logo-vtk.png" width="70" >
>
> **VTK** version greater than 7.0   
> https://vtk.org/

## Corresponding CMake path that needs to be modified (in CMakeLists.txt)

> **Qt Dir**
> - set(Qt6_DIR "D:/DevTools/Qt6/6.2.4/msvc2019_64/" CACHE PATH "qt5 cmake dir") 

> **VTK Dir**
> - set(VTK_DIR "D:/DevTools/VTK8-Install") \
> - set(VTK_Debug_Lib_DIR ${VTK_DIR}/lib-Debug) \
> - set(VTK_Debug_Dll_DIR ${VTK_DIR}/bin-Debug) \
> - set(VTK_Release_Lib_DIR ${VTK_DIR}/lib-Release) \
> - set(VTK_Release_Dll_DIR ${VTK_DIR}/bin-Release) \
> - set(VTK_Include_DIR ${VTK_DIR}/include) 

---
# <img src="Resources/Images/crystal.png" width="40" > Supported Features

## 3D medical images Cinematic rendering

> Utilize advanced visualization techniques to process medical image data, and apply physically-based rendering techniques to achieve realistic rendering effects.

## Advanced Visualization Tools

> Implement multiple visualization processing algorithms and enable interactive operation of said algorithms.

## Support for multiple light sources

> Provide support for multiple light sources, including point light, surface light, directional light, and high dynamic range environment mapping, to illuminate medical 3D data.

## Real time denoising

> Provide real-time denoising capabilities for realistic rendering, with the intention of achieving high-quality rendering results in interaction.







