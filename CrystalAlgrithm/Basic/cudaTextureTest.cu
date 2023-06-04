#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <surface_functions.h>

#include <iostream>

namespace FmCUDA {


__global__ void cudaMallocTestKernel(float* devPtr, float* output, int width, int height)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < 12 && y < 8) {
		float* row = (float*)((char*)devPtr);
		output[x + y * 12] = row[x + y * 12];
	}
}

void cudaMallocTest() {

	int width = 12, height = 8; 
	float* data = new float[width * height];
	for (int i = 0; i < 12; i++) {
		for (int j = 0; j < 8; j++) {
			data[i + j * 12] = i + j * 12 + 0.31;
		}
	}

	float* devInput;
	cudaMalloc(&devInput, width * height * sizeof(float));
	cudaMemcpy(devInput, data, width * height * sizeof(float), cudaMemcpyHostToDevice);

	delete[] data;

	float* output;
	cudaMalloc(&output, width * height * sizeof(float));
	dim3    blocks((width + 15) / 16, (height + 15) / 16);
	dim3    threads(16, 16);
	cudaMallocTestKernel << <blocks, threads >> > (devInput, output, width, height);

	cudaFree(devInput);

	float* outCpu = new float[width * height];
	cudaMemcpy(outCpu, output, width * height * sizeof(float), cudaMemcpyDeviceToHost);

	std::cout << std::endl;
	std::cout << "cudaMallocTest" << std::endl;
	for (int j = 0; j < 8; j++) {
		for (int i = 0; i < 12; i++) {
			std::cout << outCpu[i + j * 12] << " ";
		}
		std::cout << std::endl;
	}
	delete[] outCpu;
	cudaFree(output);
}


__global__ void cudaMallocPitchTestKernel(float* devPtr, float* output,
	size_t pitch, int width, int height)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < 12 && y < 8) {
		float* row = (float*)((char*)devPtr + y * pitch);
		output[x + y * 12] = row[x];
	}
}

void cudaMallocPitchTest() {

	int width = 12, height = 8; size_t pitch;
	float* data = new float[width * height];
	for (int i = 0; i < 12; i++) {
		for (int j = 0; j < 8; j++) {
			data[i + j * 12] = i + j * 12 + 0.37;
		}
	}

	//Note that a patch stored in memory may have memory alignment elements
	//Therefore, it will make the pitch (width * sizeof (float)) values different from the pitch on host 
	float* devInput;
	cudaMallocPitch(&devInput, &pitch, width * sizeof(float), height);
	// std::cout << "pitch = " << pitch << std::endl;
	cudaMemcpy2D(devInput, pitch, data, width * sizeof(float), width * sizeof(float), height, cudaMemcpyHostToDevice);

	delete[] data;

	float* output;
	cudaMalloc(&output, width * height * sizeof(float));
	dim3    blocks((width + 15) / 16, (height + 15) / 16);
	dim3    threads(16, 16);
	cudaMallocPitchTestKernel << <blocks, threads >> > (devInput, output, pitch, width, height);

	cudaFree(devInput);

	float* outCpu = new float[width * height];
	cudaMemcpy(outCpu, output, width * height * sizeof(float), cudaMemcpyDeviceToHost);

	std::cout << std::endl;
	std::cout << "cudaMallocPitchTest" << std::endl;
	for (int j = 0; j < 8; j++) {
		for (int i = 0; i < 12; i++) {
			std::cout << outCpu[i + j * 12] << " ";
		}
		std::cout << std::endl;
	}
	delete[] outCpu; 
	cudaFree(output);
}


__global__ void cudaMalloc3DTestKernel(cudaPitchedPtr devPitchedPtr, float* output, int width, int height, int depth)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < width && y < height) {

		char* devPtr = (char *)devPitchedPtr.ptr;
		size_t pitch = devPitchedPtr.pitch;
		size_t slicePitch = pitch * height;

		// z = 0,1,2
		int z = 1;
		char* slice = devPtr + z * slicePitch;
		float* row = (float*)(slice + y * pitch);
		output[x + y * width] = row[x];
	}
}

void cudaMalloc3DTest() {

	int width = 12, height = 8, depth = 3;
	float* data = new float[width * height * depth];
	for (int i = 0; i < 12; i++) {
		for (int j = 0; j < 8; j++) {
			for (int k = 0; k < 3; k++) {
				data[k * 12 * 8 + j * 12 + i] 
					= k * 12 * 8 + j * 12 + i + 0.37;
			}
		}
	}

	//Note that a patch stored in memory may have memory alignment elements
	//Therefore, it will make the pitch (width * sizeof (float)) values different from the pitch on host 
	cudaExtent extent = make_cudaExtent(width * sizeof(float), height, depth);
	cudaPitchedPtr devPitchedPtr;
	cudaMalloc3D(&devPitchedPtr, extent);

	cudaMemcpy3DParms cpy = { 0 };
	cpy.srcPtr = make_cudaPitchedPtr(data, width * sizeof(float), width, height);
	cpy.dstPtr = devPitchedPtr;
	cpy.extent = make_cudaExtent(width * sizeof(float), height, depth);
	cpy.kind = cudaMemcpyHostToDevice;
	cudaMemcpy3D(&cpy);

	delete[] data;

	float* output;
	cudaMalloc(&output, width * height * sizeof(float));
	dim3    blocks((width + 15) / 16, (height + 15) / 16);
	dim3    threads(16, 16);
	cudaMalloc3DTestKernel << <blocks, threads >> > (devPitchedPtr, output, width, height, depth);

	cudaFree(devPitchedPtr.ptr);

	float* outCpu = new float[width * height];
	cudaMemcpy(outCpu, output, width * height * sizeof(float), cudaMemcpyDeviceToHost);
	
	std::cout << std::endl;
	std::cout << "cudaMalloc3DTest" << std::endl;
	for (int j = 0; j < 8; j++) {
		for (int i = 0; i < 12; i++) {
			std::cout << outCpu[j * 12 + i] << " ";
		}
		std::cout << std::endl;
	}
		
	cudaFree(output);
	delete[] outCpu;
}


/**

Its dimensionality that specifies 
whether the texture is addressed as 
a one dimensional array using one texture coordinate, 
a two-dimensional array using two texture coordinates, 
or a three-dimensional array using three texture coordinates. 

Elements of the array are called texels (short for texture elements). 
The texture width, height, and depth refer to the size of the array in each dimension. 

The maximum texture width, height, and depth depending on the compute capability of the device.

The type of a texel, which is restricted to the basic integer and single-precision floating-point types 
and any of the 1-, 2-, and 4-component vector types defined in Built-in Vector Types 
that are derived from the basic integer and single-precision floating-point types.

The read mode, which is equal to 
cudaReadModeNormalizedFloat or cudaReadModeElementType.
If it is cudaReadModeNormalizedFloat and the type of the texel is a 16-bit or 8-bit integer type, 
the value returned by the texture fetch is actually returned as floating-point type and 
the full range of the integer type
is mapped to [0.0, 1.0] for unsigned integer type and [-1.0, 1.0] for signed integer type; 
for example, an unsigned 8-bit texture element with the value 0xff reads as 1. 
If it is cudaReadModeElementType, no conversion is performed.

Whether texture coordinates are normalized or not. 
By default, textures are referenced (by the functions of Texture Functions) using floating-point coordinates 
in the range [0, N-1] where N is the size of the texture in the dimension corresponding to the coordinate. 
For example, a texture that is 64x32 in size will be referenced with coordinates 
in the range [0, 63] and [0, 31] for the x and y dimensions, respectively. 
Normalized texture coordinates cause the coordinates to be specified 
in the range [0.0, 1.0-1/N] instead of [0, N-1], 
so the same 64x32 texture would be addressed by normalized coordinates 
in the range [0, 1-1/N] in both the x and y dimensions. 
Normalized texture coordinates are a natural fit to some applications¡¯ requirements, 
if it is preferable for the texture coordinates to be independent of the texture size.

The addressing mode. It is valid to call the device functions of Section B.8 with coordinates that are out of range. 
The addressing mode defines what happens in that case. 
The default addressing mode is to clamp the coordinates to the valid range: 
[0, N) for non-normalized coordinates and [0.0, 1.0) for normalized coordinates.
If the border mode is specified instead, texture fetches with out-of-range texture coordinates return zero. 
For normalized coordinates, the wrap mode and the mirror mode are also available. 
When using the wrap mode, each coordinate x is converted to frac(x)=x - floor(x) 
where floor(x) is the largest integer not greater than x. 
When using the mirror mode, each coordinate x is converted to frac(x) 
if floor(x) is even and 1-frac(x) if floor(x) is odd. 
The addressing mode is specified as an array of size three 
whose first, second, and third elements specify the addressing mode for 
the first, second, and third texture coordinates, respectively; 
the addressing mode are cudaAddressModeBorder, cudaAddressModeClamp, cudaAddressModeWrap, and cudaAddressModeMirror; 
cudaAddressModeWrap and cudaAddressModeMirror are only supported for normalized texture coordinates

The filtering mode which specifies how the value returned 
when fetching the texture is computed based on the input texture coordinates. 
Linear texture filtering may be done only for textures that are configured to return floating-point data. 
It performs low-precision interpolation between neighboring texels. 
When enabled, the texels surrounding a texture fetch location are read 
and the return value of the texture fetch is interpolated based on 
where the texture coordinates fell between the texels. 
Simple linear interpolation is performed for one-dimensional textures, 
bilinear interpolation for two-dimensional textures, 
and trilinear interpolation for three-dimensional textures. 
The filtering mode is equal to cudaFilterModePoint or cudaFilterModeLinear. 
If it is cudaFilterModePoint, 
the returned value is the texel whose texture coordinates are the closest to the input texture coordinates. 
If it is cudaFilterModeLinear, 
the returned value is the linear interpolation of the two (for a one-dimensional texture), 
four (for a two dimensional texture), 
or eight (for a three dimensional texture) texels 
whose texture coordinates are the closest to the input texture coordinates. 
cudaFilterModeLinear is only valid for returned values of floating-point type.


A texture object is created using cudaCreateTextureObject() 
from a resource description of type struct cudaResourceDesc, which specifies the texture
struct cudaResourceDesc
struct cudaTextureDesc.
*/

// float4 Correct interpolation
__global__ void Texture1DFloatTestKernel(
	float* output, cudaTextureObject_t texObj,
	int width, int height)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < width && y < height) {

		int offset = x + y * width;

		float u = (offset + 0.5f + 0.25) / (float)(width * height);

		float4 da = tex1D<float4>(texObj, u);

		output[x + y * width] = da.x;
	}
}

void Texture1DFloatTest() {

	int width = 12, height = 8;
	float4* data = new float4[width * height];
	for (int i = 0; i < width; i++) {
		for (int j = 0; j < height; j++) {
			float data_temp = i + j * width + 0.39;
			data[i + j * width] = make_float4(data_temp, data_temp, data_temp, i*j);
		}
	}

	// Allocate CUDA array in device memory
	cudaChannelFormatDesc channelDesc =
		cudaCreateChannelDesc(8 * sizeof(float), 8 * sizeof(float), 
			8 * sizeof(float), 8 * sizeof(float), cudaChannelFormatKindFloat);
	cudaArray_t cuArray;
	cudaMallocArray(&cuArray, &channelDesc, width * height, 1);

	// Set pitch of the source (the width in memory in bytes of the 2D array pointed
	// to by src, including padding), we dont have any padding
	const size_t spitch = width * height * sizeof(float4);
	// Copy data located at address h_data in host memory to device memory
	cudaMemcpy2DToArray(cuArray, 0, 0, data, spitch, width * height * sizeof(float4),
		1, cudaMemcpyHostToDevice);

	// Specify texture
	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = cuArray;

	// Specify texture object parameters
	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaAddressModeClamp;
	texDesc.addressMode[1] = cudaAddressModeClamp;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.readMode = cudaReadModeElementType;
	texDesc.normalizedCoords = 1;

	// Create texture object
	cudaTextureObject_t texObj = 0;
	cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

	float* output;
	cudaMalloc(&output, width * height * sizeof(float));
	dim3    blocks((width + 15) / 16, (height + 15) / 16);
	dim3    threads(16, 16);
	Texture1DFloatTestKernel << <blocks, threads >> > (output, texObj, width, height);

	float* outCpu = new float[width * height];
	cudaMemcpy(outCpu, output, width * height * sizeof(float), cudaMemcpyDeviceToHost);

	std::cout << std::endl;
	std::cout << "Texture1DFloatTest" << std::endl;
	for (int j = 0; j < height; j++) {
		for (int i = 0; i < width; i++) {
			std::cout << outCpu[i + j * width] << " ";
		}
		std::cout << std::endl;
	}
	delete[] outCpu;
	cudaFree(output);

	// Destroy texture object
	cudaDestroyTextureObject(texObj);
	// Free device memory
	cudaFreeArray(cuArray);
	// Free host memory
	free(data);
}


// uchar4 Correct interpolation
__global__ void Texture1DUnsignedCharTestKernel(
	float* output, cudaTextureObject_t texObj,
	int width, int height)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < width && y < height) {

		int offset = x + y * width;

		float u = (offset + 0.5f + 0.25) / (float)(width * height);

		float4 da = tex1D<float4>(texObj, u);

		output[x + y * width] = 255 * da.x;
	}
}

void Texture1DUnsignedCharTest() {

	int width = 12, height = 8;
	uchar4* data = new uchar4[width * height];
	for (int i = 0; i < width; i++) {
		for (int j = 0; j < height; j++) {
			unsigned char data_temp = i + j * width;
			data[i + j * width] = make_uchar4(data_temp, data_temp, data_temp, i*j);
		}
	}

	// Allocate CUDA array in device memory
	cudaChannelFormatDesc channelDesc =
		cudaCreateChannelDesc(8 * sizeof(unsigned char), 8 * sizeof(unsigned char),
			8 * sizeof(unsigned char), 8 * sizeof(unsigned char), cudaChannelFormatKindUnsigned);
	cudaArray_t cuArray;
	cudaMallocArray(&cuArray, &channelDesc, width * height, 1);

	// Set pitch of the source (the width in memory in bytes of the 2D array pointed
	// to by src, including padding), we dont have any padding
	const size_t spitch = width * height * sizeof(uchar4);
	// Copy data located at address h_data in host memory to device memory
	cudaMemcpy2DToArray(cuArray, 0, 0, data, spitch, width * height * sizeof(uchar4),
		1, cudaMemcpyHostToDevice);

	// Specify texture
	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = cuArray;

	// Specify texture object parameters
	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaAddressModeClamp;
	texDesc.addressMode[1] = cudaAddressModeClamp;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.readMode = cudaReadModeNormalizedFloat;
	texDesc.normalizedCoords = 1;

	// Create texture object
	cudaTextureObject_t texObj = 0;
	cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

	float* output;
	cudaMalloc(&output, width * height * sizeof(float));
	dim3    blocks((width + 15) / 16, (height + 15) / 16);
	dim3    threads(16, 16);
	Texture1DUnsignedCharTestKernel << <blocks, threads >> > (output, texObj, width, height);

	float* outCpu = new float[width * height];
	cudaMemcpy(outCpu, output, width * height * sizeof(float), cudaMemcpyDeviceToHost);

	std::cout << std::endl;
	std::cout << "Texture1DUnsignedCharTest" << std::endl;
	for (int j = 0; j < height; j++) {
		for (int i = 0; i < width; i++) {
			std::cout << outCpu[i + j * width] << " ";
		}
		std::cout << std::endl;
	}
	delete[] outCpu;
	cudaFree(output);

	// Destroy texture object
	cudaDestroyTextureObject(texObj);
	// Free device memory
	cudaFreeArray(cuArray);
	// Free host memory
	free(data);
}


// char4 Correct interpolation
__global__ void Texture1DCharTestKernel(
	float* output, cudaTextureObject_t texObj,
	int width, int height)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < width && y < height) {

		int offset = x + y * width;

		float u = (offset + 0.5f) / (float)(width * height);

		float4 da = tex1D<float4>(texObj, u);

		output[x + y * width] = 127.5 * da.x;
	}
}

void Texture1DCharTest() {

	int width = 12, height = 8;
	char4* data = new char4[width * height];
	for (int i = 0; i < width; i++) {
		for (int j = 0; j < height; j++) {
			char data_temp = -20 + i + j * width;
			data[i + j * width] = make_char4(data_temp, data_temp, data_temp, i*j);
		}
	}

	// Allocate CUDA array in device memory
	cudaChannelFormatDesc channelDesc =
		cudaCreateChannelDesc(8 * sizeof(char), 8 * sizeof(char),
			8 * sizeof(char), 8 * sizeof(char), cudaChannelFormatKindSigned);
	cudaArray_t cuArray;
	cudaMallocArray(&cuArray, &channelDesc, width * height, 1);

	// Set pitch of the source (the width in memory in bytes of the 2D array pointed
	// to by src, including padding), we dont have any padding
	const size_t spitch = width * height * sizeof(char4);
	// Copy data located at address h_data in host memory to device memory
	cudaMemcpy2DToArray(cuArray, 0, 0, data, spitch, width * height * sizeof(char4),
		1, cudaMemcpyHostToDevice);

	// Specify texture
	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = cuArray;

	// Specify texture object parameters
	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaAddressModeClamp;
	texDesc.addressMode[1] = cudaAddressModeClamp;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.readMode = cudaReadModeNormalizedFloat;
	texDesc.normalizedCoords = 1;

	// Create texture object
	cudaTextureObject_t texObj = 0;
	cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

	float* output;
	cudaMalloc(&output, width * height * sizeof(float));
	dim3    blocks((width + 15) / 16, (height + 15) / 16);
	dim3    threads(16, 16);
	Texture1DCharTestKernel << <blocks, threads >> > (output, texObj, width, height);

	float* outCpu = new float[width * height];
	cudaMemcpy(outCpu, output, width * height * sizeof(float), cudaMemcpyDeviceToHost);

	std::cout << std::endl;
	std::cout << "Texture1DCharTestKernel" << std::endl;
	for (int j = 0; j < height; j++) {
		for (int i = 0; i < width; i++) {
			std::cout << outCpu[i + j * width] << " ";
		}
		std::cout << std::endl;
	}
	delete[] outCpu;
	cudaFree(output);

	// Destroy texture object
	cudaDestroyTextureObject(texObj);
	// Free device memory
	cudaFreeArray(cuArray);
	// Free host memory
	free(data);
}




// Float Correct interpolation
__global__ void Texture2DFloatTestKernel(
	float* output, cudaTextureObject_t texObj,
	int width, int height)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < width && y < height) {
	
		float u = (x + 0.5f + 0.34) / (float)width;
		float v = (y + 0.5f) / (float)height;

		output[x + y * width] = tex2D<float>(texObj, u, v);;
	}
}

void Texture2DFloatTest() {

	int width = 12, height = 8;
	float* data = new float[width * height];
	for (int i = 0; i < width; i++) {
		for (int j = 0; j < height; j++) {
			data[i + j * width] = i + j * width + 0.37;
		}
	}

	// Allocate CUDA array in device memory
	cudaChannelFormatDesc channelDesc =
		cudaCreateChannelDesc(8 * sizeof(float), 0, 0, 0, cudaChannelFormatKindFloat);
	cudaArray_t cuArray;
	cudaMallocArray(&cuArray, &channelDesc, width, height);

	// Set pitch of the source (the width in memory in bytes of the 2D array pointed
	// to by src, including padding), we dont have any padding
	const size_t spitch = width * sizeof(float);
	// Copy data located at address h_data in host memory to device memory
	cudaMemcpy2DToArray(cuArray, 0, 0, data, spitch, width * sizeof(float),
		height, cudaMemcpyHostToDevice);

	// Specify texture
	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = cuArray;

	// Specify texture object parameters
	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaAddressModeWrap;
	texDesc.addressMode[1] = cudaAddressModeWrap;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.readMode = cudaReadModeElementType;
	texDesc.normalizedCoords = 1;

	// Create texture object
	cudaTextureObject_t texObj = 0;
	cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

	float* output;
	cudaMalloc(&output, width * height * sizeof(float));
	dim3    blocks((width + 15) / 16, (height + 15) / 16);
	dim3    threads(16, 16);
	Texture2DFloatTestKernel << <blocks, threads >> > (output, texObj, width, height);

	float* outCpu = new float[width * height];
	cudaMemcpy(outCpu, output, width * height * sizeof(float), cudaMemcpyDeviceToHost);

	std::cout << std::endl;
	std::cout << "Texture2DFloatTest" << std::endl;
	for (int j = 0; j < height; j++) {
		for (int i = 0; i < width; i++) {
			std::cout << outCpu[i + j * width] << " ";
		}
		std::cout << std::endl;
	}
	delete[] outCpu;
	cudaFree(output);

	// Destroy texture object
	cudaDestroyTextureObject(texObj);
	// Free device memory
	cudaFreeArray(cuArray);
	// Free host memory
	free(data);
}


// Short Error interpolation
__global__ void Texture2DShortTestKernel(
	float* output, cudaTextureObject_t texObj,
	int width, int height)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < width && y < height) {

		float u = (x + 0.5f + 0.49f) / (float)width;
		float v = (y + 0.5f) / (float)height;

		output[x + y * width] = 32767 * tex2D<float>(texObj, u, v);;
	}
}

void Texture2DShortTest() {

	int width = 12, height = 8;
	short* data = new short[width * height];
	for (int i = 0; i < width; i++) {
		for (int j = 0; j < height; j++) {
			data[i + j * width] = i + j * width;
		}
	}

	// Allocate CUDA array in device memory
	cudaChannelFormatDesc channelDesc =
		cudaCreateChannelDesc(8 * sizeof(short), 0, 0, 0, cudaChannelFormatKindSigned);
	cudaArray_t cuArray;
	cudaMallocArray(&cuArray, &channelDesc, width, height);

	// Set pitch of the source (the width in memory in bytes of the 2D array pointed
	// to by src, including padding), we dont have any padding
	const size_t spitch = width * sizeof(short);
	// Copy data located at address h_data in host memory to device memory
	cudaMemcpy2DToArray(cuArray, 0, 0, data, spitch, width * sizeof(short),
		height, cudaMemcpyHostToDevice);

	// Specify texture
	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = cuArray;

	// Specify texture object parameters
	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(cudaTextureDesc));
	texDesc.addressMode[0] = cudaAddressModeWrap;
	texDesc.addressMode[1] = cudaAddressModeWrap;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.readMode = cudaReadModeNormalizedFloat;
	texDesc.normalizedCoords = 1;

	// Create texture object
	cudaTextureObject_t texObj = 0;
	cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

	float* output;
	cudaMalloc(&output, width * height * sizeof(float));
	dim3    blocks((width + 15) / 16, (height + 15) / 16);
	dim3    threads(16, 16);
	Texture2DShortTestKernel << <blocks, threads >> > (output, texObj, width, height);

	float* outCpu = new float[width * height];
	cudaMemcpy(outCpu, output, width * height * sizeof(float), cudaMemcpyDeviceToHost);

	std::cout << std::endl;
	std::cout << "Texture2DShortTest" << std::endl;
	for (int j = 0; j < height; j++) {
		for (int i = 0; i < width; i++) {
			std::cout << outCpu[i + j * width] << " ";
		}
		std::cout << std::endl;
	}
	delete[] outCpu;
	cudaFree(output);

	// Destroy texture object
	cudaDestroyTextureObject(texObj);
	// Free device memory
	cudaFreeArray(cuArray);
	// Free host memory
	free(data);
}

// UnsignedShort Error interpolation
__global__ void Texture2DUnsignedShortTestKernel(
	float* output, cudaTextureObject_t texObj,
	int width, int height)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < width && y < height) {

		float u = (x + 0.5f + 0.25f) / (float)width;
		float v = (y + 0.5f) / (float)height;

		output[x + y * width] = 65535 * tex2D<float>(texObj, u, v);;
	}
}

void Texture2DUnsignedShortTest() {

	int width = 12, height = 8;
	unsigned short* data = new unsigned short[width * height];
	for (int i = 0; i < width; i++) {
		for (int j = 0; j < height; j++) {
			data[i + j * width] = i + j * width;
		}
	}

	// Allocate CUDA array in device memory
	cudaChannelFormatDesc channelDesc =
		cudaCreateChannelDesc(8 * sizeof(unsigned short), 0, 0, 0, cudaChannelFormatKindUnsigned);
	cudaArray_t cuArray;
	cudaMallocArray(&cuArray, &channelDesc, width, height);

	// Set pitch of the source (the width in memory in bytes of the 2D array pointed
	// to by src, including padding), we dont have any padding
	const size_t spitch = width * sizeof(unsigned short);
	// Copy data located at address h_data in host memory to device memory
	cudaMemcpy2DToArray(cuArray, 0, 0, data, spitch, width * sizeof(unsigned short),
		height, cudaMemcpyHostToDevice);

	// Specify texture
	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = cuArray;

	// Specify texture object parameters
	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(cudaTextureDesc));
	texDesc.addressMode[0] = cudaAddressModeClamp;
	texDesc.addressMode[1] = cudaAddressModeClamp;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.readMode = cudaReadModeNormalizedFloat;
	texDesc.normalizedCoords = 1;

	// Create texture object
	cudaTextureObject_t texObj = 0;
	cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

	float* output;
	cudaMalloc(&output, width * height * sizeof(float));
	dim3    blocks((width + 15) / 16, (height + 15) / 16);
	dim3    threads(16, 16);
	Texture2DUnsignedShortTestKernel << <blocks, threads >> > (output, texObj, width, height);

	float* outCpu = new float[width * height];
	cudaMemcpy(outCpu, output, width * height * sizeof(float), cudaMemcpyDeviceToHost);

	std::cout << std::endl;
	std::cout << "Texture2DUnsignedShortTest" << std::endl;
	for (int j = 0; j < height; j++) {
		for (int i = 0; i < width; i++) {
			std::cout << outCpu[i + j * width] << " ";
		}
		std::cout << std::endl;
	}
	delete[] outCpu;
	cudaFree(output);

	// Destroy texture object
	cudaDestroyTextureObject(texObj);
	// Free device memory
	cudaFreeArray(cuArray);
	// Free host memory
	free(data);
}

// UnsignedChar Correct interpolation
__global__ void Texture2DUnsignedCharTestKernel(
	float* output, cudaTextureObject_t texObj,
	int width, int height)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < width && y < height) {

		float u = (x + 0.5f + 0.25f) / (float)width;
		float v = (y + 0.5f) / (float)height;

		output[x + y * width] = 255 * tex2D<float>(texObj, u, v);;
	}
}

void Texture2DUnsignedCharTest() {

	int width = 12, height = 8;
	unsigned char* data = new unsigned char[width * height];
	for (int i = 0; i < width; i++) {
		for (int j = 0; j < height; j++) {
			data[i + j * width] = i + j * width;
		}
	}

	// Allocate CUDA array in device memory
	cudaChannelFormatDesc channelDesc =
		cudaCreateChannelDesc(8 * sizeof(unsigned char), 0, 0, 0, cudaChannelFormatKindUnsigned);
	cudaArray_t cuArray;
	cudaMallocArray(&cuArray, &channelDesc, width, height);

	// Set pitch of the source (the width in memory in bytes of the 2D array pointed
	// to by src, including padding), we dont have any padding
	const size_t spitch = width * sizeof(unsigned char);
	// Copy data located at address h_data in host memory to device memory
	cudaMemcpy2DToArray(cuArray, 0, 0, data, spitch, width * sizeof(unsigned char),
		height, cudaMemcpyHostToDevice);

	// Specify texture
	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = cuArray;

	// Specify texture object parameters
	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(cudaTextureDesc));
	texDesc.addressMode[0] = cudaAddressModeClamp;
	texDesc.addressMode[1] = cudaAddressModeClamp;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.readMode = cudaReadModeNormalizedFloat;
	texDesc.normalizedCoords = 1;

	// Create texture object
	cudaTextureObject_t texObj = 0;
	cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

	float* output;
	cudaMalloc(&output, width * height * sizeof(float));
	dim3    blocks((width + 15) / 16, (height + 15) / 16);
	dim3    threads(16, 16);
	Texture2DUnsignedCharTestKernel << <blocks, threads >> > (output, texObj, width, height);

	float* outCpu = new float[width * height];
	cudaMemcpy(outCpu, output, width * height * sizeof(float), cudaMemcpyDeviceToHost);

	std::cout << std::endl;
	std::cout << "Texture2DUnsignedCharTest" << std::endl;
	for (int j = 0; j < height; j++) {
		for (int i = 0; i < width; i++) {
			std::cout << outCpu[i + j * width] << " ";
		}
		std::cout << std::endl;
	}
	delete[] outCpu;
	cudaFree(output);

	// Destroy texture object
	cudaDestroyTextureObject(texObj);
	// Free device memory
	cudaFreeArray(cuArray);
	// Free host memory
	free(data);
}





// float Correct interpolation
__global__ void Texture3DFloatTestKernel(
	float* output, cudaTextureObject_t texObj,
	int width, int height, int depth)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < width && y < height) {

		float u = (x + 0.5f) / (float)width;
		float v = (y + 0.5f) / (float)height;
		float w = (0.0f + 0.5f) / (float)depth;

		output[x + y * width] = tex3D<float>(texObj, u, v, w);
	}
}

void Texture3DFloatTest() {

	int width = 12, height = 8, depth = 3;
	float* data = new float[width * height * depth];
	for (int i = 0; i < width; i++) {
		for (int j = 0; j < height; j++) {
			for (int k = 0; k < depth; k++) {
				data[k * width * height + j * width + i]
					= k * width * height + j * width + i + 0.37;
			}
		}
	}

	cudaArray_t cuArray = 0;
	const cudaExtent volumeSize = make_cudaExtent(width, height, depth);
	// Allocate CUDA array in device memory
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	cudaMalloc3DArray(&cuArray, &channelDesc, volumeSize);

	// copy data to 3D array
	cudaMemcpy3DParms copyParams = { 0 };
	copyParams.srcPtr =
		make_cudaPitchedPtr(data, volumeSize.width * sizeof(float),
			volumeSize.width, volumeSize.height);
	copyParams.dstArray = cuArray;
	copyParams.extent = volumeSize;
	copyParams.kind = cudaMemcpyHostToDevice;
	cudaMemcpy3D(&copyParams);

	// Specify texture
	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = cuArray;

	// Specify texture object parameters
	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaAddressModeWrap;
	texDesc.addressMode[1] = cudaAddressModeWrap;
	texDesc.addressMode[2] = cudaAddressModeWrap;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.readMode = cudaReadModeElementType;
	texDesc.normalizedCoords = 1;

	// Create texture object
	cudaTextureObject_t texObj = 0;
	cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

	float* output;
	cudaMalloc(&output, width * height * sizeof(float));
	dim3    blocks((width + 15) / 16, (height + 15) / 16);
	dim3    threads(16, 16);
	Texture3DFloatTestKernel << <blocks, threads >> > (output, texObj, width, height, depth);

	float* outCpu = new float[width * height];
	cudaMemcpy(outCpu, output, width * height * sizeof(float), cudaMemcpyDeviceToHost);

	std::cout << std::endl;
	std::cout << "Texture3DFloatTest" << std::endl;
	for (int j = 0; j < height; j++) {
		for (int i = 0; i < width; i++) {
			std::cout << outCpu[i + j * width] << " ";
		}
		std::cout << std::endl;
	}
	delete[] outCpu;
	cudaFree(output);

	// Destroy texture object
	cudaDestroyTextureObject(texObj);
	// Free device memory
	cudaFreeArray(cuArray);
	// Free host memory
	free(data);
}


// Short Error interpolation
__global__ void Texture3DShortTestKernel(
	float* output, cudaTextureObject_t texObj,
	int width, int height, int depth)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < width && y < height) {

		float u = (x + 0.5f + 0.25) / (float)width;
		float v = (y + 0.5f) / (float)height;
		float w = (0.0f + 0.5f) / (float)depth;

		output[x + y * width] = 32767 * tex3D<float>(texObj, u, v, w);
	}
}

void Texture3DShortTest() {

	int width = 12, height = 8, depth = 3;
	short* data = new short[width * height * depth];
	for (int i = 0; i < width; i++) {
		for (int j = 0; j < height; j++) {
			for (int k = 0; k < depth; k++) {
				data[k * width * height + j * width + i]
					= k * width * height + j * width + i;
			}
		}
	}

	cudaArray_t cuArray = 0;
	const cudaExtent volumeSize = make_cudaExtent(width, height, depth);
	// Allocate CUDA array in device memory
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<short>();
	cudaMalloc3DArray(&cuArray, &channelDesc, volumeSize);

	// copy data to 3D array
	cudaMemcpy3DParms copyParams = { 0 };
	copyParams.srcPtr =
		make_cudaPitchedPtr(data, volumeSize.width * sizeof(short),
			volumeSize.width, volumeSize.height);
	copyParams.dstArray = cuArray;
	copyParams.extent = volumeSize;
	copyParams.kind = cudaMemcpyHostToDevice;
	cudaMemcpy3D(&copyParams);

	// Specify texture
	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = cuArray;

	// Specify texture object parameters
	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaAddressModeClamp;
	texDesc.addressMode[1] = cudaAddressModeClamp;
	texDesc.addressMode[2] = cudaAddressModeClamp;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.readMode = cudaReadModeNormalizedFloat;
	texDesc.normalizedCoords = 1;

	// Create texture object
	cudaTextureObject_t texObj = 0;
	cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

	float* output;
	cudaMalloc(&output, width * height * sizeof(float));
	dim3    blocks((width + 15) / 16, (height + 15) / 16);
	dim3    threads(16, 16);
	Texture3DShortTestKernel << <blocks, threads >> > (output, texObj, width, height, depth);

	float* outCpu = new float[width * height];
	cudaMemcpy(outCpu, output, width * height * sizeof(float), cudaMemcpyDeviceToHost);

	std::cout << std::endl;
	std::cout << "Texture3DShortTest" << std::endl;
	for (int j = 0; j < height; j++) {
		for (int i = 0; i < width; i++) {
			std::cout << outCpu[i + j * width] << " ";
		}
		std::cout << std::endl;
	}
	delete[] outCpu;
	cudaFree(output);

	// Destroy texture object
	cudaDestroyTextureObject(texObj);
	// Free device memory
	cudaFreeArray(cuArray);
	// Free host memory
	free(data);
}


// UnsignedChar Correct interpolation
__global__ void Texture3DUnsignedCharTestKernel(
	float* output, cudaTextureObject_t texObj,
	int width, int height, int depth)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < width && y < height) {

		float u = (x + 0.5f) / (float)width;
		float v = (y + 0.5f) / (float)height;
		float w = (0.0f + 0.5f) / (float)depth;

		output[x + y * width] = 255 * tex3D<float>(texObj, u, v, w);
	}
}

void Texture3DUnsignedCharTest() {

	int width = 5, height = 4, depth = 3;
	unsigned char* data = new unsigned char[width * height * depth];
	for (int i = 0; i < width; i++) {
		for (int j = 0; j < height; j++) {
			for (int k = 0; k < depth; k++) {
				data[k * width * height + j * width + i]
					= k * width * height + j * width + i;
			}
		}
	}

	cudaArray_t cuArray = 0;
	const cudaExtent volumeSize = make_cudaExtent(width, height, depth);
	// Allocate CUDA array in device memory
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<unsigned char>();
	cudaMalloc3DArray(&cuArray, &channelDesc, volumeSize);

	// copy data to 3D array
	cudaMemcpy3DParms copyParams = { 0 };
	copyParams.srcPtr =
		make_cudaPitchedPtr(data, volumeSize.width * sizeof(unsigned char),
			volumeSize.width, volumeSize.height);
	copyParams.dstArray = cuArray;
	copyParams.extent = volumeSize;
	copyParams.kind = cudaMemcpyHostToDevice;
	cudaMemcpy3D(&copyParams);

	// Specify texture
	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = cuArray;

	// Specify texture object parameters
	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaAddressModeClamp;
	texDesc.addressMode[1] = cudaAddressModeClamp;
	texDesc.addressMode[2] = cudaAddressModeClamp;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.readMode = cudaReadModeNormalizedFloat;
	texDesc.normalizedCoords = 1;

	// Create texture object
	cudaTextureObject_t texObj = 0;
	cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

	float* output;
	cudaMalloc(&output, width * height * sizeof(float));
	dim3    blocks((width + 15) / 16, (height + 15) / 16);
	dim3    threads(16, 16);
	Texture3DUnsignedCharTestKernel << <blocks, threads >> > (output, texObj, width, height, depth);

	float* outCpu = new float[width * height];
	cudaMemcpy(outCpu, output, width * height * sizeof(float), cudaMemcpyDeviceToHost);

	std::cout << std::endl;
	std::cout << "Texture3DUnsignedCharTest" << std::endl;
	for (int j = 0; j < height; j++) {
		for (int i = 0; i < width; i++) {
			std::cout << outCpu[i + j * width] << " ";
		}
		std::cout << std::endl;
	}
	delete[] outCpu;
	cudaFree(output);

	// Destroy texture object
	cudaDestroyTextureObject(texObj);
	// Free device memory
	cudaFreeArray(cuArray);
	// Free host memory
	free(data);
}


// UnsignedShort Correct interpolation
__global__ void Texture3DUnsignedShortTestKernel(
	float* output, cudaTextureObject_t texObj,
	int width, int height, int depth)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < width && y < height) {

		float u = (x + 0.5f) / (float)width;
		float v = (y + 0.5f) / (float)height;
		float w = (0.0f + 0.5f) / (float)depth;

		output[x + y * width] = 65535 * tex3D<float>(texObj, u, v, w);
	}
}

void Texture3DUnsignedShortTest() {

	int width = 8, height = 5, depth = 3;
	unsigned short* data = new unsigned short[width * height * depth];
	for (int i = 0; i < width; i++) {
		for (int j = 0; j < height; j++) {
			for (int k = 0; k < depth; k++) {
				data[k * width * height + j * width + i]
					= k * width * height + j * width + i;
			}
		}
	}

	cudaArray_t cuArray = 0;
	const cudaExtent volumeSize = make_cudaExtent(width, height, depth);
	// Allocate CUDA array in device memory
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<unsigned short>();
	cudaMalloc3DArray(&cuArray, &channelDesc, volumeSize);

	// copy data to 3D array
	cudaMemcpy3DParms copyParams = { 0 };
	copyParams.srcPtr =
		make_cudaPitchedPtr(data, volumeSize.width * sizeof(unsigned short),
			volumeSize.width, volumeSize.height);
	copyParams.dstArray = cuArray;
	copyParams.extent = volumeSize;
	copyParams.kind = cudaMemcpyHostToDevice;
	cudaMemcpy3D(&copyParams);

	// Specify texture
	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = cuArray;

	// Specify texture object parameters
	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaAddressModeClamp;
	texDesc.addressMode[1] = cudaAddressModeClamp;
	texDesc.addressMode[2] = cudaAddressModeClamp;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.readMode = cudaReadModeNormalizedFloat;
	texDesc.normalizedCoords = 1;

	// Create texture object
	cudaTextureObject_t texObj = 0;
	cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

	float* output;
	cudaMalloc(&output, width * height * sizeof(float));
	dim3    blocks((width + 15) / 16, (height + 15) / 16);
	dim3    threads(16, 16);
	Texture3DUnsignedShortTestKernel << <blocks, threads >> > (output, texObj, width, height, depth);

	float* outCpu = new float[width * height];
	cudaMemcpy(outCpu, output, width * height * sizeof(float), cudaMemcpyDeviceToHost);

	std::cout << std::endl;
	std::cout << "Texture3DUnsignedShortTestKernel" << std::endl;
	for (int j = 0; j < height; j++) {
		for (int i = 0; i < width; i++) {
			std::cout << outCpu[i + j * width] << " ";
		}
		std::cout << std::endl;
	}
	delete[] outCpu;
	cudaFree(output);

	// Destroy texture object
	cudaDestroyTextureObject(texObj);
	// Free device memory
	cudaFreeArray(cuArray);
	// Free host memory
	free(data);
}


// Texture filtering is done only within a layer, not across layers.
/**
Textures can also be layered

A one-dimensional or two-dimensional layered texture
(also known as texture array in Direct3D and array texture in OpenGL) 
is a texture made up of a sequence of layers, 
all of which are regular textures of same dimensionality, size, and data type.

A one-dimensional layered texture is addressed using an integer index 
and a floating-point texture coordinate; 
the index denotes a layer within the sequence and the coordinate addresses a texel within that layer. 
A two-dimensional layered texture is addressed using an integer index 
and two floating-point texture coordinates; 
the index denotes a layer within the sequence and 
the coordinates address a texel within that layer.

A layered texture can only be a CUDA array by calling cudaMalloc3DArray() 
with the cudaArrayLayered flag 
(and a height of zero for one-dimensional layered texture).

Layered textures are fetched using the device functions described in tex1DLayered() and tex2DLayered(). 
Texture filtering is done only within a layer, not across layers.

*/

/**

Texture Gather describes a special texture fetch, texture gather.

Texture gather is a special texture fetch that is available for two-dimensional textures only. 
It is performed by the tex2Dgather() function, which has the same parameters as tex2D(), 
plus an additional comp parameter equal to 0, 1, 2, or 3 (see tex2Dgather()). 
It returns four 32-bit numbers that correspond to the value of the component comp 
of each of the four texels that would have been used for bilinear filtering 
during a regular texture fetch. 
For example, if these texels are of values 
(253, 20, 31, 255), (250, 25, 29, 254), (249, 16, 37, 253), (251, 22, 30, 250), 
and comp is 2, tex2Dgather() returns (31, 29, 37, 30).

Note that texture coordinates are computed with only 8 bits of fractional precision. 
tex2Dgather() may therefore return unexpected results for cases 
where tex2D() would use 1.0 for one of its weights (¦Á or ¦Â, see Linear Filtering). 
For example, with an x texture coordinate of 2.49805: xB=x-0.5=1.99805, 
however the fractional part of xB is stored in an 8-bit fixed-point format. 
Since 0.99805 is closer to 256.f/256.f than it is to 255.f/256.f, 
xB has the value 2. A tex2Dgather() in this case would therefore return indices 2 and 3 in x, 
instead of indices 1 and 2.

Texture gather is only supported for CUDA arrays 
created with the cudaArrayTextureGather flag and of 
width and height less than the maximum specified in Table 15 for texture gather, 
which is smaller than for regular texture fetch.

Texture gather is only supported on devices of compute capability 2.0 and higher.

*/




// Surface Memory
/**

For devices of compute capability 2.0 and higher, 
a CUDA array (described in Cubemap Surfaces), 
created with the cudaArraySurfaceLoadStore flag, 
can be read and written via a surface object using the functions described in Surface Functions.


A surface object is created using cudaCreateSurfaceObject() 
from a resource description of type struct cudaResourceDesc. 
Unlike texture memory, surface memory uses byte addressing. 
This means that the x-coordinate used to access a texture element via texture functions 
needs to be multiplied by the byte size of the element to access the same element 
via a surface function. For example, the element at texture coordinate x 
of a one-dimensional floating-point CUDA array bound to a texture object texObj 
and a surface object surfObj is read using tex1d(texObj, x) via texObj, 
but surf1Dread(surfObj, 4*x) via surfObj. 
Similarly, the element at texture coordinate x and y of a two-dimensional 
floating-point CUDA array bound to a texture object texObj and 
a surface object surfObj is accessed using tex2d(texObj, x, y) 
via texObj, but surf2Dread(surfObj, 4*x, y) via surObj 
(the byte offset of the y-coordinate is internally calculated from the underlying line pitch of the CUDA array).

The following code sample applies some simple transformation kernel to a surface.

*/

/**
boundaryMode specifies the boundary mode, 
that is how out-of-range surface coordinates are handled; 
it is equal to either cudaBoundaryModeClamp, 
in which case out-of-range coordinates are clamped to the valid range, 
or cudaBoundaryModeZero, in which case out-of-range reads return zero and out-of-range writes are ignored, 
or cudaBoundaryModeTrap, in which case out-of-range accesses cause the kernel execution to fail.

*/

// uchar4 type Surface memory
__global__ void Surface2DUnsignedCharWriteTestKernel(
	cudaSurfaceObject_t surfObjSrc, cudaSurfaceObject_t surfObjAim,
	int width, int height)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < width && y < height) {

		uchar4 data;
		// Read from input surface
		surf2Dread(&data, surfObjSrc, x * 4, y, cudaBoundaryModeClamp);
		// Write to output surface
		surf2Dwrite(data, surfObjAim, x * 4, y, cudaBoundaryModeClamp);
	}
}
__global__ void Surface2DUnsignedCharReadTestKernel(
	float* output, cudaSurfaceObject_t surfObjAim,
	int width, int height)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < width && y < height) {

		uchar4 data;
		// Read from input surface
		surf2Dread(&data, surfObjAim, x * 4, y);

		output[x + y * width] = data.x;
	}
}

void Surface2DUnsignedCharTest() {

	int width = 12, height = 8;
	uchar4* data = new uchar4[width * height];
	for (int i = 0; i < width; i++) {
		for (int j = 0; j < height; j++) {
			unsigned char dat_t = i + j * width;
			data[i + j * width] = make_uchar4(dat_t, dat_t, dat_t, dat_t);
		}
	}

	// Allocate CUDA array in device memory
	cudaChannelFormatDesc channelDesc =
		cudaCreateChannelDesc(8 * sizeof(unsigned char), 8 * sizeof(unsigned char), 
			8 * sizeof(unsigned char), 8 * sizeof(unsigned char), cudaChannelFormatKindUnsigned);

	cudaArray_t cuSrc;
	cudaMallocArray(&cuSrc, &channelDesc, width, height, cudaArraySurfaceLoadStore);

	cudaArray_t cuAim;
	cudaMallocArray(&cuAim, &channelDesc, width, height, cudaArraySurfaceLoadStore);

	// Set pitch of the source (the width in memory in bytes of the 2D array pointed
	// to by src, including padding), we dont have any padding
	const size_t spitch = width * sizeof(uchar4);
	// Copy data located at address h_data in host memory to device memory
	cudaMemcpy2DToArray(cuSrc, 0, 0, data, spitch, width * sizeof(uchar4),
		height, cudaMemcpyHostToDevice);

	// Specify texture
	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;

	// Create the surface objects
	resDesc.res.array.array = cuSrc;
	cudaSurfaceObject_t inputSurfObj = 0;
	cudaCreateSurfaceObject(&inputSurfObj, &resDesc);

	// Create the surface objects
	resDesc.res.array.array = cuAim;
	cudaSurfaceObject_t outputSurfObj = 0;
	cudaCreateSurfaceObject(&outputSurfObj, &resDesc);

	float* output;
	cudaMalloc(&output, width * height * sizeof(float));
	dim3    blocks((width + 15) / 16, (height + 15) / 16);
	dim3    threads(16, 16);
	Surface2DUnsignedCharWriteTestKernel << <blocks, threads >> > (inputSurfObj, outputSurfObj, width, height);

	Surface2DUnsignedCharReadTestKernel << <blocks, threads >> > (output, outputSurfObj, width, height);

	float* outCpu = new float[width * height];
	cudaMemcpy(outCpu, output, width * height * sizeof(float), cudaMemcpyDeviceToHost);

	std::cout << std::endl;
	std::cout << "Surface2DUnsignedCharTest" << std::endl;
	for (int j = 0; j < height; j++) {
		for (int i = 0; i < width; i++) {
			std::cout << outCpu[i + j * width] << " ";
		}
		std::cout << std::endl;
	}
	delete[] outCpu;
	cudaFree(output);

	// Destroy surface objects
	cudaDestroySurfaceObject(inputSurfObj);
	cudaDestroySurfaceObject(outputSurfObj);

	// Free device memory
	cudaFreeArray(cuSrc);
	cudaFreeArray(cuAim);

	// Free host memory
	free(data);
}

// float type Surface memory
__global__ void Surface3DFloatWriteTestKernel(
	cudaSurfaceObject_t surfObjSrc, cudaSurfaceObject_t surfObjAim,
	int width, int height, int depth)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < width && y < height) {

		float data;

		for (int z = 0; z < depth; z++) {
			// Read from input surface
			surf3Dread(&data, surfObjSrc, x * 4, y, z, cudaBoundaryModeClamp);
			// Write to output surface
			surf3Dwrite(data, surfObjAim, x * 4, y, z, cudaBoundaryModeClamp);
		}

	}
}
__global__ void Surface3DFloatReadTestKernel(
	float* output, cudaSurfaceObject_t surfObjAim,
	int width, int height, int depth)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < width && y < height) {

		float data;
		int z = 1;
		// Read from input surface
		surf3Dread(&data, surfObjAim, x * 4, y, z, cudaBoundaryModeClamp);

		output[x + y * width] = data;
	}
}

void Surface3DFloatTest() {

	int width = 12, height = 8, depth = 3;
	float* data = new float[width * height * depth];
	for (int i = 0; i < width; i++) {
		for (int j = 0; j < height; j++) {
			for (int k = 0; k < depth; k++) {
				data[k * width * height + j * width + i]
					= k * width * height + j * width + i + 0.37;
			}
		}
	}

	cudaArray_t cuSrc; cudaArray_t cuAim;
	const cudaExtent volumeSize = make_cudaExtent(width, height, depth);
	// Allocate CUDA array in device memory
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	cudaMalloc3DArray(&cuSrc, &channelDesc, volumeSize, cudaArraySurfaceLoadStore);
	cudaMalloc3DArray(&cuAim, &channelDesc, volumeSize, cudaArraySurfaceLoadStore);

	// copy data to 3D array
	cudaMemcpy3DParms copyParams = { 0 };
	copyParams.srcPtr =
		make_cudaPitchedPtr(data, volumeSize.width * sizeof(float),
			volumeSize.width, volumeSize.height);
	copyParams.dstArray = cuSrc;
	copyParams.extent = volumeSize;
	copyParams.kind = cudaMemcpyHostToDevice;
	cudaMemcpy3D(&copyParams);

	// Specify texture
	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;

	// Create the surface objects
	resDesc.res.array.array = cuSrc;
	cudaSurfaceObject_t inputSurfObj = 0;
	cudaCreateSurfaceObject(&inputSurfObj, &resDesc);

	// Create the surface objects
	resDesc.res.array.array = cuAim;
	cudaSurfaceObject_t outputSurfObj = 0;
	cudaCreateSurfaceObject(&outputSurfObj, &resDesc);

	float* output;
	cudaMalloc(&output, width * height * sizeof(float));
	dim3    blocks((width + 15) / 16, (height + 15) / 16);
	dim3    threads(16, 16);
	Surface3DFloatWriteTestKernel << <blocks, threads >> > (inputSurfObj, outputSurfObj, width, height, depth);

	Surface3DFloatReadTestKernel << <blocks, threads >> > (output, outputSurfObj, width, height, depth);

	float* outCpu = new float[width * height];
	cudaMemcpy(outCpu, output, width * height * sizeof(float), cudaMemcpyDeviceToHost);

	std::cout << std::endl;
	std::cout << "Surface2DUnsignedCharTest" << std::endl;
	for (int j = 0; j < height; j++) {
		for (int i = 0; i < width; i++) {
			std::cout << outCpu[i + j * width] << " ";
		}
		std::cout << std::endl;
	}
	delete[] outCpu;
	cudaFree(output);

	// Destroy surface objects
	cudaDestroySurfaceObject(inputSurfObj);
	cudaDestroySurfaceObject(outputSurfObj);

	// Free device memory
	cudaFreeArray(cuSrc);
	cudaFreeArray(cuAim);

	// Free host memory
	free(data);
}



// cudaMemcpyToSymbol(c_invViewMatrix, invViewMatrix, sizeofMatrix));













}













