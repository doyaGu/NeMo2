#ifndef VXMATH_H
#define VXMATH_H

#include "VxMathDefines.h"

// Containers
#include "XP.h"
#include "XSmartPtr.h"
#include "XString.h"
#include "XArray.h"
#include "XSArray.h"
#include "XClassArray.h"
#include "XList.h"
#include "XHashTable.h"
#include "XSHashTable.h"

// Port Class Utility
#include "VxSharedLibrary.h"
#include "VxMemoryMappedFile.h"
#include "CKPathSplitter.h"
#include "CKDirectoryParser.h"
#include "VxWindowFunctions.h"
#include "VxVector.h"
#include "Vx2dVector.h"
#include "VxMatrix.h"
#include "VxConfiguration.h"
#include "VxQuaternion.h"
#include "VxRect.h"
#include "VxOBB.h"
#include "VxRay.h"
#include "VxSphere.h"
#include "VxPlane.h"
#include "VxIntersect.h"
#include "VxDistance.h"
#include "VxFrustum.h"
#include "VxColor.h"
#include "VxMemoryPool.h"
#include "VxTimeProfiler.h"
#include "VxImageDescEx.h"

// Threads and Synchro
#include "VxMutex.h"
#include "VxThread.h"

/**
 * @brief Initializes the VxMath library. Should be called upon dynamic library load.
 */
void InitVxMath();

/**
 * @brief Detects and caches processor features.
 */
void VxDetectProcessor();

/**
 * @brief Performs linear interpolation between two float arrays.
 * @param Res Pointer to the resulting array.
 * @param array1 Pointer to the first source array.
 * @param array2 Pointer to the second source array.
 * @param factor Interpolation factor, from 0.0 (array1) to 1.0 (array2).
 * @param count The number of floats in the arrays.
 */
VX_EXPORT void InterpolateFloatArray(void *Res, void *array1, void *array2, float factor, int count);

/**
 * @brief Performs linear interpolation between two vector arrays with specified strides.
 * @param Res Pointer to the resulting strided array.
 * @param Inarray1 Pointer to the first source strided array.
 * @param Inarray2 Pointer to the second source strided array.
 * @param factor Interpolation factor, from 0.0 (Inarray1) to 1.0 (Inarray2).
 * @param count The number of vectors to interpolate.
 * @param StrideRes The stride in bytes for the result array.
 * @param StrideIn The stride in bytes for the input arrays.
 */
VX_EXPORT void InterpolateVectorArray(void *Res, void *Inarray1, void *Inarray2, float factor, int count, XULONG StrideRes, XULONG StrideIn);

/**
 * @brief Transforms a 3D bounding box to 2D screen space extents.
 * @param World_ProjectionMat The combined world, view, and projection matrix.
 * @param box The 3D bounding box to transform.
 * @param ScreenSize The dimensions of the screen. Can be NULL.
 * @param Extents Output parameter for the resulting 2D rectangle. Can be NULL.
 * @param OrClipFlags Output parameter for the combined (ORed) clip flags of the box's vertices.
 * @param AndClipFlags Output parameter for the common (ANDed) clip flags of the box's vertices.
 * @return TRUE if the box is at least partially visible, FALSE otherwise.
 */
VX_EXPORT XBOOL VxTransformBox2D(const VxMatrix &World_ProjectionMat, const VxBbox &box, VxRect *ScreenSize, VxRect *Extents, VXCLIP_FLAGS &OrClipFlags, VXCLIP_FLAGS &AndClipFlags);

/**
 * @brief Projects the Z-extents of a bounding box into homogeneous clip space.
 * @param World_ProjectionMat The combined world, view, and projection matrix.
 * @param box The 3D bounding box to project.
 * @param ZhMin Output parameter for the minimum projected Z value.
 * @param ZhMax Output parameter for the maximum projected Z value.
 */
VX_EXPORT void VxProjectBoxZExtents(const VxMatrix &World_ProjectionMat, const VxBbox &box, float &ZhMin, float &ZhMax);

/**
 * @brief Fills a strided structure array with a single source data block.
 * @param Count The number of elements to fill.
 * @param Dst Pointer to the destination strided array.
 * @param Stride The stride in bytes of the destination array.
 * @param SizeSrc The size in bytes of the source data block.
 * @param Src Pointer to the source data block.
 * @return TRUE on success, FALSE otherwise.
 */
VX_EXPORT XBOOL VxFillStructure(int Count, void *Dst, XULONG Stride, XULONG SizeSrc, void *Src);

/**
 * @brief Copies data from one strided array to another.
 * @param Count The number of elements to copy.
 * @param Dst Pointer to the destination strided array.
 * @param OutStride The stride in bytes of the destination array.
 * @param SizeSrc The size in bytes of a single source element.
 * @param Src Pointer to the source strided array.
 * @param InStride The stride in bytes of the source array.
 * @return TRUE on success, FALSE otherwise.
 */
VX_EXPORT XBOOL VxCopyStructure(int Count, void *Dst, XULONG OutStride, XULONG SizeSrc, void *Src, XULONG InStride);

/**
 * @brief Copies elements from a source array to a destination array using an index list.
 * @param Dst The destination strided data description.
 * @param Src The source strided data description.
 * @param SizeSrc The size in bytes of a single source element.
 * @param Indices An array of integers specifying the indices to copy from the source.
 * @param IndexCount The number of indices in the `Indices` array.
 * @return TRUE on success, FALSE otherwise.
 */
VX_EXPORT XBOOL VxIndexedCopy(const VxStridedData &Dst, const VxStridedData &Src, XULONG SizeSrc, int *Indices, int IndexCount);

/**
 * @brief Performs a bit-block transfer (blit) from a source image to a destination image.
 * @param src_desc The description of the source image.
 * @param dst_desc The description of the destination image.
 */
VX_EXPORT void VxDoBlit(const VxImageDescEx &src_desc, const VxImageDescEx &dst_desc);

/**
 * @brief Performs a bit-block transfer, flipping the source image vertically.
 * @param src_desc The description of the source image.
 * @param dst_desc The description of the destination image.
 */
VX_EXPORT void VxDoBlitUpsideDown(const VxImageDescEx &src_desc, const VxImageDescEx &dst_desc);

/**
 * @brief Sets the alpha channel of an image to a constant value.
 * @param dst_desc The description of the destination image.
 * @param AlphaValue The 8-bit alpha value to set for all pixels.
 */
VX_EXPORT void VxDoAlphaBlit(const VxImageDescEx &dst_desc, XBYTE AlphaValue);

/**
 * @brief Sets the alpha channel of an image using an array of alpha values.
 * @param dst_desc The description of the destination image.
 * @param AlphaValues A pointer to an array of 8-bit alpha values, one for each pixel.
 */
VX_EXPORT void VxDoAlphaBlit(const VxImageDescEx &dst_desc, XBYTE *AlphaValues);

/**
 * @brief Gets the number of bits for each color component from an image description.
 * @param desc The image description.
 * @param Rbits Output parameter for the number of red bits.
 * @param Gbits Output parameter for the number of green bits.
 * @param Bbits Output parameter for the number of blue bits.
 * @param Abits Output parameter for the number of alpha bits.
 */
VX_EXPORT void VxGetBitCounts(const VxImageDescEx &desc, XULONG &Rbits, XULONG &Gbits, XULONG &Bbits, XULONG &Abits);

/**
 * @brief Gets the bit shifts for each color component from an image description.
 * @param desc The image description.
 * @param Rshift Output parameter for the red component bit shift.
 * @param Gshift Output parameter for the green component bit shift.
 * @param Bshift Output parameter for the blue component bit shift.
 * @param Ashift Output parameter for the alpha component bit shift.
 */
VX_EXPORT void VxGetBitShifts(const VxImageDescEx &desc, XULONG &Rshift, XULONG &Gshift, XULONG &Bshift, XULONG &Ashift);

/**
 * @brief Generates the next mipmap level from a source image.
 * @param src_desc The description of the source image.
 * @param DestBuffer Pointer to the destination buffer for the new mipmap.
 */
VX_EXPORT void VxGenerateMipMap(const VxImageDescEx &src_desc, XBYTE *DestBuffer);

/**
 * @brief Resizes a 32-bit image.
 * @param src_desc The description of the source image.
 * @param dst_desc The description of the destination image with the target dimensions.
 */
VX_EXPORT void VxResizeImage32(const VxImageDescEx &src_desc, const VxImageDescEx &dst_desc);

/**
 * @brief Converts an image to a normal map.
 * @param image The image to convert, its data will be modified in place.
 * @param ColorMask A mask indicating which channel (R, G, or B) contains the height information.
 * @return TRUE on success, FALSE otherwise.
 */
VX_EXPORT XBOOL VxConvertToNormalMap(const VxImageDescEx &image, XULONG ColorMask);

/**
 * @brief Converts an image to a bump map.
 * @param image The image to convert, its data will be modified in place.
 * @return TRUE on success, FALSE otherwise.
 */
VX_EXPORT XBOOL VxConvertToBumpMap(const VxImageDescEx &image);

/**
 * @brief Gets the number of set bits in a mask.
 * @param dwMask The mask to count bits in.
 * @return The number of set bits.
 */
VX_EXPORT XULONG GetBitCount(XULONG dwMask);

/**
 * @brief Gets the bit position of the least significant bit in a mask.
 * @param dwMask The mask to analyze.
 * @return The bit shift value.
 */
VX_EXPORT XULONG GetBitShift(XULONG dwMask);

/**
 * @brief Converts an image description to a pixel format enum.
 * @param desc The image description.
 * @return The corresponding VX_PIXELFORMAT enum value.
 */
VX_EXPORT VX_PIXELFORMAT VxImageDesc2PixelFormat(const VxImageDescEx &desc);

/**
 * @brief Converts a pixel format enum to an image description.
 * @param Pf The pixel format enum value.
 * @param desc The VxImageDescEx structure to be filled.
 */
VX_EXPORT void VxPixelFormat2ImageDesc(VX_PIXELFORMAT Pf, VxImageDescEx &desc);

/**
 * @brief Gets a string representation of a pixel format.
 * @param Pf The pixel format enum value.
 * @return A constant string describing the pixel format.
 */
VX_EXPORT const char *VxPixelFormat2String(VX_PIXELFORMAT Pf);

/**
 * @brief Fills the color masks in an image description based on its bits-per-pixel value.
 * @param desc The image description to modify.
 */
VX_EXPORT void VxBppToMask(VxImageDescEx &desc);

/**
 * @brief Gets the current sampling factor used for quantization.
 * @return The quantization sampling factor.
 */
VX_EXPORT int GetQuantizationSamplingFactor();

/**
 * @brief Sets the sampling factor for quantization.
 * @param sf The new sampling factor.
 */
VX_EXPORT void SetQuantizationSamplingFactor(int sf);

/**
 * @brief Gets a string describing the system's processor.
 * @return A string with the processor's description.
 */
VX_EXPORT char *GetProcessorDescription();

/**
 * @brief Gets the frequency of the processor in MHz.
 * @return The processor frequency.
 */
VX_EXPORT int GetProcessorFrequency();

/**
 * @brief Gets a bitmask of detected processor features (e.g., MMX, SSE).
 * @return A bitmask of processor features.
 */
VX_EXPORT XULONG GetProcessorFeatures();

/**
 * @brief Manually adds or removes processor features from the detected set.
 * @param Add A bitmask of features to add.
 * @param Remove A bitmask of features to remove.
 */
VX_EXPORT void ModifyProcessorFeatures(XULONG Add, XULONG Remove);

/**
 * @brief Gets the general type of the processor.
 * @return A ProcessorsType enum value.
 */
VX_EXPORT ProcessorsType GetProcessorType();

/**
 * @brief Gets a bitmask of instruction set extensions supported by the processor.
 * @return A bitmask of instruction set extensions.
 */
VX_EXPORT XULONG GetInstructionSetExtensions();

/**
 * @brief Checks if a point is inside a rectangle.
 * @param rect Pointer to the CKRECT structure.
 * @param pt Pointer to the CKPOINT structure.
 * @return TRUE if the point is inside the rectangle, FALSE otherwise.
 */
VX_EXPORT XBOOL VxPtInRect(CKRECT *rect, CKPOINT *pt);

/**
 * @brief Computes the best-fit oriented bounding box for a set of points.
 * @param Points Pointer to the first point in a strided array of points.
 * @param Stride The stride in bytes between points.
 * @param Count The number of points.
 * @param BBoxMatrix Output parameter for the transformation matrix of the resulting OBB.
 * @param AdditionalBorder A value to expand the computed box by.
 * @return TRUE on success, FALSE otherwise.
 */
VX_EXPORT XBOOL VxComputeBestFitBBox(const XBYTE *Points, XULONG Stride, int Count, VxMatrix &BBoxMatrix, float AdditionalBorder);

#endif // VXMATH_H
