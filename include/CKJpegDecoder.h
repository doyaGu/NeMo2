#pragma once

#include "CKTypes.h"

#include <cstddef>

class CKJpegDecoder {
public:
    // Decodes a grayscale JPEG buffer into a newly allocated plane buffer of size width * height.
    // Returns TRUE on success and stores the decoded data in outPlane (caller takes ownership via delete[]).
    static CKBOOL DecodeGrayscalePlane(const CKBYTE *encodedData,
                                       size_t encodedSize,
                                       int expectedWidth,
                                       int expectedHeight,
                                       CKBYTE **outPlane);

    // Encodes an 8-bit plane into a grayscale JPEG buffer using the provided quality (1-100).
    // Returns TRUE on success and fills outBuffer/outSize with a heap allocation the caller must delete[].
    static CKBOOL EncodeGrayscalePlane(const CKBYTE *planeData,
                                       int width,
                                       int height,
                                       int quality,
                                       CKBYTE **outBuffer,
                                       int &outSize);
};
