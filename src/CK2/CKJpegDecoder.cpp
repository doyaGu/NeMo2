#include "CKJpegDecoder.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <new>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_NO_STDIO
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBI_WRITE_NO_STDIO
#include "stb_image_write.h"

void CKJpegEncoderWriteCallback(void *context, void *data, int size) {
    if (!context || !data || size <= 0)
        return;

    auto *buffer = static_cast<std::vector<CKBYTE> *>(context);
    const CKBYTE *bytes = static_cast<const CKBYTE *>(data);
    buffer->insert(buffer->end(), bytes, bytes + size);
}

CKBOOL CKJpegDecoder::DecodeGrayscalePlane(const CKBYTE *encodedData,
                                           size_t encodedSize,
                                           int expectedWidth,
                                           int expectedHeight,
                                           CKBYTE **outPlane) {
    if (outPlane)
        *outPlane = nullptr;

    if (!encodedData || encodedSize == 0 || !outPlane || expectedWidth <= 0 || expectedHeight <= 0)
        return FALSE;

    int decodedWidth = 0;
    int decodedHeight = 0;
    int decodedChannels = 0;
    stbi_uc *decoded = stbi_load_from_memory(encodedData,
                                             static_cast<int>(encodedSize),
                                             &decodedWidth,
                                             &decodedHeight,
                                             &decodedChannels,
                                             1); // Force single channel output
    if (!decoded)
        return FALSE;

    if (decodedWidth != expectedWidth || decodedHeight != expectedHeight) {
        stbi_image_free(decoded);
        return FALSE;
    }

    const size_t planeSize = static_cast<size_t>(decodedWidth) * static_cast<size_t>(decodedHeight);
    CKBYTE *plane = new (std::nothrow) CKBYTE[planeSize];
    if (!plane) {
        stbi_image_free(decoded);
        return FALSE;
    }

    memcpy(plane, decoded, planeSize);
    stbi_image_free(decoded);

    *outPlane = plane;
    return TRUE;
}

CKBOOL CKJpegDecoder::EncodeGrayscalePlane(const CKBYTE *planeData,
                                           int width,
                                           int height,
                                           int quality,
                                           CKBYTE **outBuffer,
                                           int &outSize) {
    if (outBuffer)
        *outBuffer = nullptr;
    outSize = 0;

    if (!planeData || !outBuffer || width <= 0 || height <= 0)
        return FALSE;

    quality = std::max(1, std::min(quality, 100));

    std::vector<CKBYTE> encodedBytes;
    encodedBytes.reserve(static_cast<size_t>(width) * static_cast<size_t>(height) / 4 + 512);

    if (!stbi_write_jpg_to_func(CKJpegEncoderWriteCallback,
                                &encodedBytes,
                                width,
                                height,
                                1,
                                planeData,
                                quality)) {
        return FALSE;
    }

    if (encodedBytes.empty())
        return FALSE;

    if (encodedBytes.size() > static_cast<size_t>(std::numeric_limits<int>::max()))
        return FALSE;

    CKBYTE *buffer = new (std::nothrow) CKBYTE[encodedBytes.size()];
    if (!buffer)
        return FALSE;

    memcpy(buffer, encodedBytes.data(), encodedBytes.size());
    *outBuffer = buffer;
    outSize = static_cast<int>(encodedBytes.size());
    return TRUE;
}
