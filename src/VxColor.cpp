#include "VxColor.h"

unsigned long RGBAFTOCOLOR(float r, float g, float b, float a) {
    // Convert floats to integers in range 0-255
    // Use truncation instead of rounding to match expected behavior
    unsigned int red = static_cast<unsigned int>(r * 255.0f);
    unsigned int green = static_cast<unsigned int>(g * 255.0f);
    unsigned int blue = static_cast<unsigned int>(b * 255.0f);
    unsigned int alpha = static_cast<unsigned int>(a * 255.0f);

    // Clamp values to 0-255 range
    red = (red > 255) ? 255 : red;
    green = (green > 255) ? 255 : green;
    blue = (blue > 255) ? 255 : blue;
    alpha = (alpha > 255) ? 255 : alpha;

    // Combine components into ARGB format
    return (alpha << 24) | (red << 16) | (green << 8) | blue;
}

unsigned long RGBAFTOCOLOR(const VxColor *col) {
    unsigned int red = static_cast<unsigned int>(col->r * 255.0f);
    unsigned int green = static_cast<unsigned int>(col->g * 255.0f);
    unsigned int blue = static_cast<unsigned int>(col->b * 255.0f);
    unsigned int alpha = static_cast<unsigned int>(col->a * 255.0f);

    // Clamp values
    red = (red > 255) ? 255 : red;
    green = (green > 255) ? 255 : green;
    blue = (blue > 255) ? 255 : blue;
    alpha = (alpha > 255) ? 255 : alpha;

    // Combine in ARGB format
    return (alpha << 24) | (red << 16) | (green << 8) | blue;
}

unsigned long BGRAFTOCOLOR(const VxColor *col) {
    unsigned int red = static_cast<unsigned int>(col->r * 255.0f);
    unsigned int green = static_cast<unsigned int>(col->g * 255.0f);
    unsigned int blue = static_cast<unsigned int>(col->b * 255.0f);
    unsigned int alpha = static_cast<unsigned int>(col->a * 255.0f);

    // Clamp values
    red = (red > 255) ? 255 : red;
    green = (green > 255) ? 255 : green;
    blue = (blue > 255) ? 255 : blue;
    alpha = (alpha > 255) ? 255 : alpha;

    // Combine in BGRA format
    return blue | (green << 8) | (red << 16) | (alpha << 24);
}