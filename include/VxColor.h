#ifndef VXCOLOR_H
#define VXCOLOR_H

#include "VxMathDefines.h"
#include "XUtil.h"

#define R_SHIFT 16 ///< Bit shift for the red component in an ARGB color.
#define G_SHIFT 8  ///< Bit shift for the green component in an ARGB color.
#define B_SHIFT 0  ///< Bit shift for the blue component in an ARGB color.
#define A_SHIFT 24 ///< Bit shift for the alpha component in an ARGB color.

#define R_MASK 0x00FF0000 ///< Bitmask for the red component in an ARGB color.
#define G_MASK 0x0000FF00 ///< Bitmask for the green component in an ARGB color.
#define B_MASK 0x000000FF ///< Bitmask for the blue component in an ARGB color.
#define A_MASK 0xFF000000 ///< Bitmask for the alpha component in an ARGB color.

struct VxColor;

/**
 * @brief Converts 4 float components to a 32-bit DWORD ARGB color.
 * @param r Red component (0.0f to 1.0f).
 * @param g Green component (0.0f to 1.0f).
 * @param b Blue component (0.0f to 1.0f).
 * @param a Alpha component (0.0f to 1.0f).
 * @return A 32-bit unsigned long representing the color in ARGB format.
 */
VX_EXPORT unsigned long RGBAFTOCOLOR(float r, float g, float b, float a);

/**
 * @brief Converts a VxColor object to a 32-bit DWORD ARGB color.
 * @param col Pointer to the VxColor object to convert.
 * @return A 32-bit unsigned long representing the color in ARGB format.
 */
VX_EXPORT unsigned long RGBAFTOCOLOR(const VxColor *col);

/**
 * @brief Converts a VxColor object to a 32-bit DWORD BGRA color.
 * @param col Pointer to the VxColor object to convert.
 * @return A 32-bit unsigned long representing the color in BGRA format.
 */
// VX_EXPORT unsigned long BGRAFTOCOLOR(const VxColor *col);

/**
 * @def RGBAITOCOLOR(r, g, b, a)
 * @brief Converts 4 integer components to a 32-bit DWORD ARGB color.
 * @param r Red component (0 to 255).
 * @param g Green component (0 to 255).
 * @param b Blue component (0 to 255).
 * @param a Alpha component (0 to 255).
 * @return A 32-bit unsigned long representing the color in ARGB format.
 */
#define RGBAITOCOLOR(r, g, b, a) (((a) << A_SHIFT) | ((r) << R_SHIFT) | ((g) << G_SHIFT) | ((b) << B_SHIFT))

/**
 * @def ColorGetRed(rgb)
 * @brief Extracts the Red component from a 32-bit ARGB color.
 * @param rgb The 32-bit ARGB color.
 * @return The red component as an 8-bit integer (0-255).
 */
#define ColorGetRed(rgb) (((rgb) >> R_SHIFT) & 0xffL)

/**
 * @def ColorGetAlpha(rgb)
 * @brief Extracts the Alpha component from a 32-bit ARGB color.
 * @param rgb The 32-bit ARGB color.
 * @return The alpha component as an 8-bit integer (0-255).
 */
#define ColorGetAlpha(rgb) (((rgb) >> A_SHIFT) & 0xffL)

/**
 * @def ColorGetGreen(rgb)
 * @brief Extracts the Green component from a 32-bit ARGB color.
 * @param rgb The 32-bit ARGB color.
 * @return The green component as an 8-bit integer (0-255).
 */
#define ColorGetGreen(rgb) (((rgb) >> G_SHIFT) & 0xffL)

/**
 * @def ColorGetBlue(rgb)
 * @brief Extracts the Blue component from a 32-bit ARGB color.
 * @param rgb The 32-bit ARGB color.
 * @return The blue component as an 8-bit integer (0-255).
 */
#define ColorGetBlue(rgb) (((rgb) >> B_SHIFT) & 0xffL)

/**
 * @def ColorSetAlpha(rgba, x)
 * @brief Sets the Alpha component in a 32-bit ARGB color.
 * @param rgba The original 32-bit ARGB color.
 * @param x The new alpha value (0-255).
 * @return The modified 32-bit ARGB color.
 */
#define ColorSetAlpha(rgba, x) (((x) << A_SHIFT) | ((rgba) & ~A_MASK))

/**
 * @def ColorSetRed(rgba, x)
 * @brief Sets the Red component in a 32-bit ARGB color.
 * @param rgba The original 32-bit ARGB color.
 * @param x The new red value (0-255).
 * @return The modified 32-bit ARGB color.
 */
#define ColorSetRed(rgba, x) (((x) << R_SHIFT) | ((rgba) & ~R_MASK))

/**
 * @def ColorSetGreen(rgba, x)
 * @brief Sets the Green component in a 32-bit ARGB color.
 * @param rgba The original 32-bit ARGB color.
 * @param x The new green value (0-255).
 * @return The modified 32-bit ARGB color.
 */
#define ColorSetGreen(rgba, x) (((x) << G_SHIFT) | ((rgba) & ~G_MASK))

/**
 * @def ColorSetBlue(rgba, x)
 * @brief Sets the Blue component in a 32-bit ARGB color.
 * @param rgba The original 32-bit ARGB color.
 * @param x The new blue value (0-255).
 * @return The modified 32-bit ARGB color.
 */
#define ColorSetBlue(rgba, x) (((x) << B_SHIFT) | ((rgba) & ~B_MASK))

/**
 * @brief Structure for representing a color using four float components.
 *
 * @remarks This structure describes a color with four float values for Red, Green, Blue,
 * and Alpha components. It provides various methods for construction, conversion to
 * 32-bit ARGB format, and arithmetic operations.
 *
 * A VxColor is defined as:
 * @code
 * struct VxColor {
 *     union {
 *         struct {
 *             float r, g, b, a;
 *         };
 *         float col[4];
 *     };
 * };
 * @endcode
 */
struct VxColor {
    union {
        struct {
            float r; ///< Red component.
            float g; ///< Green component.
            float b; ///< Blue component.
            float a; ///< Alpha component.
        };
        float col[4]; ///< Array access to color components {r, g, b, a}.
    };

public:
    /// @brief Default constructor. Initializes color to (0,0,0,0).
    VxColor();
    /// @brief Constructs a color from four float components (r, g, b, a).
    VxColor(float _r, float _g, float _b, float _a);
    /// @brief Constructs a color from three float components (r, g, b), setting alpha to 1.0.
    VxColor(float _r, float _g, float _b);
    /// @brief Constructs a grayscale color, setting r, g, b to the given value and alpha to 1.0.
    VxColor(float _r);
    /// @brief Constructs a color from a 32-bit ARGB unsigned long value.
    VxColor(unsigned long col);
    /// @brief Constructs a color from four integer components (0-255).
    VxColor(int _r, int _g, int _b, int _a);
    /// @brief Constructs a color from three integer components (0-255), setting alpha to 255.
    VxColor(int _r, int _g, int _b);

    /// @brief Resets all color components to 0.0f.
    void Clear();

    /**
     * @brief Clamps all components to the [0.0, 1.0] range.
     */
    void Check() {
        if (r > 1.0f) r = 1.0f; else if (r < 0.0f) r = 0.0f;
        if (g > 1.0f) g = 1.0f; else if (g < 0.0f) g = 0.0f;
        if (b > 1.0f) b = 1.0f; else if (b < 0.0f) b = 0.0f;
        if (a > 1.0f) a = 1.0f; else if (a < 0.0f) a = 0.0f;
    }

    /// @brief Sets the color from four float components.
    void Set(float _r, float _g, float _b, float _a);
    /// @brief Sets the color from three float components, setting alpha to 1.0.
    void Set(float _r, float _g, float _b);
    /// @brief Sets a grayscale color, setting r, g, b to the given value and alpha to 1.0.
    void Set(float _r);
    /// @brief Sets the color from a 32-bit ARGB unsigned long value.
    void Set(unsigned long col);
    /// @brief Sets the color from four integer components (0-255).
    void Set(int _r, int _g, int _b, int _a);
    /// @brief Sets the color from three integer components (0-255), setting alpha to 255.
    void Set(int _r, int _g, int _b);

    /// @brief Converts the color to a 32-bit DWORD in ARGB format.
    unsigned long GetRGBA() const;
    /// @brief Converts the color to a 32-bit DWORD in ARGB format, with alpha forced to 255.
    unsigned long GetRGB() const;

    /// @brief Calculates the squared Euclidean distance between two colors in RGB space (ignores alpha).
    float GetSquareDistance(const VxColor &color) const;

    /// @brief Component-wise addition assignment.
    VxColor &operator+=(const VxColor &v);
    /// @brief Component-wise subtraction assignment.
    VxColor &operator-=(const VxColor &v);
    /// @brief Component-wise multiplication assignment.
    VxColor &operator*=(const VxColor &v);
    /// @brief Component-wise division assignment.
    VxColor &operator/=(const VxColor &v);
    /// @brief Scalar multiplication assignment.
    VxColor &operator*=(float s);
    /// @brief Scalar division assignment.
    VxColor &operator/=(float s);

    /// @brief Component-wise subtraction of two colors.
    friend VxColor operator-(const VxColor &col1, const VxColor &col2);
    /// @brief Component-wise multiplication of two colors.
    friend VxColor operator*(const VxColor &col1, const VxColor &col2);
    /// @brief Component-wise addition of two colors.
    friend VxColor operator+(const VxColor &col1, const VxColor &col2);
    /// @brief Component-wise division of two colors.
    friend VxColor operator/(const VxColor &col1, const VxColor &col2);
    /// @brief Scalar multiplication of a color.
    friend VxColor operator*(const VxColor &col, float s);

    /// @brief Checks for bitwise equality between two colors.
    friend int operator==(const VxColor &col1, const VxColor &col2);
    /// @brief Checks for bitwise inequality between two colors.
    friend int operator!=(const VxColor &col1, const VxColor &col2);

    /**
     * @brief Utility to convert float components to a 32-bit ARGB DWORD.
     * @param _r Red component (0.0f-1.0f), clamped to range.
     * @param _g Green component (0.0f-1.0f), clamped to range.
     * @param _b Blue component (0.0f-1.0f), clamped to range.
     * @param _a Alpha component (0.0f-1.0f), clamped to range. Defaults to 1.0f.
     * @return 32-bit ARGB color.
     */
    static unsigned long Convert(float _r, float _g, float _b, float _a = 1.0f) {
        XThreshold(_r, 0.0f, 1.0f);
        XThreshold(_g, 0.0f, 1.0f);
        XThreshold(_b, 0.0f, 1.0f);
        XThreshold(_a, 0.0f, 1.0f);
        return RGBAFTOCOLOR(_r, _g, _b, _a);
    }

    /**
     * @brief Utility to convert integer components to a 32-bit ARGB DWORD.
     * @param _r Red component (0-255), clamped to range.
     * @param _g Green component (0-255), clamped to range.
     * @param _b Blue component (0-255), clamped to range.
     * @param _a Alpha component (0-255), clamped to range. Defaults to 255.
     * @return 32-bit ARGB color.
     */
    static unsigned long Convert(int _r, int _g, int _b, int _a = 255) {
        XThreshold(_r, 0, 255);
        XThreshold(_g, 0, 255);
        XThreshold(_b, 0, 255);
        XThreshold(_a, 0, 255);
        return RGBAITOCOLOR(_r, _g, _b, _a);
    }
};

inline VxColor::VxColor(float _r, float _g, float _b, float _a) {
    r = _r; g = _g; b = _b; a = _a;
}

inline VxColor::VxColor() {
    r = g = b = a = 0.0f;
}

inline VxColor::VxColor(int _r, int _g, int _b, int _a) {
    r = (float) _r / 255.0f; g = (float) _g / 255.0f; b = (float) _b / 255.0f; a = (float) _a / 255.0f;
}

inline VxColor::VxColor(int _r, int _g, int _b) {
    r = (float) _r / 255.0f; g = (float) _g / 255.0f; b = (float) _b / 255.0f; a = 1.0f;
}

inline VxColor::VxColor(float _r, float _g, float _b) {
    r = _r; g = _g; b = _b; a = 1.0f;
}

inline VxColor::VxColor(float _r) {
    r = _r; g = _r; b = _r; a = 1.0f;
}

inline VxColor::VxColor(unsigned long colz) {
    r = (float) (ColorGetRed(colz)) * 0.003921568627f; // 1/255
    g = (float) (ColorGetGreen(colz)) * 0.003921568627f;
    b = (float) (ColorGetBlue(colz)) * 0.003921568627f;
    a = (float) (ColorGetAlpha(colz)) * 0.003921568627f;
}

inline float VxColor::GetSquareDistance(const VxColor &color) const {
    VxColor col = color - *this;
    return (col.r * col.r + col.g * col.g + col.b * col.b);
}

inline void VxColor::Clear() {
    r = g = b = a = 0.0f;
}

inline void VxColor::Set(float _r, float _g, float _b, float _a) {
    r = _r; g = _g; b = _b; a = _a;
}

inline void VxColor::Set(int _r, int _g, int _b, int _a) {
    r = (float) _r / 255.0f; g = (float) _g / 255.0f; b = (float) _b / 255.0f; a = (float) _a / 255.0f;
}

inline void VxColor::Set(int _r, int _g, int _b) {
    r = (float) _r / 255.0f; g = (float) _g / 255.0f; b = (float) _b / 255.0f; a = 1.0f;
}

inline void VxColor::Set(float _r, float _g, float _b) {
    r = _r; g = _g; b = _b; a = 1.0f;
}

inline void VxColor::Set(float _r) {
    r = _r; g = _r; b = _r; a = 1.0f;
}

inline void VxColor::Set(unsigned long colz) {
    r = (float) (ColorGetRed(colz)) / 255.0f; g = (float) (ColorGetGreen(colz)) / 255.0f; b = (float) (ColorGetBlue(colz)) / 255.0f; a = (float) (ColorGetAlpha(colz)) / 255.0f;
}

inline VxColor &VxColor::operator+=(const VxColor &v) {
    r += v.r; g += v.g; b += v.b; a += v.a; return *this;
}

inline VxColor &VxColor::operator-=(const VxColor &v) {
    r -= v.r; g -= v.g; b -= v.b; a -= v.a; return *this;
}

inline VxColor &VxColor::operator*=(const VxColor &v) {
    r *= v.r; g *= v.g; b *= v.b; a *= v.a; return *this;
}

inline VxColor &VxColor::operator/=(const VxColor &v) {
    r /= v.r; g /= v.g; b /= v.b; a /= v.a; return *this;
}

inline VxColor &VxColor::operator*=(float s) {
    r *= s; g *= s; b *= s; a *= s; return *this;
}

inline VxColor &VxColor::operator/=(float s) {
    r /= s; g /= s; b /= s; a /= s; return *this;
}

inline VxColor operator-(const VxColor &col1, const VxColor &col2) {
    return VxColor(col1.r - col2.r, col1.g - col2.g, col1.b - col2.b, col1.a - col2.a);
}

inline VxColor operator+(const VxColor &col1, const VxColor &col2) {
    return VxColor(col1.r + col2.r, col1.g + col2.g, col1.b + col2.b, col1.a + col2.a);
}

inline VxColor operator*(const VxColor &col1, const VxColor &col2) {
    return VxColor(col1.r * col2.r, col1.g * col2.g, col1.b * col2.b, col1.a * col2.a);
}

inline VxColor operator/(const VxColor &col1, const VxColor &col2) {
    return VxColor(col1.r / col2.r, col1.g / col2.g, col1.b / col2.b, col1.a / col2.a);
}

inline VxColor operator*(const VxColor &col, float s) {
    return VxColor(s * col.r, s * col.g, s * col.b, s * col.a);
}

inline int operator==(const VxColor &col1, const VxColor &col2) {
    return (col1.r == col2.r && col1.g == col2.g && col1.b == col2.b && col1.a == col2.a);
}

inline int operator!=(const VxColor &col1, const VxColor &col2) {
    return (col1.r != col2.r || col1.g != col2.g || col1.b != col2.b || col1.a != col2.a);
}

inline unsigned long VxColor::GetRGBA() const {
    return RGBAFTOCOLOR(this);
}

inline unsigned long VxColor::GetRGB() const {
    return RGBAFTOCOLOR(this) | A_MASK;
}

#endif // VXCOLOR_H
