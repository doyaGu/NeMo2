#ifndef VXVECTOR_H
#define VXVECTOR_H

#include "VxMathDefines.h"
#include "XUtil.h"

// Forward declarations
struct VxCompressedVector;
struct VxCompressedVectorOld;
class VxMatrix;

/** @name Trigonometry Utility Functions */
///@{

/**
 * @brief Converts radians to a custom angular unit (likely for table lookup).
 * @param val Angle in radians.
 * @return The angle in the custom unit.
 */
VX_EXPORT int radToAngle(float val);

/**
 * @brief Computes sine from a custom angular unit (likely using a lookup table).
 * @param angle Angle in the custom unit.
 * @return The sine of the angle.
 */
VX_EXPORT float Tsin(int angle);

/**
 * @brief Computes cosine from a custom angular unit (likely using a lookup table).
 * @param angle Angle in the custom unit.
 * @return The cosine of the angle.
 */
VX_EXPORT float Tcos(int angle);
///@}

/**
 * @brief Represents a point or direction in 3D space with x, y, and z components.
 *
 * @remarks
 * A VxVector is a fundamental class for 3D mathematics. Its elements can be
 * accessed either via named members (x, y, z) or through array-style indexing.
 *
 * @code
 * VxVector v(1.0f, 2.0f, 3.0f);
 * float x_val = v.x;    // Direct member access
 * float y_val = v[1];   // Array-style access
 * @endcode
 *
 * @see VxVector4, VxCompressedVector
 */
struct VxVector {
#if !defined(_MSC_VER)
    float x; ///< The x-component of the vector.
    float y; ///< The y-component of the vector.
    float z; ///< The z-component of the vector.
#else
    union {
        struct {
            float x; ///< The x-component of the vector.
            float y; ///< The y-component of the vector.
            float z; ///< The z-component of the vector.
        };
        float v[3]; ///< Array access to components {x, y, z}.
    };
#endif

public:
    /** @name Constructors */
    ///@{

    /// @brief Default constructor. Initializes the vector to (0, 0, 0).
    VxVector();

    /**
     * @brief Uniform constructor. Initializes all components to the same value.
     * @param f The value to assign to all three components.
     */
    VxVector(float f);

    /**
     * @brief Component-wise constructor.
     * @param _x The value for the x-component.
     * @param _y The value for the y-component.
     * @param _z The value for the z-component.
     */
    VxVector(float _x, float _y, float _z);

    /**
     * @brief Array constructor.
     * @param f A pointer to a 3-element float array [x, y, z].
     */
    VxVector(const float f[3]);
    ///@}

    /// @brief Converts each component of the vector to its absolute value.
    void Absolute();

    /**
     * @brief Returns the squared magnitude (length) of the vector.
     * @return The square of the vector's length (x*x + y*y + z*z).
     * @remarks This is more efficient than Magnitude() as it avoids a square root operation.
     * @see Magnitude()
     */
    float SquareMagnitude() const { return x * x + y * y + z * z; }

    /**
     * @brief Returns the magnitude (length) of the vector.
     * @return The length of the vector (sqrt(x*x + y*y + z*z)).
     * @warning Calling this on a null vector (0,0,0) will result in NaN (Not aNumber).
     * @see SquareMagnitude()
     */
    float Magnitude() const { return sqrtf(SquareMagnitude()); }

    /** @name Element Access */
    ///@{

    /**
     * @brief Provides const array-style access to the vector's components.
     * @param i The component index (0 for x, 1 for y, 2 for z).
     * @return A const reference to the specified component.
     */
    const float &operator[](int i) const;

    /**
     * @brief Provides array-style access to the vector's components.
     * @param i The component index (0 for x, 1 for y, 2 for z).
     * @return A reference to the specified component.
     */
    float &operator[](int i);

    /**
     * @brief Sets the three components of the vector.
     * @param X The new x-component.
     * @param Y The new y-component.
     * @param Z The new z-component.
     */
    void Set(float X, float Y, float Z);
    ///@}

    /** @name Assignment Operators */
    ///@{

    /// @brief Performs component-wise addition and assignment.
    VxVector &operator+=(const VxVector &v);
    /// @brief Performs component-wise subtraction and assignment.
    VxVector &operator-=(const VxVector &v);
    /// @brief Performs component-wise multiplication and assignment.
    VxVector &operator*=(const VxVector &v);
    /// @brief Performs component-wise division and assignment.
    VxVector &operator/=(const VxVector &v);
    /// @brief Performs scalar multiplication and assignment.
    VxVector &operator*=(float s);
    /// @brief Performs scalar division and assignment.
    VxVector &operator/=(float s);

    /**
     * @brief Assigns the value of a compressed vector to this vector.
     * @param v The compressed vector.
     * @return A reference to this modified vector.
     */
    VX_EXPORT VxVector &operator=(const VxCompressedVector &v);
    ///@}

    /**
     * @brief Computes the dot product with another vector.
     * @param iV The other vector.
     * @return The scalar dot product.
     */
    float Dot(const VxVector &iV) const { return x * iV.x + y * iV.y + z * iV.z; }

    /**
     * @brief Adds a scalar value to all components.
     * @param s The scalar value to add.
     * @return A new vector with the result.
     */
    VxVector operator+(float s) const { return VxVector(x + s, y + s, z + s); }

    /**
     * @brief Subtracts a scalar value from all components.
     * @param s The scalar value to subtract.
     * @return A new vector with the result.
     */
    VxVector operator-(float s) const { return VxVector(x - s, y - s, z - s); }

    /** @name Unary Operators */
    ///@{
    friend const VxVector operator+(const VxVector &v);
    friend const VxVector operator-(const VxVector &v);
    ///@}

    /** @name Binary Operators */
    ///@{
    friend const VxVector operator+(const VxVector &v1, const VxVector &v2);
    friend const VxVector operator-(const VxVector &v1, const VxVector &v2);
    friend const VxVector operator*(const VxVector &v, float s);
    friend const VxVector operator*(float s, const VxVector &v);
    friend const VxVector operator/(const VxVector &v, float s);
    friend const VxVector operator*(const VxVector &v1, const VxVector &v2);
    friend const VxVector operator/(const VxVector &v1, const VxVector &v2);
    friend int operator<(const VxVector &v1, const VxVector &v2);
    friend int operator<=(const VxVector &v1, const VxVector &v2);
    friend int operator==(const VxVector &v1, const VxVector &v2);
    friend int operator!=(const VxVector &v1, const VxVector &v2);
    friend float Min(const VxVector &v);
    friend float Max(const VxVector &v);
    friend const VxVector Minimize(const VxVector &v1, const VxVector &v2);
    friend const VxVector Maximize(const VxVector &v1, const VxVector &v2);
    friend const VxVector Interpolate(float step, const VxVector &v1, const VxVector &v2);
    ///@}

    /**
     * @brief Normalizes the vector, making its length equal to 1.
     * @warning The result is undefined for a null vector (0,0,0).
     */
    VX_EXPORT void Normalize();

    /**
     * @brief Rotates the vector by applying a transformation matrix.
     * @param M The rotation matrix to apply.
     */
    VX_EXPORT void Rotate(const VxMatrix &M);

    /** @name Predefined Vectors */
    ///@{
    VX_EXPORT const static VxVector &axisX(); ///< Returns a unit vector along the X-axis (1,0,0).
    VX_EXPORT const static VxVector &axisY(); ///< Returns a unit vector along the Y-axis (0,1,0).
    VX_EXPORT const static VxVector &axisZ(); ///< Returns a unit vector along the Z-axis (0,0,1).
    VX_EXPORT const static VxVector &axis0(); ///< Returns a zero vector (0,0,0).
    VX_EXPORT const static VxVector &axis1(); ///< Returns a unit vector (1,1,1).
    ///@}

private:
    /** @name Static Vector Constants */
    ///@{
    const static VxVector m_AxisX; ///< Unit vector along the X-axis.
    const static VxVector m_AxisY; ///< Unit vector along the Y-axis.
    const static VxVector m_AxisZ; ///< Unit vector along the Z-axis.
    const static VxVector m_Axis0; ///< Zero vector.
    const static VxVector m_Axis1; ///< Unit vector (1,1,1).
    ///@}
};

/** @name Vector Standalone Functions */
///@{

/**
 * @brief Returns the squared magnitude (length) of a vector.
 * @param v The vector.
 * @return The square of the vector's length (x*x + y*y + z*z).
 * @remarks This is more efficient than Magnitude() as it avoids a square root operation.
 * @see Magnitude()
 */
inline float SquareMagnitude(const VxVector &v) { return v.x * v.x + v.y * v.y + v.z * v.z; }

/**
 * @brief Returns the magnitude (length) of a vector.
 * @param v The vector.
 * @return The length of the vector (sqrt(x*x + y*y + z*z)).
 * @warning Calling this on a null vector (0,0,0) will result in NaN (Not aNumber).
 * @see SquareMagnitude()
 */
inline float Magnitude(const VxVector &v) { return sqrtf(SquareMagnitude(v)); }

/**
 * @brief Returns the inverse of the squared magnitude of a vector.
 * @param v The vector.
 * @return 1.0f / (x*x + y*y + z*z).
 * @warning The result is undefined for a null vector (0,0,0).
 * @see SquareMagnitude()
 */
inline float InvSquareMagnitude(const VxVector &v) { return 1.0f / SquareMagnitude(v); }

/**
 * @brief Returns the inverse of the magnitude of a vector.
 * @param v The vector.
 * @return 1.0f / sqrt(x*x + y*y + z*z).
 * @warning Calling this on a null vector (0,0,0) will result in NaN (Not aNumber).
 * @see Magnitude()
 */
inline float InvMagnitude(const VxVector &v) { return 1.0f / Magnitude(v); }

/**
 * @brief Returns a normalized copy of a vector.
 * @param v The vector to normalize.
 * @return A new unit vector pointing in the same direction as v.
 * @warning Returns a vector with NaN components for a null input vector (0,0,0).
 * @remarks This free function may be more precise than the member function `VxVector::Normalize()`.
 * @see VxVector::Normalize(), Magnitude()
 */
inline const VxVector Normalize(const VxVector &v) { return v * InvMagnitude(v); }
inline const VxVector Normalize(const VxVector *vect) { return Normalize(*vect); }

/**
 * @brief Calculates the dot product of two vectors.
 * @param v1 The first vector.
 * @param v2 The second vector.
 * @return The scalar dot product.
 * @see VxVector::Dot()
 */
inline float DotProduct(const VxVector &v1, const VxVector &v2) {
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

/**
 * @brief Calculates the cross product of two vectors.
 * @param Vect1 The first vector.
 * @param Vect2 The second vector.
 * @return A new vector perpendicular to both input vectors (Vect1 x Vect2).
 */
inline const VxVector CrossProduct(const VxVector &Vect1, const VxVector &Vect2) {
    return VxVector(Vect1.y * Vect2.z - Vect1.z * Vect2.y,
                    Vect1.z * Vect2.x - Vect1.x * Vect2.z, Vect1.x * Vect2.y - Vect1.y * Vect2.x);
}

/**
 * @brief Reflects an incident vector about a surface normal.
 * @param v1 The incident vector (should point towards the surface).
 * @param Norm The normal of the reflection plane (should be a unit vector).
 * @return The reflected vector, calculated as R = v1 - 2 * (v1 . N) * N.
 * @remarks For a correct reflection, the incident vector should point towards the surface.
 * @see DotProduct()
 */
inline const VxVector Reflect(const VxVector &v1, const VxVector &Norm) {
    float dp2 = 2.0f * (DotProduct(v1, Norm));
    return VxVector(v1.x - dp2 * Norm.x, v1.y - dp2 * Norm.y, v1.z - dp2 * Norm.z);
}

/// @brief Rotates a vector using a transformation matrix.
VX_EXPORT const VxVector Rotate(const VxMatrix &mat, const VxVector &pt);
/// @brief Rotates a vector around an axis by a specified angle.
VX_EXPORT const VxVector Rotate(const VxVector &v1, const VxVector &v2, float angle);
/// @brief Rotates a vector around an axis by a specified angle (pointer version).
VX_EXPORT const VxVector Rotate(const VxVector *v1, const VxVector *v2, float angle);

///@}

/**
 * @struct VxCompressedVector
 * @brief A memory-efficient representation of a unit vector using two 16-bit polar angles.
 *
 * @remarks
 * This structure reduces the memory usage for a unit vector from 12 bytes (three floats)
 * to 4 bytes (two shorts). It is primarily used for storing normal vectors where precision
 * is less critical than memory footprint.
 *
 * @see VxVector, VxCompressedVectorOld
 */
typedef struct VxCompressedVector {
    short int xa; ///< The polar angle for the x-axis rotation.
    short int ya; ///< The polar angle for the y-axis rotation.

    /// @brief Default constructor. Initializes angles to 0.
    VxCompressedVector() { xa = ya = 0; }
    /// @brief Constructs and sets the compressed vector from three float components.
    VxCompressedVector(float _x, float _y, float _z) { Set(_x, _y, _z); }

    /// @brief Sets the compressed vector from a 3D vector's components.
    void Set(float X, float Y, float Z);

    /// @brief Performs spherical linear interpolation between two compressed vectors.
    void Slerp(float step, VxCompressedVector &v1, VxCompressedVector &v2);

    /// @brief Assigns a VxVector to this compressed vector, performing the conversion.
    VxCompressedVector &operator=(const VxVector &v);
    /// @brief Assigns an old format compressed vector to this one.
    VxCompressedVector &operator=(const VxCompressedVectorOld &v);
} VxCompressedVector;

/**
 * @struct VxCompressedVectorOld
 * @brief A legacy version of VxCompressedVector using 32-bit integers for angles.
 * @deprecated Prefer `VxCompressedVector` for new development.
 */
typedef struct VxCompressedVectorOld {
    int xa; ///< The polar angle for the x-axis rotation.
    int ya; ///< The polar angle for the y-axis rotation.

    /// @brief Default constructor. Initializes angles to 0.
    VxCompressedVectorOld() { xa = ya = 0; }
    /// @brief Constructs and sets the compressed vector from three float components.
    VxCompressedVectorOld(float _x, float _y, float _z) { Set(_x, _y, _z); }

    /// @brief Sets the compressed vector from a 3D vector's components.
    void Set(float X, float Y, float Z);
    /// @brief Performs spherical linear interpolation between two old compressed vectors.
    void Slerp(float step, VxCompressedVectorOld &v1, VxCompressedVectorOld &v2);

    /// @brief Assigns a VxVector to this compressed vector.
    VxCompressedVectorOld &operator=(const VxVector &v);
    /// @brief Assigns a modern compressed vector to this old format one.
    VxCompressedVectorOld &operator=(const VxCompressedVector &v);
} VxCompressedVectorOld;

/**
 * @class VxVector4
 * @brief A 4D vector class, extending VxVector with a 'w' component.
 *
 * @remarks
 * VxVector4 is suitable for homogeneous coordinates used in 3D transformations,
 * particularly with projection matrices where the 'w' component is essential
 * for perspective division.
 *
 * @see VxVector
 */
class VxVector4 : public VxVector {
public:
    float w; ///< The w-component of the vector.

    /// @brief Default constructor. Initializes to (0,0,0,0).
    VxVector4() { x = y = z = w = 0.0f; }
    /// @brief Uniform constructor. Initializes all four components to the same value.
    VxVector4(float f) { x = y = z = w = f; }
    /// @brief Component-wise constructor.
    VxVector4(float _x, float _y, float _z, float _w) {
        x = _x; y = _y; z = _z; w = _w;
    }
    /// @brief Array constructor.
    VxVector4(const float f[4]) {
        x = f[0]; y = f[1]; z = f[2]; w = f[3];
    }

    /**
     * @brief Assignment from a 3D vector. The 'w' component is not modified.
     * @param v The 3D vector to assign from.
     * @return A reference to this modified vector.
     */
    VxVector4 &operator=(const VxVector &v) {
        x = v.x; y = v.y; z = v.z;
        return *this;
    }

    // =====================================
    // Access grants
    // =====================================
    /// @brief Provides const array-style access to components.
    const float &operator[](int i) const;
    /// @brief Provides mutable array-style access to components.
    float &operator[](int i);

    /**
     * @brief Implicit conversion to a float pointer for direct memory access.
     * @return A pointer to the first component (x).
     */
#if !defined(_MSC_VER)
    operator float *() const { return (float *)&x; }
#else
    operator float *() const { return (float *) &v[0]; }
#endif

    // Initialization
    /// @brief Sets all four components of the vector.
    void Set(float X, float Y, float Z, float W);
    /// @brief Sets the x, y, and z components, leaving w unmodified.
    void Set(float X, float Y, float Z);
    /// @brief Computes the 3D dot product (ignoring the w component).
    float Dot(const VxVector4 &iV) const { return x * iV.x + y * iV.y + z * iV.z; }

    /// @brief Component-wise addition and assignment.
    VxVector4 &operator+=(const VxVector4 &v);
    /// @brief Component-wise subtraction and assignment.
    VxVector4 &operator-=(const VxVector4 &v);
    /// @brief Component-wise multiplication and assignment.
    VxVector4 &operator*=(const VxVector4 &v);
    /// @brief Component-wise division and assignment.
    VxVector4 &operator/=(const VxVector4 &v);

    /// @brief Component-wise addition and assignment with a 3D vector.
    VxVector4 &operator+=(const VxVector &v);
    /// @brief Component-wise subtraction and assignment with a 3D vector.
    VxVector4 &operator-=(const VxVector &v);
    /// @brief Component-wise multiplication and assignment with a 3D vector.
    VxVector4 &operator*=(const VxVector &v);
    /// @brief Component-wise division and assignment with a 3D vector.
    VxVector4 &operator/=(const VxVector &v);

    /// @brief Scalar multiplication and assignment.
    VxVector4 &operator*=(float s);
    /// @brief Scalar division and assignment.
    VxVector4 &operator/=(float s);
    /// @brief Adds a scalar to all components.
    VxVector4 operator+(float s) const { return VxVector4(x + s, y + s, z + s, w + s); }
    /// @brief Subtracts a scalar from all components.
    VxVector4 operator-(float s) const { return VxVector4(x - s, y - s, z - s, w - s); }

    // =====================================
    // Unary operators
    // =====================================
    friend const VxVector4 operator+(const VxVector4 &v);
    friend const VxVector4 operator-(const VxVector4 &v);

    // =====================================
    // Binary operators
    // =====================================
    friend const VxVector4 operator+(const VxVector4 &v1, const VxVector4 &v2);
    friend const VxVector4 operator-(const VxVector4 &v1, const VxVector4 &v2);
    friend const VxVector4 operator*(const VxVector4 &v, float s);
    friend const VxVector4 operator*(float s, const VxVector4 &v);
    friend const VxVector4 operator/(const VxVector4 &v, float s);
    friend const VxVector4 operator*(const VxVector4 &v1, const VxVector4 &v2);
    friend const VxVector4 operator/(const VxVector4 &v1, const VxVector4 &v2);

    // Bitwise equality
    friend int operator==(const VxVector4 &v1, const VxVector4 &v2);
    friend int operator!=(const VxVector4 &v1, const VxVector4 &v2);
};

/**
 * @struct VxBbox
 * @brief Represents an axis-aligned bounding box (AABB) in 3D space.
 *
 * @remarks
 * A VxBbox is defined by its minimum and maximum corner points. It provides
 * methods for merging, intersection, containment testing, and other spatial
 * operations commonly used in 3D graphics and physics.
 *
 * @see VxVector
 */
typedef struct VxBbox {
#if !defined(_MSC_VER)
    VxVector Max; ///< The maximum corner of the box (highest x, y, and z coordinates).
    VxVector Min; ///< The minimum corner of the box (lowest x, y, and z coordinates).
#else
    union {
        struct {
            VxVector Max; ///< The maximum corner of the box (highest x, y, and z coordinates).
            VxVector Min; ///< The minimum corner of the box (lowest x, y, and z coordinates).
        };
        float v[6]; ///< Array access to components {Max.x, Max.y, Max.z, Min.x, Min.y, Min.z}.
    };
#endif

    /**
     * @brief Default constructor. Creates an invalid box ready for merging.
     * @remarks The box is initialized with inverted bounds (Min > Max) so that the
     * first call to `Merge` will correctly initialize its extents.
     */
    VxBbox() : Max(-1e6f, -1e6f, -1e6f), Min(1e6f, 1e6f, 1e6f) {}

    /**
     * @brief Constructs a box from minimum and maximum corner points.
     * @param iMin The minimum corner point.
     * @param iMax The maximum corner point.
     */
    VxBbox(VxVector iMin, VxVector iMax) : Max(iMax), Min(iMin) {}

    /**
     * @brief Constructs a box centered at the origin with a given radius.
     * @param value The half-size of the box in all dimensions.
     */
    VxBbox(float value) {
        Max.x = value; Max.y = value; Max.z = value;
        Min.x = -value; Min.y = -value; Min.z = -value;
    }

    /**
     * @brief Checks if the bounding box has valid extents.
     * @return TRUE if Min.x <= Max.x, Min.y <= Max.y, and Min.z <= Max.z.
     */
    XBOOL IsValid() const {
        if (Min.x > Max.x) return FALSE;
        if (Min.y > Max.y) return FALSE;
        if (Min.z > Max.z) return FALSE;
        return TRUE;
    }

    /// @brief Returns the size of the box (width, height, depth).
    VxVector GetSize() const { return Max - Min; }
    /// @brief Returns the half-size of the box (half-width, half-height, half-depth).
    VxVector GetHalfSize() const { return (Max - Min) * 0.5f; }
    /// @brief Returns the center point of the box.
    VxVector GetCenter() const { return (Max + Min) * 0.5f; }

    /// @brief Sets the box's corners from min and max points.
    void SetCorners(const VxVector &min, const VxVector &max) {
        Min = min; Max = max;
    }
    /// @brief Sets the box from a position (min corner) and size.
    void SetDimension(const VxVector &position, const VxVector &size) {
        Min = position; Max = position + size;
    }
    /// @brief Sets the box from its center and half-size.
    void SetCenter(const VxVector &center, const VxVector &halfsize) {
        Min = center - halfsize; Max = center + halfsize;
    }

    /**
     * @brief Resets the box to an invalid state, ready for merging.
     * @remarks This sets Min to a large positive value and Max to a large negative
     * value, ensuring any subsequent `Merge` operation will correctly define the box.
     */
    void Reset() {
        Max.x = -1e6f; Max.y = -1e6f; Max.z = -1e6f;
        Min.x = 1e6f; Min.y = 1e6f; Min.z = 1e6f;
    }

    /**
     * @brief Expands this box to enclose another box.
     * @param v The other VxBbox to merge with this one.
     */
    void Merge(const VxBbox &v) {
        Max.x = XMax(v.Max.x, Max.x); Max.y = XMax(v.Max.y, Max.y); Max.z = XMax(v.Max.z, Max.z);
        Min.x = XMin(v.Min.x, Min.x); Min.y = XMin(v.Min.y, Min.y); Min.z = XMin(v.Min.z, Min.z);
    }

    /**
     * @brief Expands this box to enclose a point.
     * @param v The VxVector point to merge with this one.
     */
    void Merge(const VxVector &v) {
        if (v.x > Max.x) Max.x = v.x;
        if (v.x < Min.x) Min.x = v.x;
        if (v.y > Max.y) Max.y = v.y;
        if (v.y < Min.y) Min.y = v.y;
        if (v.z > Max.z) Max.z = v.z;
        if (v.z < Min.z) Min.z = v.z;
    }

    /**
     * @brief Classifies a point's position relative to the box's planes.
     * @param iPoint The point to classify.
     * @return A bitmask of `VXCLIP_FLAGS` indicating which planes the point is outside of.
     */
    XULONG Classify(const VxVector &iPoint) const {
        XULONG flag = 0;
        if (iPoint.x < Min.x) flag |= VXCLIP_LEFT;
        else if (iPoint.x > Max.x) flag |= VXCLIP_RIGHT;
        if (iPoint.y < Min.y) flag |= VXCLIP_BOTTOM;
        else if (iPoint.y > Max.y) flag |= VXCLIP_TOP;
        if (iPoint.z < Min.z) flag |= VXCLIP_BACK;
        else if (iPoint.z > Max.z) flag |= VXCLIP_FRONT;
        return flag;
    }

    /**
     * @brief Classifies a box's position relative to this box.
     * @param iBox The box to classify.
     * @return A bitmask of `VXCLIP_FLAGS` indicating full exclusion on one or more axes.
     */
    XULONG Classify(const VxBbox &iBox) const {
        XULONG flag = 0;
        if (iBox.Max.z < Min.z) flag |= VXCLIP_BACK;
        else if (iBox.Min.z > Max.z) flag |= VXCLIP_FRONT;
        if (iBox.Max.x < Min.x) flag |= VXCLIP_LEFT;
        else if (iBox.Min.x > Max.x) flag |= VXCLIP_RIGHT;
        if (iBox.Max.y < Min.y) flag |= VXCLIP_BOTTOM;
        else if (iBox.Min.y > Max.y) flag |= VXCLIP_TOP;
        return flag;
    }

    /**
     * @brief Classifies the relative position of two boxes from a viewpoint.
     * @param box2 The other box.
     * @param pt The viewpoint.
     * @return 2 if `box2` is on the opposite side of this box from `pt`, 1 if `box2` is inside, 0 otherwise.
     */
    VX_EXPORT int Classify(const VxBbox &box2, const VxVector &pt) const;

    /**
     * @brief Classifies an array of vertices against the box planes.
     * @param iVcount The number of vertices to classify.
     * @param iVertices Pointer to the start of the vertex data.
     * @param iStride The byte offset between consecutive vertices.
     * @param oFlags An output array to be filled with `VXCLIP_FLAGS` for each vertex.
     */
    VX_EXPORT void ClassifyVertices(const int iVcount, XBYTE *iVertices, XULONG iStride, XULONG *oFlags) const;

    /**
     * @brief Classifies an array of vertices against a single axis of the box.
     * @param iVcount The number of vertices.
     * @param iVertices Pointer to the vertex data.
     * @param iStride The byte offset between vertices.
     * @param iAxis The axis to test against (0=x, 1=y, 2=z).
     * @param oFlags An output array to be filled with flags (0x01 if < min, 0x10 if > max).
     */
    VX_EXPORT void ClassifyVerticesOneAxis(const int iVcount, XBYTE *iVertices, XULONG iStride, const int iAxis, XULONG *oFlags) const;

    /**
     * @brief Calculates the intersection of this box with another box.
     * @param v The VxBbox to intersect with this one.
     * @remarks The result modifies this box to become the intersection volume.
     */
    void Intersect(const VxBbox &v) {
        Max.x = XMin(v.Max.x, Max.x); Max.y = XMin(v.Max.y, Max.y); Max.z = XMin(v.Max.z, Max.z);
        Min.x = XMax(v.Min.x, Min.x); Min.y = XMax(v.Min.y, Min.y); Min.z = XMax(v.Min.z, Min.z);
    }

    /**
     * @brief Tests if a point is inside the box.
     * @param v The point to test.
     * @return TRUE if the point is inside or on the boundary of this box, FALSE otherwise.
     */
    XBOOL VectorIn(const VxVector &v) const {
        if (v.x < Min.x) return FALSE;
        if (v.x > Max.x) return FALSE;
        if (v.y < Min.y) return FALSE;
        if (v.y > Max.y) return FALSE;
        if (v.z < Min.z) return FALSE;
        if (v.z > Max.z) return FALSE;
        return TRUE;
    }

    /**
     * @brief Tests if another box is totally inside this box.
     * @param b The box to test.
     * @return TRUE if box b is completely contained within this box, FALSE otherwise.
     */
    XBOOL IsBoxInside(const VxBbox &b) const {
        if (b.Min.x < Min.x) return 0;
        if (b.Min.y < Min.y) return 0;
        if (b.Min.z < Min.z) return 0;
        if (b.Max.x > Max.x) return 0;
        if (b.Max.y > Max.y) return 0;
        if (b.Max.z > Max.z) return 0;
        return 1;
    }

    /// @brief Equality operator.
    bool operator==(const VxBbox &iBox) const {
        return (Max == iBox.Max) && (Min == iBox.Min);
    }

    /**
     * @brief Transforms the eight corner points of this box by a matrix.
     * @param pts A pointer to an array to store the eight resulting points.
     * @param Mat The transformation matrix.
     */
    VX_EXPORT void TransformTo(VxVector *pts, const VxMatrix &Mat) const;

    /**
     * @brief Creates this box by transforming another box by a matrix.
     * @param sbox The source box to transform.
     * @param Mat The transformation matrix.
     */
    VX_EXPORT void TransformFrom(const VxBbox &sbox, const VxMatrix &Mat);
} VxBbox;

// =================================================================
// VxVector Inline Implementations
// =================================================================

inline VxVector::VxVector() : x(0), y(0), z(0) {}
inline VxVector::VxVector(float f) : x(f), y(f), z(f) {}
inline VxVector::VxVector(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}
inline VxVector::VxVector(const float f[3]) : x(f[0]), y(f[1]), z(f[2]) {}
inline void VxVector::Set(float _x, float _y, float _z) { x = _x; y = _y; z = _z; }
inline const float &VxVector::operator[](int i) const { return *((&x) + i); }
inline float &VxVector::operator[](int i) { return *((&x) + i); }
inline VxVector &VxVector::operator+=(const VxVector &v) { x += v.x; y += v.y; z += v.z; return *this; }
inline VxVector &VxVector::operator-=(const VxVector &v) { x -= v.x; y -= v.y; z -= v.z; return *this; }
inline VxVector &VxVector::operator*=(const VxVector &v) { x *= v.x; y *= v.y; z *= v.z; return *this; }
inline VxVector &VxVector::operator/=(const VxVector &v) { x /= v.x; y /= v.y; z /= v.z; return *this; }
inline VxVector &VxVector::operator*=(float s) { x *= s; y *= s; z *= s; return *this; }
inline VxVector &VxVector::operator/=(float s) { float temp = 1.0f / s; x *= temp; y *= temp; z *= temp; return *this; }

inline const VxVector operator+(const VxVector &v) { return v; }
inline const VxVector operator-(const VxVector &v) { return VxVector(-v.x, -v.y, -v.z); }
inline const VxVector operator+(const VxVector &v1, const VxVector &v2) { return VxVector(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z); }
inline const VxVector operator-(const VxVector &v1, const VxVector &v2) { return VxVector(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z); }
inline const VxVector operator*(const VxVector &v1, const VxVector &v2) { return VxVector(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z); }
inline const VxVector operator/(const VxVector &v1, const VxVector &v2) { return VxVector(v1.x / v2.x, v1.y / v2.y, v1.z / v2.z); }
inline int operator<(const VxVector &v1, const VxVector &v2) { return v1[0] < v2[0] && v1[1] < v2[1] && v1[2] < v2[2]; }
inline int operator<=(const VxVector &v1, const VxVector &v2) { return v1[0] <= v2[0] && v1[1] <= v2[1] && v1[2] <= v2[2]; }
inline const VxVector operator*(const VxVector &v, float s) { return VxVector(s * v.x, s * v.y, s * v.z); }
inline const VxVector operator*(float s, const VxVector &v) { return VxVector(s * v.x, s * v.y, s * v.z); }
inline const VxVector operator/(const VxVector &v, float s) { float temp = 1.0f / s; return VxVector(v.x * temp, v.y * temp, v.z * temp); }
inline int operator==(const VxVector &v1, const VxVector &v2) { return ((v1.x == v2.x) && (v1.y == v2.y) && (v1.z == v2.z)); }
inline int operator!=(const VxVector &v1, const VxVector &v2) { return !(v1 == v2); }

/**
 * @brief Returns a vector with each component set to its absolute value.
 * @param v The input vector.
 * @return A new vector with absolute values of the components of v.
 */
inline const VxVector Absolute(const VxVector &v) { return VxVector(XAbs(v.x), XAbs(v.y), XAbs(v.z)); }
inline void VxVector::Absolute() { x = XAbs(x); y = XAbs(y); z = XAbs(z); }

/**
 * @brief Calculates the minimum value among the components of a vector.
 * @param v The vector.
 * @return The minimum component value of v.
 */
inline float Min(const VxVector &v) { return XMin(v.x, v.y, v.z); }

/**
 * @brief Calculates the maximum value among the components of a vector.
 * @param v The vector.
 * @return The maximum component value of v.
 */
inline float Max(const VxVector &v) {
    float ret = v.x;
    if (ret < v.y) ret = v.y;
    if (ret < v.z) ret = v.z;
    return ret;
}

/**
 * @brief Constructs a vector from the component-wise minimum of two vectors.
 * @param v1 The first vector.
 * @param v2 The second vector.
 * @return A vector where each component is the minimum of the corresponding components of v1 and v2.
 */
inline const VxVector Minimize(const VxVector &v1, const VxVector &v2) {
    return VxVector(XMin(v1[0], v2[0]), XMin(v1[1], v2[1]), XMin(v1[2], v2[2]));
}

/**
 * @brief Constructs a vector from the component-wise maximum of two vectors.
 * @param v1 The first vector.
 * @param v2 The second vector.
 * @return A vector where each component is the maximum of the corresponding components of v1 and v2.
 */
inline const VxVector Maximize(const VxVector &v1, const VxVector &v2) {
    return VxVector(XMax(v1[0], v2[0]), XMax(v1[1], v2[1]), XMax(v1[2], v2[2]));
}

/**
 * @brief Performs a linear interpolation between two vectors.
 * @param step The interpolation factor, clamped between 0.0 (v1) and 1.0 (v2).
 * @param v1 The starting vector.
 * @param v2 The ending vector.
 * @return The interpolated vector.
 */
inline const VxVector Interpolate(float step, const VxVector &v1, const VxVector &v2) {
    return VxVector(v1.x + (v2.x - v1.x) * step,
                    v1.y + (v2.y - v1.y) * step,
                    v1.z + (v2.z - v1.z) * step);
}

// =================================================================
// VxVector4 Inline Implementations
// =================================================================

inline VxVector4 &VxVector4::operator+=(const VxVector4 &v) { x += v.x; y += v.y; z += v.z; w += v.w; return *this; }
inline VxVector4 &VxVector4::operator-=(const VxVector4 &v) { x -= v.x; y -= v.y; z -= v.z; w -= v.w; return *this; }
inline VxVector4 &VxVector4::operator*=(const VxVector4 &v) { x *= v.x; y *= v.y; z *= v.z; w *= v.w; return *this; }
inline VxVector4 &VxVector4::operator/=(const VxVector4 &v) { x /= v.x; y /= v.y; z /= v.z; w /= v.w; return *this; }
inline VxVector4 &VxVector4::operator+=(const VxVector &v) { x += v.x; y += v.y; z += v.z; return *this; }
inline VxVector4 &VxVector4::operator-=(const VxVector &v) { x -= v.x; y -= v.y; z -= v.z; return *this; }
inline VxVector4 &VxVector4::operator*=(const VxVector &v) { x *= v.x; y *= v.y; z *= v.z; return *this; }
inline VxVector4 &VxVector4::operator/=(const VxVector &v) { x /= v.x; y /= v.y; z /= v.z; return *this; }
inline VxVector4 &VxVector4::operator*=(float s) { x *= s; y *= s; z *= s; w *= s; return *this; }
inline VxVector4 &VxVector4::operator/=(float s) { float temp = 1.0f / s; x *= temp; y *= temp; z *= temp; w *= temp; return *this; }
inline const VxVector4 operator+(const VxVector4 &v) { return v; }
inline const VxVector4 operator-(const VxVector4 &v) { return VxVector4(-v.x, -v.y, -v.z, -v.w); }
inline const VxVector4 operator+(const VxVector4 &v1, const VxVector4 &v2) { return VxVector4(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z, v1.w + v2.w); }
inline const VxVector4 operator-(const VxVector4 &v1, const VxVector4 &v2) { return VxVector4(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z, v1.w - v2.w); }
inline const VxVector4 operator*(const VxVector4 &v1, const VxVector4 &v2) { return VxVector4(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z, v1.w * v2.w); }
inline const VxVector4 operator/(const VxVector4 &v1, const VxVector4 &v2) { return VxVector4(v1.x / v2.x, v1.y / v2.y, v1.z / v2.z, v1.w / v2.w); }
inline const VxVector4 operator*(const VxVector4 &v, float s) { return VxVector4(s * v.x, s * v.y, s * v.z, s * v.w); }
inline const VxVector4 operator*(float s, const VxVector4 &v) { return VxVector4(s * v.x, s * v.y, s * v.z, s * v.w); }
inline const VxVector4 operator/(const VxVector4 &v, float s) { float invs = 1.0f / s; return VxVector4(invs * v.x, invs * v.y, invs * v.z, invs * v.w); }
inline int operator==(const VxVector4 &v1, const VxVector4 &v2) { return ((v1.x == v2.x) && (v1.y == v2.y) && (v1.z == v2.z) && (v1.w == v2.w)); }
inline int operator!=(const VxVector4 &v1, const VxVector4 &v2) { return !(v1 == v2); }
inline void VxVector4::Set(float _x, float _y, float _z, float _w) { x = _x; y = _y; z = _z; w = _w; }
inline void VxVector4::Set(float _x, float _y, float _z) { x = _x; y = _y; z = _z; }
inline const float &VxVector4::operator[](int i) const { return *((&x) + i); }
inline float &VxVector4::operator[](int i) { return *((&x) + i); }

// =================================================================
// VxCompressedVector Inline Implementations
// =================================================================

/**
 * @brief Performs spherical linear interpolation between two compressed vectors.
 * @param step The interpolation factor (0 for v1, 1 for v2).
 * @param v1 The starting vector.
 * @param v2 The ending vector.
 */
inline void VxCompressedVectorOld::Slerp(float step, VxCompressedVectorOld &v1, VxCompressedVectorOld &v2) {
    int v1y = ((int) v1.ya + 16384) & 16383;
    int v2y = ((int) v2.ya + 16384) & 16383;
    v2y = (v2y - v1y);
    if (v2y > 8192) v2y = 16384 - v2y;
    else if (v2y < -8192) v2y = 16384 + v2y;
    xa = (int) ((float) v1.xa + (float) (v2.xa - v1.xa) * step);
    ya = (int) ((float) v1y + (float) v2y * step);
}

inline VxCompressedVectorOld &VxCompressedVectorOld::operator=(const VxVector &v) {
    Set(v.x, v.y, v.z);
    return *this;
}

/**
 * @brief Sets the compressed vector from 3 float components.
 * @param X The x-component.
 * @param Y The y-component.
 * @param Z The z-component.
 */
inline void VxCompressedVectorOld::Set(float X, float Y, float Z) {
    xa = -radToAngle((float) asin(Y));
    ya = radToAngle((float) atan2(X, Z));
}

inline void VxCompressedVector::Slerp(float step, VxCompressedVector &v1, VxCompressedVector &v2) {
    int coef = (int) (65536.0f * step);
    int v1y = ((int) v1.ya + 16384) & 16383;
    int v2y = ((int) v2.ya + 16384) & 16383;
    v2y = (v2y - v1y);
    if (v2y > 8192) v2y = 16384 - v2y;
    else if (v2y < -8192) v2y = 16384 + v2y;
    xa = (short int) ((int) v1.xa + (((v2.xa - v1.xa) * coef) >> 16));
    ya = (short int) (v1y + ((v2y * coef) >> 16));
}

inline VxCompressedVector &VxCompressedVector::operator=(const VxVector &v) {
    Set(v.x, v.y, v.z);
    return *this;
}

inline void VxCompressedVector::Set(float X, float Y, float Z) {
    xa = (short int) -radToAngle((float) asin(Y));
    ya = (short int) radToAngle((float) atan2(X, Z));
}

inline VxCompressedVectorOld &VxCompressedVectorOld::operator=(const VxCompressedVector &v) {
    xa = (int) v.xa;
    ya = (int) v.ya;
    return *this;
}

inline VxCompressedVector &VxCompressedVector::operator=(const VxCompressedVectorOld &v) {
    xa = (short int) v.xa;
    ya = (short int) v.ya;
    return *this;
}

#endif // VXVECTOR_H
