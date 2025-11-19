#ifndef VX2DVECTOR_H
#define VX2DVECTOR_H

#include "XUtil.h"

/**
 * @brief Represents a vector in 2-dimensional space.
 *
 * @remarks
 * A Vx2DVector is defined by two float components, x and y.
 * It can be accessed as a struct with x and y members, or as a float array.
 * @code
 * struct Vx2DVector {
 *     union {
 *         struct {
 *             float x, y;
 *         };
 *         float v[2];
 *     };
 * };
 * @endcode
 */
struct Vx2DVector {
    union {
        struct {
            float x, y;
        };
        float v[2];
    };

    /**
     * @brief Calculates the length (magnitude) of the vector.
     * @return The length of the vector.
     * @warning Calling this function with a null vector (0,0) will result in NaN (Not a Number).
     * @see Normalize, SquareMagnitude
     */
    float Magnitude() const {
        return sqrtf(SquareMagnitude());
    }

    /**
     * @brief Calculates the squared length of the vector.
     * @return The squared length of the vector.
     * @remarks This method is more efficient than Magnitude() as it avoids a square root operation.
     * It is useful for comparing vector lengths.
     * @see Normalize, Magnitude
     */
    float SquareMagnitude() const {
        return (x * x + y * y);
    }

    /**
     * @brief Normalizes the vector to a unit length of 1.0.
     * @return A reference to the modified vector.
     * @remarks This method modifies the vector in place.
     * @warning Calling this function on a null vector (0,0) will result in a vector with NaN components.
     * @see Magnitude
     */
    Vx2DVector &Normalize() {
        float m = Magnitude();
        if (m != 0.00000f) {
            m = 1.0f / m;
            x *= m;
            y *= m;
        }
        return *this;
    }

    /**
     * @brief Sets the components of the vector.
     * @param iX The new x-component.
     * @param iY The new y-component.
     */
    void Set(float iX, float iY) {
        x = iX;
        y = iY;
    }

    /**
     * @brief Sets the components of the vector from integer values.
     * @param iX The new x-component.
     * @param iY The new y-component.
     */
    void Set(int iX, int iY) {
        x = (float) iX;
        y = (float) iY;
    }

    /**
     * @brief Calculates the dot product of this vector with another.
     * @param iV The other vector for the dot product calculation.
     * @return The result of the dot product (this . iV).
     * @remarks To calculate a distance from a point to a line using a dot product, the vector representing the line's direction should be normalized.
     * @see Normalize
     */
    float Dot(const Vx2DVector &iV) const {
        return x * iV.x + y * iV.y;
    }

    /**
     * @brief Calculates the perpendicular vector (2D cross product equivalent).
     * @return A new vector that is perpendicular to the original vector.
     * @see Normalize
     */
    Vx2DVector Cross() const {
        return Vx2DVector(-y, x);
    }

    // =====================================
    // Access grants
    // =====================================

    /**
     * @brief Provides access to the vector's components by index (const version).
     * @param i The index of the component (0 for x, 1 for y).
     * @return A const reference to the specified component.
     */
    const float &operator[](int i) const {
        return v[i];
    }

    /**
     * @brief Provides access to the vector's components by index.
     * @param i The index of the component (0 for x, 1 for y).
     * @return A reference to the specified component.
     */
    float &operator[](int i) {
        return v[i];
    }

    // =====================================
    // Assignment operators
    // =====================================

    /// @brief Adds another vector component-wise and assigns the result to this vector.
    Vx2DVector &operator+=(const Vx2DVector &v) {
        x += v.x;
        y += v.y;
        return *this;
    }

    /// @brief Subtracts another vector component-wise and assigns the result to this vector.
    Vx2DVector &operator-=(const Vx2DVector &v) {
        x -= v.x;
        y -= v.y;
        return *this;
    }

    /// @brief Multiplies by another vector component-wise and assigns the result to this vector.
    Vx2DVector &operator*=(const Vx2DVector &v) {
        x *= v.x;
        y *= v.y;
        return *this;
    }

    /// @brief Divides by another vector component-wise and assigns the result to this vector.
    Vx2DVector &operator/=(const Vx2DVector &v) {
        x /= v.x;
        y /= v.y;
        return *this;
    }

    /// @brief Multiplies the vector by a scalar and assigns the result to this vector.
    Vx2DVector &operator*=(float s) {
        x *= s;
        y *= s;
        return *this;
    }

    /// @brief Divides the vector by a scalar and assigns the result to this vector.
    Vx2DVector &operator/=(float s) {
        s = 1.0f / s;
        x *= s;
        y *= s;
        return *this;
    }

    // =====================================
    // Unary operators
    // =====================================

    /// @brief Unary plus operator, returns a copy of the vector.
    Vx2DVector operator+() const {
        return *this;
    }

    /// @brief Unary minus operator, returns the negation of the vector.
    Vx2DVector operator-() const {
        return Vx2DVector(-x, -y);
    }

    // =====================================
    // Binary operators
    // =====================================

    /// @brief Adds two vectors component-wise.
    Vx2DVector operator+(const Vx2DVector &v) const {
        return Vx2DVector(x + v.x, y + v.y);
    }

    /// @brief Subtracts two vectors component-wise.
    Vx2DVector operator-(const Vx2DVector &v) const {
        return Vx2DVector(x - v.x, y - v.y);
    }

    /// @brief Multiplies two vectors component-wise.
    Vx2DVector operator*(const Vx2DVector &v) const {
        return Vx2DVector(x * v.x, y * v.y);
    }

    /// @brief Divides two vectors component-wise.
    Vx2DVector operator/(const Vx2DVector &v) const {
        return Vx2DVector(x / v.x, y / v.y);
    }

    /// @brief Multiplies the vector by a scalar.
    Vx2DVector operator*(float s) const {
        return Vx2DVector(x * s, y * s);
    }

    /// @brief Divides the vector by a scalar.
    Vx2DVector operator/(float s) const {
        s = 1.0f / s;
        return Vx2DVector(x * s, y * s);
    }

    /// @brief Friend operator for scalar multiplication (s * v).
    friend Vx2DVector operator*(float s, const Vx2DVector &v);

    /// @brief Friend operator for scalar division (s / v).
    friend Vx2DVector operator/(float s, const Vx2DVector &v);

    /// @brief Performs a component-wise less-than comparison.
    bool operator<(const Vx2DVector &v) const {
        return (x < v.x) && (y < v.y);
    }

    /// @brief Performs a component-wise less-than-or-equal-to comparison.
    bool operator<=(const Vx2DVector &v) const {
        return (x <= v.x) && (y <= v.y);
    }

    /// @brief Checks for bitwise equality between two vectors.
    bool operator==(const Vx2DVector &v) const {
        return (x == v.x) && (y == v.y);
    }

    /// @brief Checks for bitwise inequality between two vectors.
    bool operator!=(const Vx2DVector &v) const {
        return (x != v.x) || (y != v.y);
    }

    /**
     * @brief Returns the minimum of the two components of the vector.
     * @return The minimum component value.
     * @see Max
     */
    float Min() const {
        return XMin(x, y);
    }

    /**
     * @brief Returns the maximum of the two components of the vector.
     * @return The maximum component value.
     * @see Min
     */
    float Max() const {
        return XMax(x, y);
    }

    // =====================================
    // Constructors
    // =====================================

    /// @brief Default constructor, initializes the vector to (0.0, 0.0).
    Vx2DVector() : x(0.0f), y(0.0f) {}

    /// @brief Constructs a vector with both components set to a single float value.
    Vx2DVector(float f) : x(f), y(f) {}

    /// @brief Constructs a vector from two float components.
    Vx2DVector(float iX, float iY) : x(iX), y(iY) {}

    /// @brief Constructs a vector from two integer components.
    Vx2DVector(int iX, int iY) : x((float) iX), y((float) iY) {}

    /// @brief Constructs a vector from an array of two floats.
    Vx2DVector(const float f[2]) : x(f[0]), y(f[1]) {}
};

/**
 * @brief Free-standing operator for scalar multiplication (s * v).
 * @param s The scalar value.
 * @param v The vector to multiply.
 * @return The resulting vector.
 */
inline Vx2DVector operator*(float s, const Vx2DVector &v) {
    return Vx2DVector(s * v.x, s * v.y);
}

/**
 * @brief Free-standing operator for scalar division (s / v).
 * @param s The scalar numerator.
 * @param v The vector denominator (component-wise).
 * @return The resulting vector.
 */
inline Vx2DVector operator/(float s, const Vx2DVector &v) {
    return Vx2DVector(s / v.x, s / v.y);
}

#endif // VX2DVECTOR_H
