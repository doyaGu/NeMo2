#ifndef VXRECT_H
#define VXRECT_H

#include "VxMathDefines.h"
#include "Vx2dVector.h"

/**
 * @enum VXRECT_INTERSECTION
 * @brief Describes the result of an intersection test between two rectangles.
 */
typedef enum VXRECT_INTERSECTION {
    ALLOUTSIDE = 0, ///< The rectangle is completely outside the clipping rectangle.
    ALLINSIDE  = 1, ///< The rectangle is completely inside the clipping rectangle.
    PARTINSIDE = 2  ///< The rectangle is partially inside (intersecting) the clipping rectangle.
} VXRECT_INTERSECTION;

/**
 * @brief Represents a 2D rectangle defined by four float values.
 *
 * @remarks
 * A VxRect is defined by its left, top, right, and bottom coordinates.
 * It is used to represent a 2D region.
 *
 * The class is defined as:
 * @code
 * class VxRect {
 * public:
 *     float left;
 *     float top;
 *     float right;
 *     float bottom;
 * };
 * @endcode
 *
 * For convenience, especially under MSVC, a union allows accessing the corners
 * as Vx2DVector objects (`m_TopLeft`, `m_BottomRight`). Elements can be accessed
 * directly or through various accessor methods for more sophisticated operations.
 */
class VxRect {
public:
    // Members
#if !defined(_MSC_VER)
    float left;   ///< The x-coordinate of the left edge.
    float top;    ///< The y-coordinate of the top edge.
    float right;  ///< The x-coordinate of the right edge.
    float bottom; ///< The y-coordinate of the bottom edge.
#else
    union {
        struct {
            float left;   ///< The x-coordinate of the left edge.
            float top;    ///< The y-coordinate of the top edge.
            float right;  ///< The x-coordinate of the right edge.
            float bottom; ///< The y-coordinate of the bottom edge.
        };
        struct {
            Vx2DVector m_TopLeft;       ///< The top-left corner as a 2D vector.
            Vx2DVector m_BottomRight;   ///< The bottom-right corner as a 2D vector.
        };
    };
#endif

    // Methods
    /// @brief Default constructor. Does not initialize members.
    VxRect() {}

#if !defined(_MSC_VER)
    /**
     * @brief Constructs a rectangle from two corner vectors.
     * @param topleft A vector representing the top-left corner.
     * @param bottomright A vector representing the bottom-right corner.
     */
    VxRect(Vx2DVector &topleft, Vx2DVector &bottomright) : left(topleft.x), top(topleft.y), right(bottomright.x), bottom(bottomright.y) {}
    /**
     * @brief Constructs a rectangle from four float values.
     * @param l The left coordinate.
     * @param t The top coordinate.
     * @param r The right coordinate.
     * @param b The bottom coordinate.
     */
    VxRect(float l, float t, float r, float b) : left(l), top(t), right(r), bottom(b) {}
#else
    /**
     * @brief Constructs a rectangle from two corner vectors.
     * @param topleft A vector representing the top-left corner.
     * @param bottomright A vector representing the bottom-right corner.
     */
    VxRect(Vx2DVector &topleft, Vx2DVector &bottomright) : m_TopLeft(topleft), m_BottomRight(bottomright) {}
    /**
     * @brief Constructs a rectangle from four float values.
     * @param l The left coordinate.
     * @param t The top coordinate.
     * @param r The right coordinate.
     * @param b The bottom coordinate.
     */
    VxRect(float l, float t, float r, float b) : m_TopLeft(l, t), m_BottomRight(r, b) {}
#endif

    /**
     * @brief Sets the width of the rectangle, adjusting the right coordinate.
     * @param w The new width.
     */
    void SetWidth(float w) { right = left + w; };

    /**
     * @brief Returns the width of the rectangle.
     * @return The width.
     */
    float GetWidth() const { return right - left; }

    /**
     * @brief Sets the height of the rectangle, adjusting the bottom coordinate.
     * @param h The new height.
     */
    void SetHeight(float h) { bottom = top + h; }

    /**
     * @brief Returns the height of the rectangle.
     * @return The height.
     */
    float GetHeight() const { return bottom - top; }

    /**
     * @brief Returns the horizontal center of the rectangle.
     * @return The x-coordinate of the center.
     */
    float GetHCenter() const { return left + 0.5f * GetWidth(); }

    /**
     * @brief Returns the vertical center of the rectangle.
     * @return The y-coordinate of the center.
     */
    float GetVCenter() const { return top + 0.5f * GetHeight(); }

    /**
     * @brief Sets the size of the rectangle.
     * @param v A Vx2DVector where x is the new width and y is the new height.
     */
    void SetSize(const Vx2DVector &v) {
        SetWidth(v.x);
        SetHeight(v.y);
    }

    /**
     * @brief Returns the size of the rectangle.
     * @return A Vx2DVector containing the width and height.
     */
    Vx2DVector GetSize() const { return Vx2DVector(GetWidth(), GetHeight()); }

    /**
     * @brief Sets the rectangle's dimensions based on a center point and half-size vector.
     * @param v The half-size vector (half-width, half-height).
     */
    void SetHalfSize(const Vx2DVector &v) {
        Vx2DVector c = GetCenter();
        SetCenter(c, v);
    }

    /**
     * @brief Gets the half-size of the rectangle.
     * @return A Vx2DVector containing half the width and half the height.
     */
    Vx2DVector GetHalfSize() const { return Vx2DVector(0.5f * GetWidth(), 0.5f * GetHeight()); }

    /**
     * @brief Moves the rectangle to a new center position while preserving its size.
     * @param v The new center point.
     */
    void SetCenter(const Vx2DVector &v) {
        Vx2DVector hs = GetHalfSize();
        SetCenter(v, hs);
    }

    /**
     * @brief Gets the center point of the rectangle.
     * @return A Vx2DVector representing the center point.
     */
    Vx2DVector GetCenter() const { return Vx2DVector(GetHCenter(), GetVCenter()); }

    /**
     * @brief Sets the top-left corner of the rectangle.
     * @param v The new top-left coordinates.
     */
    void SetTopLeft(const Vx2DVector &v) {
        left = v.x;
        top = v.y;
    }

#if defined(_MSC_VER)
    /// @brief Gets the top-left corner as a const vector reference.
    const Vx2DVector &GetTopLeft() const { return m_TopLeft; }
    /// @brief Gets the top-left corner as a vector reference.
    Vx2DVector &GetTopLeft() { return m_TopLeft; }
#else
    /// @brief Gets the top-left corner as a const vector reference.
    const Vx2DVector &GetTopLeft() const { return (const Vx2DVector &)*(Vx2DVector *)&left; }
    /// @brief Gets the top-left corner as a vector reference.
    Vx2DVector &GetTopLeft() { return (Vx2DVector &)*(Vx2DVector *)&left; }
#endif

    /**
     * @brief Sets the bottom-right corner of the rectangle.
     * @param v The new bottom-right coordinates.
     */
    void SetBottomRight(const Vx2DVector &v) {
        right = v.x;
        bottom = v.y;
    }
#if defined(_MSC_VER)
    /// @brief Gets the bottom-right corner as a const vector reference.
    const Vx2DVector &GetBottomRight() const { return m_BottomRight; }
    /// @brief Gets the bottom-right corner as a vector reference.
    Vx2DVector &GetBottomRight() { return m_BottomRight; }
#else
    /// @brief Gets the bottom-right corner as a const vector reference.
    const Vx2DVector &GetBottomRight() const { return (const Vx2DVector &)*(Vx2DVector *)&right; }
    /// @brief Gets the bottom-right corner as a vector reference.
    Vx2DVector &GetBottomRight() { return (Vx2DVector &)*(Vx2DVector *)&right; }
#endif

    /**
     * @brief Resets the rectangle to a null state (all coordinates set to 0).
     */
    void Clear() { SetCorners(0, 0, 0, 0); }

    /**
     * @brief Creates a rectangle from two corner points.
     * @param topleft A Vx2DVector for the top-left corner.
     * @param bottomright A Vx2DVector for the bottom-right corner.
     * @see SetDimension, SetCenter
     */
    void SetCorners(const Vx2DVector &topleft, const Vx2DVector &bottomright) {
        left = topleft.x;
        top = topleft.y;
        right = bottomright.x;
        bottom = bottomright.y;
    }

    /**
     * @brief Creates a rectangle from four coordinate values.
     * @param l Left coordinate.
     * @param t Top coordinate.
     * @param r Right coordinate.
     * @param b Bottom coordinate.
     * @see SetDimension, SetCenter
     */
    void SetCorners(float l, float t, float r, float b) {
        left = l;
        top = t;
        right = r;
        bottom = b;
    }

    /**
     * @brief Creates a rectangle from a top-left position and a size.
     * @param position A Vx2DVector for the top-left corner.
     * @param size A Vx2DVector for the width and height.
     * @see SetCorners, SetCenter
     */
    void SetDimension(const Vx2DVector &position, const Vx2DVector &size) {
        left = position.x;
        top = position.y;
        right = left + size.x;
        bottom = top + size.y;
    }

    /**
     * @brief Creates a rectangle from a top-left position and a size.
     * @param x The left coordinate.
     * @param y The top coordinate.
     * @param w The width.
     * @param h The height.
     * @see SetCorners, SetCenter
     */
    void SetDimension(float x, float y, float w, float h) {
        left = x;
        top = y;
        right = x + w;
        bottom = y + h;
    }

    /**
     * @brief Creates a rectangle from its center point and half-size.
     * @param center A Vx2DVector for the center position.
     * @param halfsize A Vx2DVector for the half-width and half-height.
     * @see SetCorners, SetDimension
     */
    void SetCenter(const Vx2DVector &center, const Vx2DVector &halfsize) {
        left = center.x - halfsize.x;
        top = center.y - halfsize.y;
        right = center.x + halfsize.x;
        bottom = center.y + halfsize.y;
    }

    /**
     * @brief Creates a rectangle from its center point and half-size.
     * @param cx The horizontal center position.
     * @param cy The vertical center position.
     * @param hw The half-width.
     * @param hh The half-height.
     * @see SetCorners, SetDimension
     */
    void SetCenter(float cx, float cy, float hw, float hh) {
        left = cx - hw;
        top = cy - hh;
        right = cx + hw;
        bottom = cy + hh;
    }

    /**
     * @brief Copies the coordinates from a CKRECT (integer-based rectangle).
     * @param iRect The source CKRECT.
     */
    void CopyFrom(const CKRECT &iRect) {
        left = (float) iRect.left;
        top = (float) iRect.top;
        right = (float) iRect.right;
        bottom = (float) iRect.bottom;
    }

    /**
     * @brief Copies the coordinates to a CKRECT, truncating to integers.
     * @param oRect Pointer to the destination CKRECT.
     */
    void CopyTo(CKRECT *oRect) const {
        XASSERT(oRect);
        oRect->left = (int) left;
        oRect->top = (int) top;
        oRect->right = (int) right;
        oRect->bottom = (int) bottom;
    }

    /**
     * @brief Creates the smallest rectangle that encloses two given points.
     * @param p1 The first point.
     * @param p2 The second point.
     * @see SetCorners, SetDimension
     */
    void Bounding(const Vx2DVector &p1, const Vx2DVector &p2) {
        if (p1.x < p2.x) { left = p1.x; right = p2.x; } else { left = p2.x; right = p1.x; }
        if (p1.y < p2.y) { top = p1.y; bottom = p2.y; } else { top = p2.y; bottom = p1.y; }
    }

    /**
     * @brief Ensures the rectangle is valid by making `left` <= `right` and `top` <= `bottom`.
     * @remarks Use this function when creating a rectangle from two corners without being sure
     * that the top-left and bottom-right corners were provided in the correct order.
     * @see SetCorners, SetDimension
     */
    void Normalize() {
        if (left > right) XSwap(right, left);
        if (top > bottom) XSwap(top, bottom);
    }

    /**
     * @brief Moves the rectangle's top-left corner to a new position.
     * @param pos The new position for the top-left corner.
     * @see Translate
     */
    void Move(const Vx2DVector &pos) {
        right += (pos.x - left);
        bottom += (pos.y - top);
        left = pos.x;
        top = pos.y;
    }

    /**
     * @brief Translates the rectangle by a given offset.
     * @param t A Vx2DVector representing the translation offset.
     * @see Move
     */
    void Translate(const Vx2DVector &t) {
        left += t.x; right += t.x;
        top += t.y; bottom += t.y;
    }

    /// @brief Moves the left edge to a new horizontal position, preserving width.
    void HMove(float h) {
        right += (h - left);
        left = h;
    }

    /// @brief Moves the top edge to a new vertical position, preserving height.
    void VMove(float v) {
        bottom += (v - top);
        top = v;
    }

    /// @brief Translates the rectangle horizontally.
    void HTranslate(float h) {
        right += h;
        left += h;
    }

    /// @brief Translates the rectangle vertically.
    void VTranslate(float v) {
        bottom += v;
        top += v;
    }

    /**
     * @brief Transforms a point from homogeneous coordinates [0,1] to screen coordinates within this rectangle.
     * @param dest The destination vector (in screen coordinates).
     * @param srchom The source vector (in homogeneous coordinates).
     * @remarks The implementation for the y-coordinate `dest.y = left + GetHeight() * srchom.y` appears to contain a typo and likely should be `top + GetHeight() * srchom.y`.
     */
    void TransformFromHomogeneous(Vx2DVector &dest, const Vx2DVector &srchom) const {
        dest.x = left + GetWidth() * srchom.x;
        dest.y = left + GetHeight() * srchom.y;
    }

    /**
     * @brief Scales the rectangle's size relative to its top-left corner.
     * @param s A Vx2DVector containing the scaling factors for width and height.
     * @see Inflate
     */
    void Scale(const Vx2DVector &s) {
        SetWidth(s.x * GetWidth());
        SetHeight(s.y * GetHeight());
    }

    /**
     * @brief Inflates or deflates the rectangle by an offset, moving each edge outwards or inwards.
     * @param pt A Vx2DVector where x is the horizontal inflation amount and y is the vertical inflation amount.
     * @see Scale
     */
    void Inflate(const Vx2DVector &pt) {
        left -= pt.x; right += pt.x;
        top -= pt.y; bottom += pt.y;
    }

    /**
     * @brief Interpolates this rectangle towards another rectangle.
     * @param value The interpolation factor (0.0 = this rectangle, 1.0 = rectangle a).
     * @param a The target rectangle for interpolation.
     */
    void Interpolate(float value, const VxRect &a) {
        left += (a.left - left) * value;
        right += (a.right - right) * value;
        top += (a.top - top) * value;
        bottom += (a.bottom - bottom) * value;
    }

    /**
     * @brief Expands this rectangle to encompass another rectangle.
     * @param a The rectangle to merge with this one.
     */
    void Merge(const VxRect &a) {
        if (a.left < left) left = a.left;
        if (a.right > right) right = a.right;
        if (a.top < top) top = a.top;
        if (a.bottom > bottom) bottom = a.bottom;
    }

    /**
     * @brief Classifies this rectangle against a clipping rectangle.
     * @param cliprect The clipping rectangle.
     * @return A `VXRECT_INTERSECTION` value: `ALLOUTSIDE`, `PARTINSIDE`, or `ALLINSIDE`.
     * @see IsOutside
     */
    int IsInside(const VxRect &cliprect) const {
        if (left >= cliprect.right) return ALLOUTSIDE;
        if (right < cliprect.left) return ALLOUTSIDE;
        if (top >= cliprect.bottom) return ALLOUTSIDE;
        if (bottom < cliprect.top) return ALLOUTSIDE;
        if (left < cliprect.left) return PARTINSIDE;
        if (right > cliprect.right) return PARTINSIDE;
        if (top < cliprect.top) return PARTINSIDE;
        if (bottom > cliprect.bottom) return PARTINSIDE;
        return ALLINSIDE;
    }

    /**
     * @brief Tests if this rectangle is completely outside a clipping rectangle.
     * @param cliprect The clipping rectangle.
     * @return TRUE if this rectangle is completely outside the clipping rectangle, FALSE otherwise.
     * @see IsInside
     */
    XBOOL IsOutside(const VxRect &cliprect) const {
        if (left >= cliprect.right) return TRUE;
        if (right < cliprect.left) return TRUE;
        if (top >= cliprect.bottom) return TRUE;
        if (bottom < cliprect.top) return TRUE;
        return FALSE;
    }

    /**
     * @brief Tests if a point is inside this rectangle.
     * @param pt The Vx2DVector to test.
     * @return TRUE if the point is inside or on the boundary, FALSE otherwise.
     * @see IsOutside
     */
    XBOOL IsInside(const Vx2DVector &pt) const {
        if (pt.x < left) return FALSE;
        if (pt.x > right) return FALSE;
        if (pt.y < top) return FALSE;
        if (pt.y > bottom) return FALSE;
        return TRUE;
    }

    /**
     * @brief Tests if the rectangle is null (all coordinates are zero).
     * @return TRUE if the rectangle is null, FALSE otherwise.
     * @see IsEmpty, Clear
     */
    XBOOL IsNull() const { return (left == 0 && right == 0 && bottom == 0 && top == 0); }

    /**
     * @brief Tests if a rectangle has zero width or zero height.
     * @return TRUE if the rectangle is empty, FALSE otherwise.
     * @see IsNull, Clear
     */
    XBOOL IsEmpty() const { return ((left == right) || (bottom == top)); }

    /**
     * @brief Clips this rectangle against a clipping rectangle.
     * @param cliprect The clipping rectangle.
     * @return TRUE if the rectangle is still visible (not completely clipped), FALSE otherwise.
     * @see IsOutside, IsInside
     */
    XBOOL Clip(const VxRect &cliprect) {
        if (IsOutside(cliprect)) return FALSE;
        if (left < cliprect.left) left = cliprect.left;
        if (right > cliprect.right) right = cliprect.right;
        if (top < cliprect.top) top = cliprect.top;
        if (bottom > cliprect.bottom) bottom = cliprect.bottom;
        return TRUE;
    }

    /**
     * @brief Clips a point to be within this rectangle.
     * @param pt The point to clip.
     * @param excluderightbottom If TRUE, the point is clamped to `right-1` and `bottom-1` if it is on or beyond the right/bottom edge.
     * @see IsInside
     */
    void Clip(Vx2DVector &pt, XBOOL excluderightbottom = TRUE) const {
        if (pt.x < left) pt.x = left;
        else if (pt.x >= right) {
            if (excluderightbottom) pt.x = right - 1;
            else pt.x = right;
        }
        if (pt.y < top) pt.y = top;
        else if (pt.y >= bottom) {
            if (excluderightbottom) pt.y = bottom - 1;
            else pt.y = bottom;
        }
    }

    /**
     * @brief Transforms the rectangle's coordinates from a source screen space to a destination screen space.
     * @param destScreen A rectangle defining the destination screen area.
     * @param srcScreen A rectangle defining the source screen area.
     * @see TransformToHomogeneous
     */
    VX_EXPORT void Transform(const VxRect &destScreen, const VxRect &srcScreen);
    /**
     * @brief Transforms the rectangle's coordinates from a source screen size to a destination screen size.
     * @param destScreenSize A vector defining the destination screen size.
     * @param srcScreenSize A vector defining the source screen size.
     */
    VX_EXPORT void Transform(const Vx2DVector &destScreenSize, const Vx2DVector &srcScreenSize);

    /**
     * @brief Transforms a screen coordinate rectangle to a homogeneous rectangle [0,1].
     * @param screen A rectangle defining the screen space.
     * @see Transform, TransformFromHomogeneous
     */
    VX_EXPORT void TransformToHomogeneous(const VxRect &screen);

    /**
     * @brief Transforms a homogeneous rectangle [0,1] to a screen coordinate rectangle.
     * @param screen A rectangle defining the screen space.
     * @see Transform, TransformToHomogeneous
     */
    VX_EXPORT void TransformFromHomogeneous(const VxRect &screen);

    /// @brief Translates the rectangle by adding a vector.
    VxRect &operator+=(const Vx2DVector &t) {
        left += t.x; right += t.x;
        top += t.y; bottom += t.y;
        return *this;
    }

    /// @brief Translates the rectangle by subtracting a vector.
    VxRect &operator-=(const Vx2DVector &t) {
        left -= t.x; right -= t.x;
        top -= t.y; bottom -= t.y;
        return *this;
    }

    /// @brief Multiplies the rectangle's coordinates by a vector's components.
    VxRect &operator*=(const Vx2DVector &t) {
        left *= t.x; right *= t.x;
        top *= t.y; bottom *= t.y;
        return *this;
    }

    /// @brief Divides the rectangle's coordinates by a vector's components.
    VxRect &operator/=(const Vx2DVector &t) {
        float x = 1.0f / t.x;
        float y = 1.0f / t.y;
        left *= x; right *= x;
        top *= y; bottom *= y;
        return *this;
    }

    /// @brief Checks for equality between two rectangles.
    friend int operator==(const VxRect &r1, const VxRect &r2);
    /// @brief Checks for inequality between two rectangles.
    friend int operator!=(const VxRect &r1, const VxRect &r2);
};

/// @brief Checks if two rectangles have identical coordinates.
inline int operator==(const VxRect &r1, const VxRect &r2) {
    return (r1.left == r2.left && r1.right == r2.right && r1.top == r2.top && r1.bottom == r2.bottom);
}

/// @brief Checks if two rectangles do not have identical coordinates.
inline int operator!=(const VxRect &r1, const VxRect &r2) {
    return !(r1 == r2);
}

#endif // VXRECT_H
