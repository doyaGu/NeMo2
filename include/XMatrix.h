#ifndef XSMATRIX_H
#define XSMATRIX_H

#include "VxMathDefines.h"
#include "XUtil.h"

/**
 * @class XMatrix
 * @brief A template class for a 2D matrix of variable width and height.
 *
 * @tparam T The type of elements to be stored in the matrix.
 */
template <class T>
class XMatrix {
public:
    /**
     * @brief Constructs a matrix with specified dimensions.
     * @param iWidth The width (number of columns) of the matrix.
     * @param iHeight The height (number of rows) of the matrix.
     */
    XMatrix(int iWidth = 0, int iHeight = 0) : m_Data(NULL), m_Width(0), m_Height(0) {
        Allocate(iWidth, iHeight);
    }

    /**
     * @brief Destructor. Frees the memory allocated for the matrix data.
     */
    ~XMatrix() {
        delete[] m_Data;
    }

    /// @name Accessors
    ///@{

    /**
     * @brief Returns the width of the matrix (number of columns).
     */
    int GetWidth() const { return m_Width; }

    /**
     * @brief Returns the height of the matrix (number of rows).
     */
    int GetHeight() const { return m_Height; }

    /**
     * @brief Returns the total memory size of the matrix data in bytes.
     */
    int Size() const { return m_Width * m_Height * sizeof(T); }

    /**
     * @brief Frees the memory used by the matrix and resets its dimensions to zero.
     */
    void Clear() {
        delete[] m_Data;
        m_Data = NULL;
        m_Width = 0;
        m_Height = 0;
    }

    /**
     * @brief Creates a matrix with new dimensions, clearing any previous data.
     * @param iWidth The new width.
     * @param iHeight The new height.
     */
    void Create(int iWidth, int iHeight) {
        Clear();
        Allocate(iWidth, iHeight);
    }

    /**
     * @brief Provides const access to an element of the matrix.
     * @param iX The column index.
     * @param iY The row index.
     * @return A const reference to the element at (iX, iY).
     */
    const T &operator()(const int iX, const int iY) const {
        XASSERT(iX < m_Width && iY < m_Height);
        return m_Data[iY * m_Width + iX];
    }

    /**
     * @brief Provides mutable access to an element of the matrix.
     * @param iX The column index.
     * @param iY The row index.
     * @return A reference to the element at (iX, iY).
     */
    T &operator()(const int iX, const int iY) {
        XASSERT(iX < m_Width && iY < m_Height);
        return m_Data[iY * m_Width + iX];
    }
    ///@}

private:
    /**
     * @brief Allocates the memory for the matrix data.
     * @internal
     * @param iWidth The width of the matrix.
     * @param iHeight The height of the matrix.
     */
    void Allocate(int iWidth, int iHeight) {
        int count = iWidth * iHeight;
        if (count > 0) {
            m_Data = new T[count];
            XASSERT(m_Data); // Assert on allocation failure
            m_Width = iWidth;
            m_Height = iHeight;
        }
    }

    /// @name Members
    ///@{

    /// @brief Pointer to the raw data of the matrix, stored in row-major order.
    T *m_Data;

    /// @brief The width of the matrix (number of columns).
    int m_Width;
    /// @brief The height of the matrix (number of rows).
    int m_Height;
    ///@}
};

#endif // XSMATRIX_H