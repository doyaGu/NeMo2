#ifndef RADIXSORT_H
#define RADIXSORT_H

#include "CKTypes.h"

/// Radix Sort implementation for integer and floating-point values.
/// Maintains sorted indices without modifying the original buffer.
class RadixSorter {
public:
    RadixSorter();
    ~RadixSorter();

    /// Sorts 32-bit integer values.
    /// @param input Array of values to sort
    /// @param nb Number of elements
    /// @param signedvalues True to handle signed integers (including negatives)
    RadixSorter &Sort(CKDWORD *input, CKDWORD nb, bool signedvalues = true);

    /// Sorts floating-point values.
    /// @param input Array of float values to sort
    /// @param nb Number of elements
    RadixSorter &Sort(float *input, CKDWORD nb);

    /// Returns sorted indices array.
    CKDWORD *GetIndices() { return m_Indices; }

    /// Resets indices to sequential order (0, 1, 2, ...).
    RadixSorter &ResetIndices();

private:
    CKDWORD *m_Histogram;  // Byte frequency counters (4 passes × 256)
    CKDWORD *m_Offset;     // Cumulative distribution offsets (256 entries)
    CKDWORD m_CurrentSize; // Current capacity of indices arrays
    CKDWORD *m_Indices;    // Primary indices array (sorted result)
    CKDWORD *m_Indices2;   // Secondary buffer (swapped each pass)
};

#endif // RADIXSORT_H
