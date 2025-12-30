#include "RadixSort.h"

////////////////////////////////////////////////////////////////////////////////
// Constructor
////////////////////////////////////////////////////////////////////////////////
RadixSorter::RadixSorter() {
    m_Indices = NULL;
    m_Indices2 = NULL;
    m_CurrentSize = 0;

    m_Histogram = new CKDWORD[256 * 4];
    m_Offset = new CKDWORD[256];

    ResetIndices();
}

////////////////////////////////////////////////////////////////////////////////
// Destructor
////////////////////////////////////////////////////////////////////////////////
RadixSorter::~RadixSorter() {
    delete[] m_Offset;
    delete[] m_Histogram;
    delete[] m_Indices2;
    delete[] m_Indices;
}

////////////////////////////////////////////////////////////////////////////////
// Sort - Integer values
////////////////////////////////////////////////////////////////////////////////
/// Sorts 32-bit integers using 4-pass radix sort.
/// Includes temporal coherence optimization and pass skipping.
RadixSorter &RadixSorter::Sort(CKDWORD *input, CKDWORD nb, bool signedvalues) {
    // Resize if needed
    if (nb > m_CurrentSize) {
        delete[] m_Indices2;
        delete[] m_Indices;
        m_Indices = new CKDWORD[nb];
        m_Indices2 = new CKDWORD[nb];
        m_CurrentSize = nb;
        ResetIndices();
    }

    // Clear histograms
    memset(m_Histogram, 0, 256 * 4 * sizeof(CKDWORD));

    // Build histograms in single pass
    bool AlreadySorted = true;
    CKDWORD *Indices = m_Indices;
    CKBYTE *p = (CKBYTE *) input;
    CKBYTE *pe = &p[nb * 4];
    CKDWORD *h0 = &m_Histogram[0];
    CKDWORD *h1 = &m_Histogram[256];
    CKDWORD *h2 = &m_Histogram[512];
    CKDWORD *h3 = &m_Histogram[768];

    if (!signedvalues) {
        CKDWORD PrevVal = input[m_Indices[0]];
        while (p != pe) {
            CKDWORD Val = input[*Indices++];
            if (Val < PrevVal)
                AlreadySorted = false;
            PrevVal = Val;
            h0[*p++]++;
            h1[*p++]++;
            h2[*p++]++;
            h3[*p++]++;
        }
    } else {
        int PrevVal = (int) input[m_Indices[0]];
        while (p != pe) {
            int Val = (int) input[*Indices++];
            if (Val < PrevVal)
                AlreadySorted = false;
            PrevVal = Val;
            h0[*p++]++;
            h1[*p++]++;
            h2[*p++]++;
            h3[*p++]++;
        }
    }

    if (AlreadySorted)
        return *this;

    // Count negative values for signed handling
    CKDWORD NbNegativeValues = 0;
    if (signedvalues) {
        CKDWORD *h3 = &m_Histogram[768];
        for (CKDWORD i = 128; i < 256; i++)
            NbNegativeValues += h3[i];
    }

    // 4-pass radix sort
    for (CKDWORD j = 0; j < 4; j++) {
        CKDWORD *CurCount = &m_Histogram[j << 8];
        bool PerformPass = true;

        // Skip pass if all bytes are identical
        for (CKDWORD i = 0; i < 256; i++) {
            if (CurCount[i] == nb) {
                PerformPass = false;
                break;
            }
            if (CurCount[i])
                break;
        }

        if (PerformPass) {
            if (j != 3 || !signedvalues) {
                // Standard offset calculation
                m_Offset[0] = 0;
                for (CKDWORD i = 1; i < 256; i++)
                    m_Offset[i] = m_Offset[i - 1] + CurCount[i - 1];
            } else {
                // Handle signed integers in MSB pass
                m_Offset[0] = NbNegativeValues;
                for (CKDWORD i = 1; i < 128; i++)
                    m_Offset[i] = m_Offset[i - 1] + CurCount[i - 1];
                m_Offset[128] = 0;
                for (CKDWORD i = 129; i < 256; i++)
                    m_Offset[i] = m_Offset[i - 1] + CurCount[i - 1];
            }

            // Perform sort pass
            CKBYTE *InputBytes = (CKBYTE *) input;
            CKDWORD *Indices = m_Indices;
            CKDWORD *IndicesEnd = &m_Indices[nb];
            InputBytes += j;
            while (Indices != IndicesEnd) {
                CKDWORD id = *Indices++;
                m_Indices2[m_Offset[InputBytes[id << 2]]++] = id;
            }

            // Swap buffers
            CKDWORD *Tmp = m_Indices;
            m_Indices = m_Indices2;
            m_Indices2 = Tmp;
        }
    }
    return *this;
}

////////////////////////////////////////////////////////////////////////////////
// Sort - Floating-point values
////////////////////////////////////////////////////////////////////////////////
/// Sorts floats using radix sort on their binary representation.
/// Negative floats are sorted in reverse order.
RadixSorter &RadixSorter::Sort(float *input2, CKDWORD nb) {
    CKDWORD *input = (CKDWORD *) input2;

    // Resize if needed
    if (nb > m_CurrentSize) {
        delete[] m_Indices2;
        delete[] m_Indices;
        m_Indices = new CKDWORD[nb];
        m_Indices2 = new CKDWORD[nb];
        m_CurrentSize = nb;
        ResetIndices();
    }

    // Clear histograms
    memset(m_Histogram, 0, 256 * 4 * sizeof(CKDWORD));

    // Build histograms in single pass
    {
        float PrevVal = input2[m_Indices[0]];
        bool AlreadySorted = true;
        CKDWORD *Indices = m_Indices;
        CKBYTE *p = (CKBYTE *) input;
        CKBYTE *pe = &p[nb * 4];
        CKDWORD *h0 = &m_Histogram[0];
        CKDWORD *h1 = &m_Histogram[256];
        CKDWORD *h2 = &m_Histogram[512];
        CKDWORD *h3 = &m_Histogram[768];

        while (p != pe) {
            float Val = input2[*Indices++];
            if (Val < PrevVal)
                AlreadySorted = false;
            PrevVal = Val;
            h0[*p++]++;
            h1[*p++]++;
            h2[*p++]++;
            h3[*p++]++;
        }

        if (AlreadySorted)
            return *this;
    }

    // Count negative values
    CKDWORD NbNegativeValues = 0;
    CKDWORD *h3 = &m_Histogram[768];
    for (CKDWORD i = 128; i < 256; i++)
        NbNegativeValues += h3[i];

    // 4-pass radix sort
    for (CKDWORD j = 0; j < 4; j++) {
        CKDWORD *CurCount = &m_Histogram[j << 8];
        bool PerformPass = true;

        for (CKDWORD i = 0; i < 256; i++) {
            if (CurCount[i] == nb) {
                PerformPass = false;
                break;
            }
            if (CurCount[i])
                break;
        }

        if (PerformPass) {
            if (j != 3) {
                // Standard offset calculation
                m_Offset[0] = 0;
                for (CKDWORD i = 1; i < 256; i++)
                    m_Offset[i] = m_Offset[i - 1] + CurCount[i - 1];

                CKBYTE *InputBytes = (CKBYTE *) input;
                CKDWORD *Indices = m_Indices;
                CKDWORD *IndicesEnd = &m_Indices[nb];
                InputBytes += j;
                while (Indices != IndicesEnd) {
                    CKDWORD id = *Indices++;
                    m_Indices2[m_Offset[InputBytes[id << 2]]++] = id;
                }
            } else {
                // Handle MSB pass with negative float reversal
                CKDWORD i;
                m_Offset[0] = NbNegativeValues;
                for (i = 1; i < 128; i++)
                    m_Offset[i] = m_Offset[i - 1] + CurCount[i - 1];

                m_Offset[255] = 0;
                for (i = 0; i < 127; i++)
                    m_Offset[254 - i] = m_Offset[255 - i] + CurCount[255 - i];
                for (i = 128; i < 256; i++)
                    m_Offset[i] += CurCount[i];

                for (i = 0; i < nb; i++) {
                    CKDWORD Radix = input[m_Indices[i]] >> 24;
                    if (Radix < 128)
                        m_Indices2[m_Offset[Radix]++] = m_Indices[i];
                    else
                        m_Indices2[--m_Offset[Radix]] = m_Indices[i];
                }
            }

            // Swap buffers
            CKDWORD *Tmp = m_Indices;
            m_Indices = m_Indices2;
            m_Indices2 = Tmp;
        }
    }
    return *this;
}

////////////////////////////////////////////////////////////////////////////////
// ResetIndices
////////////////////////////////////////////////////////////////////////////////
/// Resets indices to sequential order.
RadixSorter &RadixSorter::ResetIndices() {
    for (CKDWORD i = 0; i < m_CurrentSize; i++) {
        m_Indices[i] = i;
    }
    return *this;
}
