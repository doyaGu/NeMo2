#include "NeuQuant.h"

#include <string.h>

//------------------------------------------------------------------------------
// NeuQuant Neural-Net Image Quantization Algorithm
// Based on NeuQuant by Anthony Dekker (1994)
//------------------------------------------------------------------------------

// NeuQuant constants
static const int NCYCLES = 100;           // Number of learning cycles
static const int MAXNETPOS = NETSIZE - 1;
static const int NETBIASSHIFT = 4;        // Bias for color values >> 4
static const int INTBIASSHIFT = 16;       // Bias for fractions
static const int INTBIAS = 1 << INTBIASSHIFT;
static const int GAMMASHIFT = 10;
static const int GAMMA = 1 << GAMMASHIFT;
static const int BETASHIFT = 10;
static const int BETA = INTBIAS >> BETASHIFT; // beta = 1/1024
static const int BETAGAMMA = INTBIAS << (GAMMASHIFT - BETASHIFT);

// Decreasing radius factor
static const int RADIUSBIASSHIFT = 6;
static const int RADIUSBIAS = 1 << RADIUSBIASSHIFT;
static const int INITRADIUS = INITRAD * RADIUSBIAS;
static const int RADIUSDEC = 30;          // Factor to decrease radius

// Decreasing alpha factor
static const int ALPHABIASSHIFT = 10;
static const int INITALPHA = 1 << ALPHABIASSHIFT;

// Radbias and alpharadbias tables
static const int RADBIASSHIFT = 8;
static const int RADBIAS = 1 << RADBIASSHIFT;
static const int ALPHARADBSHIFT = ALPHABIASSHIFT + RADBIASSHIFT;
static const int ALPHARADBIAS = 1 << ALPHARADBSHIFT;

// Prime numbers for sampling step (avoid correlations)
static const int PRIME1 = 499;
static const int PRIME2 = 491;
static const int PRIME3 = 487;
static const int PRIME4 = 503;

NeuQuant::NeuQuant() : m_ThePicture(nullptr), m_LengthCount(0), m_SampleFactor(1) {
    memset(m_Network, 0, sizeof(m_Network));
    memset(m_NetIndex, 0, sizeof(m_NetIndex));
    memset(m_Bias, 0, sizeof(m_Bias));
    memset(m_Freq, 0, sizeof(m_Freq));
    memset(m_Radpower, 0, sizeof(m_Radpower));
    memset(m_Palette, 0, sizeof(m_Palette));
}

/**
 * @brief Initialize network with sample pixels
 * @param thepicture Pointer to RGB data (3 bytes per pixel, R, G, B order)
 * @param len Total byte length of picture data
 * @param sample Sampling factor (1 = every pixel, N = every Nth pixel)
 */
void NeuQuant::initnet(const XBYTE *thepicture, int len, int sample) {
    m_ThePicture = thepicture;
    m_LengthCount = len;
    m_SampleFactor = sample;

    // Initialize bias and freq arrays
    for (int i = 0; i < NETSIZE; ++i) {
        m_Bias[i] = 0;
        m_Freq[i] = INTBIAS / NETSIZE; // 1/NETSIZE
    }

    // Initialize network with evenly distributed colors
    for (int i = 0; i < NETSIZE; ++i) {
        int val = (i << (NETBIASSHIFT + 8)) / NETSIZE; // i * 4096 / 256 = i * 16
        m_Network[i][0] = val; // Red
        m_Network[i][1] = val; // Green
        m_Network[i][2] = val; // Blue
        m_Network[i][3] = i;   // Index
    }
}

/**
 * @brief Main learning loop - trains network on image samples
 */
void NeuQuant::learn() {
    if (m_LengthCount < 4) return;

    int alphadec = 30 + (m_SampleFactor - 1) / 3;
    int samplepixels = m_LengthCount / (3 * m_SampleFactor);
    int delta = samplepixels / NCYCLES;
    if (delta == 0) delta = 1;

    int alpha = INITALPHA;
    int radius = INITRADIUS;
    int rad = radius >> RADIUSBIASSHIFT;
    if (rad <= 1) rad = 0;

    // Initialize radpower lookup
    for (int i = 0; i < rad; ++i) {
        m_Radpower[i] = alpha * (((rad * rad - i * i) * RADBIAS) / (rad * rad));
    }

    // Choose step size based on picture length to avoid correlations
    int step;
    if (m_LengthCount < 4 * PRIME1) {
        step = 3;
    } else if ((m_LengthCount % PRIME1) != 0) {
        step = 3 * PRIME1;
    } else if ((m_LengthCount % PRIME2) != 0) {
        step = 3 * PRIME2;
    } else if ((m_LengthCount % PRIME3) != 0) {
        step = 3 * PRIME3;
    } else {
        step = 3 * PRIME4;
    }

    const XBYTE *p = m_ThePicture;
    const XBYTE *pend = m_ThePicture + m_LengthCount;

    int i = 0;
    while (i < samplepixels) {
        // Get pixel values (scaled up by NETBIASSHIFT)
        // Buffer is in RGB order: p[0]=R, p[1]=G, p[2]=B
        int r = p[0] << NETBIASSHIFT;
        int g = p[1] << NETBIASSHIFT;
        int b = p[2] << NETBIASSHIFT;

        // Find winner neuron and update it
        int j = contest(r, g, b);
        altersingle(alpha, j, r, g, b);

        // Alter neighbors
        if (rad > 0) {
            alterneigh(rad, j, r, g, b);
        }

        // Move to next sample pixel
        p += step;
        if (p >= pend) {
            p -= m_LengthCount;
        }

        ++i;

        // Decay alpha and radius
        if (delta == 0) delta = 1;
        if (i % delta == 0) {
            alpha -= alpha / alphadec;
            radius -= radius / RADIUSDEC;
            rad = radius >> RADIUSBIASSHIFT;
            if (rad <= 1) rad = 0;

            // Recalculate radpower
            for (int k = 0; k < rad; ++k) {
                m_Radpower[k] = alpha * (((rad * rad - k * k) * RADBIAS) / (rad * rad));
            }
        }
    }
}

/**
 * @brief Unbias network for palette lookup
 */
void NeuQuant::unbiasnet() {
    for (int i = 0; i < NETSIZE; ++i) {
        m_Network[i][0] >>= NETBIASSHIFT;
        m_Network[i][1] >>= NETBIASSHIFT;
        m_Network[i][2] >>= NETBIASSHIFT;
        m_Network[i][3] = i; // Store index
        
        // Store unbiased palette values BEFORE sorting
        // Network stores [R, G, B] based on learn() order
        m_Palette[i][0] = m_Network[i][0]; // R
        m_Palette[i][1] = m_Network[i][1]; // G
        m_Palette[i][2] = m_Network[i][2]; // B
    }
}

/**
 * @brief Build index lookup table for fast search
 */
void NeuQuant::inxbuild() {
    int previouscol = 0;
    int startpos = 0;

    for (int i = 0; i < NETSIZE; ++i) {
        // Find neuron with smallest green (index 1) value not yet processed
        int smallval = 1000;
        int smallpos = -1;
        for (int j = i; j < NETSIZE; ++j) {
            if (m_Network[j][1] < smallval) {
                smallval = m_Network[j][1];
                smallpos = j;
            }
        }

        // Swap with position i
        if (smallpos != i) {
            int temp[4];
            temp[0] = m_Network[i][0];
            temp[1] = m_Network[i][1];
            temp[2] = m_Network[i][2];
            temp[3] = m_Network[i][3];
            m_Network[i][0] = m_Network[smallpos][0];
            m_Network[i][1] = m_Network[smallpos][1];
            m_Network[i][2] = m_Network[smallpos][2];
            m_Network[i][3] = m_Network[smallpos][3];
            m_Network[smallpos][0] = temp[0];
            m_Network[smallpos][1] = temp[1];
            m_Network[smallpos][2] = temp[2];
            m_Network[smallpos][3] = temp[3];
        }

        // Build index for green values
        if (smallval != previouscol) {
            m_NetIndex[previouscol] = (startpos + i) >> 1;
            for (int j = previouscol + 1; j < smallval; ++j) {
                m_NetIndex[j] = i;
            }
            previouscol = smallval;
            startpos = i;
        }
    }
    m_NetIndex[previouscol] = (startpos + MAXNETPOS) >> 1;
    for (int j = previouscol + 1; j < 256; ++j) {
        m_NetIndex[j] = MAXNETPOS;
    }
}

/**
 * @brief Search for best matching color
 * @param r Red value (0-255)
 * @param g Green value (0-255)
 * @param b Blue value (0-255)
 * @return Best matching palette index
 */
int NeuQuant::inxsearch(int r, int g, int b) const {
    int bestd = 1000;    // Best distance
    int best = -1;       // Best index
    int i = m_NetIndex[g]; // Index to start at (based on green)
    int j = i - 1;

    // Network stores [R, G, B, idx] - green is at [1] for sorting
    // Search forward and backward until distance is too great
    while (i < NETSIZE || j >= 0) {
        if (i < NETSIZE) {
            int dist = m_Network[i][1] - g;
            if (dist >= bestd) {
                i = NETSIZE; // Stop searching forward
            } else {
                ++i;
                if (dist < 0) dist = -dist;
                int a = m_Network[i - 1][0] - r;  // [0] is R
                if (a < 0) a = -a;
                dist += a;
                if (dist < bestd) {
                    a = m_Network[i - 1][2] - b;  // [2] is B
                    if (a < 0) a = -a;
                    dist += a;
                    if (dist < bestd) {
                        bestd = dist;
                        best = m_Network[i - 1][3];
                    }
                }
            }
        }
        if (j >= 0) {
            int dist = g - m_Network[j][1];
            if (dist >= bestd) {
                j = -1; // Stop searching backward
            } else {
                --j;
                if (dist < 0) dist = -dist;
                int a = m_Network[j + 1][0] - r;  // [0] is R
                if (a < 0) a = -a;
                dist += a;
                if (dist < bestd) {
                    a = m_Network[j + 1][2] - b;  // [2] is B
                    if (a < 0) a = -a;
                    dist += a;
                    if (dist < bestd) {
                        bestd = dist;
                        best = m_Network[j + 1][3];
                    }
                }
            }
        }
    }
    return best;
}

/**
 * @brief Get palette entry
 * @param i Palette index (0-255)
 * @param r Output red value
 * @param g Output green value
 * @param b Output blue value
 */
void NeuQuant::getpalette(int i, XBYTE &r, XBYTE &g, XBYTE &b) const {
    // Palette was stored in original order before inxbuild sorting
    // Network stores [R, G, B] based on the order fed in learn()
    r = (XBYTE)m_Palette[i][0];
    g = (XBYTE)m_Palette[i][1];
    b = (XBYTE)m_Palette[i][2];
}

/**
 * @brief Search for biased BMU (Best Matching Unit)
 * @return Index of winning neuron
 */
int NeuQuant::contest(int r, int g, int b) {
    // Finds closest neuron (using bias and freq)
    int bestd = INT_MAX;
    int bestbiasd = INT_MAX;
    int bestpos = -1;
    int bestbiaspos = -1;

    for (int i = 0; i < NETSIZE; ++i) {
        // Calculate Manhattan distance (faster than Euclidean)
        // Network stores [R, G, B, idx]
        int dist = m_Network[i][0] - r;
        if (dist < 0) dist = -dist;
        int a = m_Network[i][1] - g;
        if (a < 0) a = -a;
        dist += a;
        a = m_Network[i][2] - b;
        if (a < 0) a = -a;
        dist += a;

        if (dist < bestd) {
            bestd = dist;
            bestpos = i;
        }

        // Factor in bias for frequently used neurons
        int biasdist = dist - ((m_Bias[i]) >> (INTBIASSHIFT - NETBIASSHIFT));
        if (biasdist < bestbiasd) {
            bestbiasd = biasdist;
            bestbiaspos = i;
        }

        // Update frequency and bias
        int betafreq = m_Freq[i] >> BETASHIFT;
        m_Freq[i] -= betafreq;
        m_Bias[i] += betafreq << GAMMASHIFT;
    }

    // Boost frequency of best match
    m_Freq[bestpos] += BETA;
    m_Bias[bestpos] -= BETAGAMMA;

    return bestbiaspos;
}

/**
 * @brief Move single neuron towards sample
 */
void NeuQuant::altersingle(int alpha, int i, int r, int g, int b) {
    // Adjust weights - network stores [R, G, B, idx]
    m_Network[i][0] -= (alpha * (m_Network[i][0] - r)) / INITALPHA;
    m_Network[i][1] -= (alpha * (m_Network[i][1] - g)) / INITALPHA;
    m_Network[i][2] -= (alpha * (m_Network[i][2] - b)) / INITALPHA;
}

/**
 * @brief Move neighboring neurons towards sample
 */
void NeuQuant::alterneigh(int rad, int i, int r, int g, int b) {
    int lo = i - rad;
    if (lo < -1) lo = -1;
    int hi = i + rad;
    if (hi > NETSIZE) hi = NETSIZE;

    int j = i + 1;
    int k = i - 1;
    int *q = m_Radpower;

    while (j < hi || k > lo) {
        int *radp = ++q;
        int a = *radp;

        if (j < hi) {
            // Network stores [R, G, B, idx]
            m_Network[j][0] -= (a * (m_Network[j][0] - r)) / ALPHARADBIAS;
            m_Network[j][1] -= (a * (m_Network[j][1] - g)) / ALPHARADBIAS;
            m_Network[j][2] -= (a * (m_Network[j][2] - b)) / ALPHARADBIAS;
            ++j;
        }
        if (k > lo) {
            // Network stores [R, G, B, idx]
            m_Network[k][0] -= (a * (m_Network[k][0] - r)) / ALPHARADBIAS;
            m_Network[k][1] -= (a * (m_Network[k][1] - g)) / ALPHARADBIAS;
            m_Network[k][2] -= (a * (m_Network[k][2] - b)) / ALPHARADBIAS;
            --k;
        }
    }
}
