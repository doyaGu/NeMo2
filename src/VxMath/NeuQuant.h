#ifndef NEUQUANT_H
#define NEUQUANT_H

#include "VxMathDefines.h"

// NeuQuant constants
#define NETSIZE 256             // Number of colors used
#define INITRAD (NETSIZE >> 3)  // Initial radius

/**
 * @brief NeuQuant neural network quantizer class
 */
class NeuQuant {
public:
    NeuQuant();

    /**
     * @brief Initialize network with sample pixels
     * @param thepicture Pointer to RGB data (3 bytes per pixel, R, G, B order)
     * @param len Total byte length of picture data
     * @param sample Sampling factor (1 = every pixel, N = every Nth pixel)
     */
    void initnet(const XBYTE *thepicture, int len, int sample);

    /**
     * @brief Main learning loop - trains network on image samples
     */
    void learn();
    /**
     * @brief Unbias network for palette lookup
     */
    void unbiasnet();

    /**
     * @brief Build index lookup table for fast search
     */
    void inxbuild();

    /**
     * @brief Search for best matching color
     * @param r Red value (0-255)
     * @param g Green value (0-255)
     * @param b Blue value (0-255)
     * @return Best matching palette index
     */
    int inxsearch(int r, int g, int b) const;

    /**
     * @brief Get palette entry
     * @param i Palette index (0-255)
     * @param r Output red value
     * @param g Output green value
     * @param b Output blue value
     */
    void getpalette(int i, XBYTE &r, XBYTE &g, XBYTE &b) const;

private:
    /**
     * @brief Search for biased BMU (Best Matching Unit)
     * @return Index of winning neuron
     */
    int contest(int r, int g, int b);

    /**
     * @brief Move single neuron towards sample
     */
    void altersingle(int alpha, int i, int r, int g, int b);

    /**
     * @brief Move neighboring neurons towards sample
     */
    void alterneigh(int rad, int i, int r, int g, int b);

    // Neural network data
    int m_Network[NETSIZE][4];   // [B, G, R, idx] for each neuron
    int m_NetIndex[256];          // Fast lookup index
    int m_Bias[NETSIZE];          // Bias array
    int m_Freq[NETSIZE];          // Frequency array
    int m_Radpower[INITRAD];      // Radpower lookup
    int m_Palette[NETSIZE][3];    // Palette storage [B, G, R] (saved before sorting)

    // Image data
    const XBYTE *m_ThePicture;
    int m_LengthCount;
    int m_SampleFactor;
};

#endif // NEUQUANT_H