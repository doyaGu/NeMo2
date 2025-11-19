#ifndef VXTIMEPROFILER_H
#define VXTIMEPROFILER_H

#include "VxMathDefines.h"

/**
 * @brief A class for high-precision performance timing.
 *
 * @remarks
 * This class provides simple methods to accurately measure the time elapsed
 * between operations, which is useful for profiling and performance analysis.
 *
 * @example
 * @code
 * // To profile a piece of code:
 * VxTimeProfiler MyProfiler; // Starts the timer
 *
 * // ... code to be measured ...
 *
 * float elapsed_ms = MyProfiler.Current(); // Get elapsed time in milliseconds
 *
 * // To profile multiple items sequentially:
 * MyProfiler.Reset(); // Restart timer for the next item
 *
 * // ... second piece of code ...
 *
 * float elapsed_ms2 = MyProfiler.Current();
 *
 * // To get time and reset simultaneously:
 * float elapsed_ms3 = MyProfiler.Split();
 * @endcode
 */
class VX_EXPORT VxTimeProfiler {
public:
    /**
     * @brief Constructs a VxTimeProfiler and starts the timer by calling Reset().
     */
    VxTimeProfiler() { Reset(); }

    /// @brief Assignment operator.
    VxTimeProfiler &operator=(const VxTimeProfiler &t);

    /**
     * @brief Resets the timer, setting the start time to the current time.
     */
    void Reset();

    /**
     * @brief Returns the time elapsed in milliseconds since the last Reset() or construction.
     * @return The elapsed time in milliseconds as a float.
     */
    float Current();

    /**
     * @brief Returns the current elapsed time and immediately resets the timer.
     * @return The elapsed time in milliseconds before the timer was reset.
     */
    float Split() {
        float c = Current();
        Reset();
        return c;
    }

protected:
    /// @brief Internal storage for timing data, likely OS-specific high-resolution timer values.
    XULONG Times[4];
};

#endif // VXTIMEPROFILER_H
