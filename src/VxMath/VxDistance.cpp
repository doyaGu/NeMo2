#include "VxDistance.h"

#include "VxRay.h"
#include "VxVector.h"

// Line-Line distance calculations

float VxDistance::LineLineSquareDistance(const VxRay &line0, const VxRay &line1, float *t0, float *t1) {
    // Compute difference between origins
    float diffX = line0.m_Origin.x - line1.m_Origin.x;
    float diffY = line0.m_Origin.y - line1.m_Origin.y;
    float diffZ = line0.m_Origin.z - line1.m_Origin.z;

    // a = |dir0|^2
    float a = line0.m_Direction.x * line0.m_Direction.x
            + line0.m_Direction.y * line0.m_Direction.y
            + line0.m_Direction.z * line0.m_Direction.z;

    // b = -dot(dir0, dir1)
    float b = -(line0.m_Direction.z * line1.m_Direction.z
              + line0.m_Direction.y * line1.m_Direction.y
              + line0.m_Direction.x * line1.m_Direction.x);

    // c = |dir1|^2
    float c = line1.m_Direction.x * line1.m_Direction.x
            + line1.m_Direction.y * line1.m_Direction.y
            + line1.m_Direction.z * line1.m_Direction.z;

    // d = dot(diff, dir0)
    float d = diffZ * line0.m_Direction.z + diffY * line0.m_Direction.y + diffX * line0.m_Direction.x;

    // f = |diff|^2
    float f = diffZ * diffZ + diffY * diffY + diffX * diffX;

    // det = a*c - b*b
    float det = XAbs(c * a - b * b);

    float s0, s1;
    float result;

    if (det < EPSILON) {
        // Parallel case
        s1 = 0.0f;
        s0 = -(d / a);
        result = d * s0;
    } else {
        // e = -dot(diff, dir1)
        float e = -(diffZ * line1.m_Direction.z + diffY * line1.m_Direction.y + diffX * line1.m_Direction.x);
        float invDet = 1.0f / det;
        s0 = (e * b - d * c) * invDet;
        s1 = (d * b - e * a) * invDet;
        // Squared distance = |diff + s0*dir0 - s1*dir1|^2
        // Expanded: f + 2*(s0*d + s1*e) + s0^2*a + s1^2*c + 2*s0*s1*b
        result = (s1 * c + s0 * b + e + e) * s1 + (s1 * b + s0 * a + d + d) * s0;
    }

    if (t0) *t0 = s0;
    if (t1) *t1 = s1;

    return XAbs(result + f);
}

float VxDistance::LineRaySquareDistance(const VxRay &line, const VxRay &ray, float *t0, float *t1) {
    // Compute difference between origins
    float diffX = line.m_Origin.x - ray.m_Origin.x;
    float diffY = line.m_Origin.y - ray.m_Origin.y;
    float diffZ = line.m_Origin.z - ray.m_Origin.z;

    // a = |dir0|^2
    float a = line.m_Direction.x * line.m_Direction.x
            + line.m_Direction.y * line.m_Direction.y
            + line.m_Direction.z * line.m_Direction.z;

    // b = -dot(dir0, dir1)
    float b = -(line.m_Direction.z * ray.m_Direction.z
              + line.m_Direction.y * ray.m_Direction.y
              + line.m_Direction.x * ray.m_Direction.x);

    // c = |dir1|^2
    float c = ray.m_Direction.x * ray.m_Direction.x
            + ray.m_Direction.y * ray.m_Direction.y
            + ray.m_Direction.z * ray.m_Direction.z;

    // d = dot(diff, dir0)
    float d = diffZ * line.m_Direction.z + diffY * line.m_Direction.y + diffX * line.m_Direction.x;

    // f = |diff|^2
    float f = diffZ * diffZ + diffY * diffY + diffX * diffX;

    // det = a*c - b*b
    float det = XAbs(c * a - b * b);

    float s0, s1;
    float result;

    // e = -dot(diff, dir1)
    float e = -(diffZ * ray.m_Direction.z + diffY * ray.m_Direction.y + diffX * ray.m_Direction.x);

    // Check for parallel or ray parameter < 0
    float rayParam = d * b - e * a;

    if (det < EPSILON || rayParam < 0.0f) {
        // Parallel case or ray parameter negative - clamp ray parameter to 0
        s1 = 0.0f;
        s0 = -(d / a);
        result = d * s0;
    } else {
        float invDet = 1.0f / det;
        s0 = (e * b - d * c) * invDet;
        s1 = invDet * rayParam;
        result = (b * s0 + s1 * c + e + e) * s1 + (s0 * a + s1 * b + d + d) * s0;
    }

    if (t0) *t0 = s0;
    if (t1) *t1 = s1;

    return XAbs(result + f);
}

float VxDistance::LineSegmentSquareDistance(const VxRay &line, const VxRay &segment, float *t0, float *t1) {
    // Compute difference between origins
    float diffX = line.m_Origin.x - segment.m_Origin.x;
    float diffY = line.m_Origin.y - segment.m_Origin.y;
    float diffZ = line.m_Origin.z - segment.m_Origin.z;

    // a = |dir0|^2
    float a = line.m_Direction.x * line.m_Direction.x
            + line.m_Direction.y * line.m_Direction.y
            + line.m_Direction.z * line.m_Direction.z;

    // b = -dot(dir0, dir1)
    float b = -(line.m_Direction.z * segment.m_Direction.z
              + line.m_Direction.y * segment.m_Direction.y
              + line.m_Direction.x * segment.m_Direction.x);

    // c = |dir1|^2
    float c = segment.m_Direction.x * segment.m_Direction.x
            + segment.m_Direction.y * segment.m_Direction.y
            + segment.m_Direction.z * segment.m_Direction.z;

    // d = dot(diff, dir0)
    float d = diffZ * line.m_Direction.z + diffY * line.m_Direction.y + diffX * line.m_Direction.x;

    // f = |diff|^2
    float f = diffZ * diffZ + diffY * diffY + diffX * diffX;

    // det = a*c - b*b
    float det = XAbs(c * a - b * b);

    float s0, s1;
    float result;

    // e = -dot(diff, dir1)
    float e = -(diffZ * segment.m_Direction.z + diffY * segment.m_Direction.y + diffX * segment.m_Direction.x);

    // Check segment parameter = d*b - e*a
    float segParam = d * b - e * a;

    if (det < EPSILON || segParam < 0.0f) {
        // Parallel case or segment parameter negative - clamp to 0
        s1 = 0.0f;
        s0 = -(d / a);
        result = d * s0;
    } else if (segParam > det) {
        // Segment parameter > 1 - clamp to 1
        float dPlusB = d + b;
        s1 = 1.0f;
        s0 = -(dPlusB / a);
        result = dPlusB * s0 + e + e + f + c;
        goto output;
    } else {
        float invDet = 1.0f / det;
        s0 = (e * b - d * c) * invDet;
        s1 = invDet * segParam;
        result = (b * s0 + s1 * c + e + e) * s1 + (s0 * a + s1 * b + d + d) * s0;
    }

    result = result + f;

output:
    if (t0) *t0 = s0;
    if (t1) *t1 = s1;

    return XAbs(result);
}

float VxDistance::RayRaySquareDistance(const VxRay &ray0, const VxRay &ray1, float *t0, float *t1) {
    // Compute difference between origins
    float diffX = ray0.m_Origin.x - ray1.m_Origin.x;
    float diffY = ray0.m_Origin.y - ray1.m_Origin.y;
    float diffZ = ray0.m_Origin.z - ray1.m_Origin.z;

    // a = |dir0|^2
    float a = ray0.m_Direction.x * ray0.m_Direction.x
            + ray0.m_Direction.y * ray0.m_Direction.y
            + ray0.m_Direction.z * ray0.m_Direction.z;

    // b = -dot(dir0, dir1)
    float b = -(ray0.m_Direction.z * ray1.m_Direction.z
              + ray0.m_Direction.y * ray1.m_Direction.y
              + ray0.m_Direction.x * ray1.m_Direction.x);

    // c = |dir1|^2
    float c = ray1.m_Direction.x * ray1.m_Direction.x
            + ray1.m_Direction.y * ray1.m_Direction.y
            + ray1.m_Direction.z * ray1.m_Direction.z;

    // d = dot(diff, dir0)
    float d = diffZ * ray0.m_Direction.z + diffY * ray0.m_Direction.y + diffX * ray0.m_Direction.x;

    // f = |diff|^2
    float f = diffZ * diffZ + diffY * diffY + diffX * diffX;

    // det = a*c - b*b
    float det = XAbs(c * a - b * b);

    float s0, s1;

    if (det < EPSILON) {
        // Parallel case
        if (d >= 0.0f) {
            s0 = 0.0f;
            // e = -dot(diff, dir1)
            float e = -(diffZ * ray1.m_Direction.z + diffY * ray1.m_Direction.y + diffX * ray1.m_Direction.x);
            s1 = -(e / c);
            f = f + s1 * e;
        } else {
            s1 = 0.0f;
            s0 = -(d / a);
            f = f + s0 * d;
        }
    } else {
        // e = -dot(diff, dir1)
        float e = -(diffZ * ray1.m_Direction.z + diffY * ray1.m_Direction.y + diffX * ray1.m_Direction.x);
        float ray0Param = e * b - d * c;
        float ray1Param = d * b - e * a;

        if (ray0Param < 0.0f) {
            // ray0 parameter < 0
            if (ray1Param < 0.0f) {
                // Both parameters < 0
                if (d >= 0.0f) {
                    if (e >= 0.0f) {
                        s0 = 0.0f;
                        s1 = 0.0f;
                    } else {
                        s0 = 0.0f;
                        s1 = -(e / c);
                        f = f + s1 * e;
                    }
                } else {
                    s1 = 0.0f;
                    s0 = -(d / a);
                    f = f + s0 * d;
                }
            } else {
                // ray0 param < 0, ray1 param >= 0
                if (d >= 0.0f) {
                    if (e >= 0.0f) {
                        s0 = 0.0f;
                        s1 = 0.0f;
                    } else {
                        s0 = 0.0f;
                        s1 = -(e / c);
                        f = f + s1 * e;
                    }
                } else {
                    s1 = 0.0f;
                    s0 = -(d / a);
                    f = f + s0 * d;
                }
            }
        } else if (ray1Param < 0.0f) {
            // ray0 param >= 0, ray1 param < 0
            s1 = 0.0f;
            if (d < 0.0f) {
                s0 = -(d / a);
                f = f + s0 * d;
            } else {
                s0 = 0.0f;
            }
        } else {
            // Both parameters >= 0
            float invDet = 1.0f / det;
            s0 = ray0Param * invDet;
            s1 = invDet * ray1Param;
            f = f + (s1 * c + s0 * b + e + e) * s1 + (s1 * b + s0 * a + d + d) * s0;
        }
    }

    if (t0) *t0 = s0;
    if (t1) *t1 = s1;

    return XAbs(f);
}

float VxDistance::RaySegmentSquareDistance(const VxRay &ray, const VxRay &segment, float *t0, float *t1) {
    // Compute difference between origins
    float diffX = ray.m_Origin.x - segment.m_Origin.x;
    float diffY = ray.m_Origin.y - segment.m_Origin.y;
    float diffZ = ray.m_Origin.z - segment.m_Origin.z;

    // a = |dir0|^2
    float a = ray.m_Direction.x * ray.m_Direction.x
            + ray.m_Direction.y * ray.m_Direction.y
            + ray.m_Direction.z * ray.m_Direction.z;

    // b = -dot(dir0, dir1)
    float b = -(ray.m_Direction.z * segment.m_Direction.z
              + ray.m_Direction.y * segment.m_Direction.y
              + ray.m_Direction.x * segment.m_Direction.x);

    // c = |dir1|^2
    float c = segment.m_Direction.x * segment.m_Direction.x
            + segment.m_Direction.y * segment.m_Direction.y
            + segment.m_Direction.z * segment.m_Direction.z;

    // d = dot(diff, dir0)
    float d = diffZ * ray.m_Direction.z + diffY * ray.m_Direction.y + diffX * ray.m_Direction.x;

    // f = |diff|^2
    float f = diffZ * diffZ + diffY * diffY + diffX * diffX;

    // det = a*c - b*b
    float det = XAbs(c * a - b * b);

    float s0, s1;

    if (det < EPSILON) {
        // Parallel case
        if (b < 0.0f) {
            // Directions point in same general direction
            // Check endpoint at segment t=1
            s1 = 1.0f;
            // e = -dot(diff, dir1)
            float e = -(diffZ * segment.m_Direction.z + diffY * segment.m_Direction.y + diffX * segment.m_Direction.x);
            float dPlusB = d + b;
            if (dPlusB >= 0.0f) {
                s0 = 0.0f;
                f = f + e + e + c;
            } else {
                s0 = -(dPlusB / a);
                f = f + dPlusB * s0 + e + e + c;
            }
        } else {
            // Check endpoint at segment t=0
            if (d < 0.0f) {
                s1 = 0.0f;
                s0 = -(d / a);
                f = f + s0 * d;
            } else {
                s0 = 0.0f;
                s1 = 0.0f;
            }
        }
    } else {
        // e = -dot(diff, dir1)
        float e = -(diffZ * segment.m_Direction.z + diffY * segment.m_Direction.y + diffX * segment.m_Direction.x);
        float rayParam = e * b - d * c;
        float segParam = d * b - e * a;

        if (rayParam < 0.0f) {
            // Ray parameter < 0
            if (segParam <= 0.0f) {
                // Both <= 0, origin point of ray and segment
                if (d < 0.0f) {
                    s1 = 0.0f;
                    s0 = -(d / a);
                    f = f + s0 * d;
                    goto output;
                }
            } else if (segParam > det) {
                // Segment param > 1
                float dPlusB = d + b;
                if (dPlusB < 0.0f) {
                    s1 = 1.0f;
                    s0 = -(dPlusB / a);
                    f = f + dPlusB * s0 + e + e + c;
                    goto output;
                }
            }
            // Check e
            if (e >= 0.0f) {
                s0 = 0.0f;
                s1 = 0.0f;
                goto output;
            }
            // Check if -e > c
            if (-e > c) {
                s0 = -(e / c);
                s1 = s0;
                f = f + s0 * e;
                goto output;
            }
            s1 = 1.0f;
            s0 = 0.0f;
            f = f + e + e + c;
            goto output;
        }

        if (segParam < 0.0f) {
            // Segment parameter < 0
            s1 = 0.0f;
            if (d < 0.0f) {
                s0 = -(d / a);
            } else {
                s0 = 0.0f;
            }
            f = f + s0 * d;
        } else if (segParam > det) {
            // Segment parameter > 1
            s1 = 1.0f;
            if (-b <= d) {
                s0 = 0.0f;
                f = f + e + e + c;
            } else {
                float dPlusB = d + b;
                s0 = -(dPlusB / a);
                f = f + dPlusB * s0 + e + e + c;
            }
        } else {
            // Both in range
            float invDet = 1.0f / det;
            s0 = rayParam * invDet;
            s1 = invDet * segParam;
            f = f + (s1 * c + s0 * b + e + e) * s1 + (s1 * b + s0 * a + d + d) * s0;
        }
    }

output:
    if (t0) *t0 = s0;
    if (t1) *t1 = s1;

    return XAbs(f);
}

float VxDistance::SegmentSegmentSquareDistance(const VxRay &segment0, const VxRay &segment1, float *t0, float *t1) {
    // Compute difference between origins
    float diffX = segment0.m_Origin.x - segment1.m_Origin.x;
    float diffY = segment0.m_Origin.y - segment1.m_Origin.y;
    float diffZ = segment0.m_Origin.z - segment1.m_Origin.z;

    // a = |dir0|^2
    float a = segment0.m_Direction.x * segment0.m_Direction.x
            + segment0.m_Direction.y * segment0.m_Direction.y
            + segment0.m_Direction.z * segment0.m_Direction.z;

    // b = -dot(dir0, dir1)
    float b = -(segment0.m_Direction.z * segment1.m_Direction.z
              + segment0.m_Direction.y * segment1.m_Direction.y
              + segment0.m_Direction.x * segment1.m_Direction.x);

    // c = |dir1|^2
    float c = segment1.m_Direction.x * segment1.m_Direction.x
            + segment1.m_Direction.y * segment1.m_Direction.y
            + segment1.m_Direction.z * segment1.m_Direction.z;

    // d = dot(diff, dir0)
    float d = diffZ * segment0.m_Direction.z + diffY * segment0.m_Direction.y + diffX * segment0.m_Direction.x;

    // f = |diff|^2
    float f = diffZ * diffZ + diffY * diffY + diffX * diffX;

    // det = a*c - b*b
    float det = XAbs(c * a - b * b);

    float s0, s1;

    if (det < EPSILON) {
        // Parallel case
        if (d <= 0.0f) {
            if (-d <= a) {
                if (d <= 0.0f) {
                    s0 = 0.0f;
                    s1 = 0.0f;
                } else if (-d > a) {
                    s1 = 0.0f;
                    s0 = 1.0f;
                    f = f + d + d + a;
                } else {
                    s0 = -(d / a);
                    s1 = 0.0f;
                    f = f + s0 * d;
                }
            } else {
                // -d > a, meaning d < -a
                s1 = 0.0f;
                s0 = 1.0f;
                f = f + d + d + a;
            }
        } else {
            // d > 0
            if (-d < a) {
                // e = -dot(diff, dir1)
                float e = -(diffZ * segment1.m_Direction.z + diffY * segment1.m_Direction.y + diffX * segment1.m_Direction.x);
                float dPlusA = d + a;
                s0 = 1.0f;
                if (-dPlusA < b) {
                    s1 = 1.0f;
                    f = f + e + d + b + e + d + b + c + a;
                } else {
                    s1 = -(dPlusA / b);
                    f = f + (e + d + b + e + d + b + s1 * c) * s1 + d + d + a;
                }
            } else {
                s0 = -(d / a);
                s1 = 0.0f;
                f = f + s0 * d;
            }
        }
    } else {
        // Non-parallel case
        // e = -dot(diff, dir1)
        float e = -(diffZ * segment1.m_Direction.z + diffY * segment1.m_Direction.y + diffX * segment1.m_Direction.x);
        float seg0Param = e * b - d * c;
        float seg1Param = d * b - e * a;

        if (seg0Param < 0.0f) {
            // seg0 parameter < 0
            if (seg1Param < 0.0f) {
                // Both < 0
                if (d < 0.0f) {
                    s1 = 0.0f;
                    if (-d >= a) {
                        s0 = 1.0f;
                        f = f + d + d + a;
                    } else {
                        s0 = -(d / a);
                        f = f + s0 * d;
                    }
                } else {
                    s0 = 0.0f;
                    if (e >= 0.0f) {
                        s1 = 0.0f;
                    } else if (-e >= c) {
                        s1 = 1.0f;
                        f = f + e + e + c;
                    } else {
                        s1 = -(e / c);
                        f = f + s1 * e;
                    }
                }
            } else if (seg1Param > det) {
                // seg0 param < 0, seg1 param > 1
                float dPlusB = d + b;
                if (dPlusB < 0.0f) {
                    s1 = 1.0f;
                    if (-dPlusB >= a) {
                        s0 = 1.0f;
                        f = f + dPlusB + e + dPlusB + e + c + a;
                    } else {
                        s0 = -(dPlusB / a);
                        f = f + dPlusB * s0 + e + e + c;
                    }
                } else {
                    s0 = 0.0f;
                    if (e >= 0.0f) {
                        s1 = 0.0f;
                    } else if (-e >= c) {
                        s1 = 1.0f;
                        f = f + e + e + c;
                    } else {
                        s1 = -(e / c);
                        f = f + s1 * e;
                    }
                }
            } else {
                // seg0 param < 0, seg1 param in [0, 1]
                if (d < 0.0f) {
                    s1 = 0.0f;
                    if (-d >= a) {
                        s0 = 1.0f;
                        f = f + d + d + a;
                    } else {
                        s0 = -(d / a);
                        f = f + s0 * d;
                    }
                } else {
                    s0 = 0.0f;
                    if (e >= 0.0f) {
                        s1 = 0.0f;
                    } else if (-e >= c) {
                        s1 = 1.0f;
                        f = f + e + e + c;
                    } else {
                        s1 = -(e / c);
                        f = f + s1 * e;
                    }
                }
            }
        } else if (seg0Param > det) {
            // seg0 parameter > 1
            if (seg1Param < 0.0f) {
                // seg0 param > 1, seg1 param < 0
                float dPlusB = d + b;
                if (dPlusB < 0.0f) {
                    s1 = 1.0f;
                    if (-dPlusB >= a) {
                        s0 = 1.0f;
                        f = f + dPlusB + e + dPlusB + e + c + a;
                    } else {
                        s0 = -(dPlusB / a);
                        f = f + dPlusB * s0 + e + e + c;
                    }
                } else {
                    s0 = 0.0f;
                    if (e >= 0.0f) {
                        s1 = 0.0f;
                    } else if (-e >= c) {
                        s1 = 1.0f;
                        f = f + e + e + c;
                    } else {
                        s1 = -(e / c);
                        f = f + s1 * e;
                    }
                }
            } else if (seg1Param > det) {
                // Both > det (both segments at end)
                float ePlusB = e + b;
                s0 = 1.0f;
                if (-ePlusB < c) {
                    s1 = -(ePlusB / c);
                    f = f + ePlusB * s1 + d + d + a;
                } else {
                    s1 = 1.0f;
                    f = f + ePlusB + d + b + ePlusB + d + b + c + a;
                }
            } else {
                // seg0 param > 1, seg1 param in [0, 1]
                // FIX: Clamp s0 to 1, compute optimal s1 (was incorrectly clamping s1)
                s0 = 1.0f;
                float ePlusB = e + b;
                if (ePlusB >= 0.0f) {
                    // s1 would be negative, clamp to 0
                    s1 = 0.0f;
                    f = f + d + d + a;
                } else if (-ePlusB >= c) {
                    // s1 would be > 1, clamp to 1
                    s1 = 1.0f;
                    f = f + ePlusB + ePlusB + c + d + d + a;
                } else {
                    // s1 is in (0, 1)
                    s1 = -(ePlusB / c);
                    f = f + ePlusB * s1 + d + d + a;
                }
            }
        } else {
            // seg0 param in [0, det]
            if (seg1Param < 0.0f) {
                // seg0 param in range, seg1 param < 0
                if (d < 0.0f) {
                    s1 = 0.0f;
                    if (-d >= a) {
                        s0 = 1.0f;
                        f = f + d + d + a;
                    } else {
                        s0 = -(d / a);
                        f = f + s0 * d;
                    }
                } else {
                    s0 = 0.0f;
                    if (e >= 0.0f) {
                        s1 = 0.0f;
                    } else if (-e >= c) {
                        s1 = 1.0f;
                        f = f + e + e + c;
                    } else {
                        s1 = -(e / c);
                        f = f + s1 * e;
                    }
                }
            } else if (seg1Param > det) {
                // seg0 param in range, seg1 param > 1
                float dPlusB = d + b;
                s1 = 1.0f;
                if (dPlusB < 0.0f) {
                    if (-dPlusB >= a) {
                        s0 = 1.0f;
                        f = f + dPlusB + e + dPlusB + e + c + a;
                    } else {
                        s0 = -(dPlusB / a);
                        f = f + dPlusB * s0 + e + e + c;
                    }
                } else {
                    s0 = 0.0f;
                    f = f + e + e + c;
                }
            } else {
                // Both in range
                float invDet = 1.0f / det;
                s0 = seg0Param * invDet;
                s1 = invDet * seg1Param;
                f = f + (s1 * c + s0 * b + e + e) * s1 + (s1 * b + s0 * a + d + d) * s0;
            }
        }
    }

    if (t0) *t0 = s0;
    if (t1) *t1 = s1;

    return XAbs(f);
}

// Point-Line distance calculations

float VxDistance::PointLineSquareDistance(const VxVector &point, const VxRay &line, float *t0) {
    // Compute difference
    float diffX = point.x - line.m_Origin.x;
    float diffY = point.y - line.m_Origin.y;
    float diffZ = point.z - line.m_Origin.z;

    // t = dot(diff, dir) / dot(dir, dir)
    float t = (diffZ * line.m_Direction.z + diffY * line.m_Direction.y + diffX * line.m_Direction.x)
            / (line.m_Direction.x * line.m_Direction.x
             + line.m_Direction.y * line.m_Direction.y
             + line.m_Direction.z * line.m_Direction.z);

    // Compute closest point offset
    float closestY = t * line.m_Direction.y;
    float closestZ = t * line.m_Direction.z;
    float dx = diffX - t * line.m_Direction.x;

    if (t0) *t0 = t;

    float dy = diffY - closestY;
    float dz = diffZ - closestZ;

    return dz * dz + dx * dx + dy * dy;
}

float VxDistance::PointRaySquareDistance(const VxVector &point, const VxRay &ray, float *t0) {
    // Compute difference
    float diffX = point.x - ray.m_Origin.x;
    float diffY = point.y - ray.m_Origin.y;
    float diffZ = point.z - ray.m_Origin.z;

    // dot(diff, dir)
    float dotDiffDir = diffZ * ray.m_Direction.z + diffY * ray.m_Direction.y + diffX * ray.m_Direction.x;

    float t;
    if (dotDiffDir > 0.0f) {
        // t = dot(diff, dir) / dot(dir, dir)
        t = dotDiffDir
          / (ray.m_Direction.x * ray.m_Direction.x
           + ray.m_Direction.y * ray.m_Direction.y
           + ray.m_Direction.z * ray.m_Direction.z);

        // Compute closest point offset
        float closestY = t * ray.m_Direction.y;
        float closestZ = t * ray.m_Direction.z;
        diffX = diffX - t * ray.m_Direction.x;
        diffY = diffY - closestY;
        diffZ = diffZ - closestZ;
    } else {
        t = 0.0f;
    }

    if (t0) *t0 = t;

    return diffZ * diffZ + diffX * diffX + diffY * diffY;
}

float VxDistance::PointSegmentSquareDistance(const VxVector &point, const VxRay &segment, float *t0) {
    // Compute difference
    float diffX = point.x - segment.m_Origin.x;
    float diffY = point.y - segment.m_Origin.y;
    float diffZ = point.z - segment.m_Origin.z;

    // dot(diff, dir)
    float dotDiffDir = diffZ * segment.m_Direction.z + diffY * segment.m_Direction.y + diffX * segment.m_Direction.x;

    float t;
    if (dotDiffDir > 0.0f) {
        // dot(dir, dir)
        float dotDirDir = segment.m_Direction.x * segment.m_Direction.x
                        + segment.m_Direction.y * segment.m_Direction.y
                        + segment.m_Direction.z * segment.m_Direction.z;

        if (dotDiffDir < dotDirDir) {
            // t is in (0, 1)
            t = dotDiffDir / dotDirDir;
            float closestY = t * segment.m_Direction.y;
            float closestZ = t * segment.m_Direction.z;
            diffX = diffX - t * segment.m_Direction.x;
            diffY = diffY - closestY;
            diffZ = diffZ - closestZ;
        } else {
            // t >= 1, clamp to 1
            diffX = diffX - segment.m_Direction.x;
            diffY = diffY - segment.m_Direction.y;
            t = 1.0f;
            diffZ = diffZ - segment.m_Direction.z;
        }
    } else {
        t = 0.0f;
    }

    if (t0) *t0 = t;

    return diffZ * diffZ + diffY * diffY + diffX * diffX;
}
