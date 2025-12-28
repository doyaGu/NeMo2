#include "VxDistance.h"

#include <float.h>

#include "VxRay.h"
#include "VxVector.h"

struct RayPairTerms {
    VxVector diff;
    float a;   // |dir0|^2
    float b;   // -dot(dir0, dir1)
    float c;   // |dir1|^2
    float d;   // dot(diff, dir0)
    float e;   // -dot(diff, dir1)
    float f;   // |diff|^2
    float det; // |a*c - b*b|
};

static inline RayPairTerms MakeRayPairTerms(const VxRay& r0, const VxRay& r1) {
    RayPairTerms t;
    t.diff = r0.m_Origin - r1.m_Origin;
    t.a = SquareMagnitude(r0.m_Direction);
    t.b = -DotProduct(r0.m_Direction, r1.m_Direction);
    t.c = SquareMagnitude(r1.m_Direction);
    t.d = DotProduct(t.diff, r0.m_Direction);
    t.e = -DotProduct(t.diff, r1.m_Direction);
    t.f = SquareMagnitude(t.diff);
    t.det = XAbs(t.c * t.a - t.b * t.b);
    return t;
}

// Matches the exact quadratic expansion used in the original implementation.
static inline float DistanceQuadraticPart(const RayPairTerms& t, float s0, float s1) {
    return (s1 * t.c + s0 * t.b + t.e + t.e) * s1 + (s1 * t.b + s0 * t.a + t.d + t.d) * s0;
}

// Line-Line distance calculations

float VxDistance::LineLineSquareDistance(const VxRay &line0, const VxRay &line1, float *t0, float *t1) {
    const RayPairTerms t = MakeRayPairTerms(line0, line1);

    float s0;
    float s1;
    float quad;

    if (t.det < EPSILON) {
        // Parallel case
        s1 = 0.0f;
        s0 = -(t.d / t.a);
        quad = t.d * s0;
    } else {
        const float invDet = 1.0f / t.det;
        s0 = (t.e * t.b - t.d * t.c) * invDet;
        s1 = (t.d * t.b - t.e * t.a) * invDet;
        quad = DistanceQuadraticPart(t, s0, s1);
    }

    if (t0) *t0 = s0;
    if (t1) *t1 = s1;

    return XAbs(quad + t.f);
}

float VxDistance::LineRaySquareDistance(const VxRay &line, const VxRay &ray, float *t0, float *t1) {
    const RayPairTerms t = MakeRayPairTerms(line, ray);

    float s0;
    float s1;
    float quad;

    const float rayParam = t.d * t.b - t.e * t.a;
    if (t.det < EPSILON || rayParam < 0.0f) {
        // Parallel case or ray parameter negative - clamp ray parameter to 0
        s1 = 0.0f;
        s0 = -(t.d / t.a);
        quad = t.d * s0;
    } else {
        const float invDet = 1.0f / t.det;
        s0 = (t.e * t.b - t.d * t.c) * invDet;
        s1 = invDet * rayParam;
        quad = DistanceQuadraticPart(t, s0, s1);
    }

    if (t0) *t0 = s0;
    if (t1) *t1 = s1;

    return XAbs(quad + t.f);
}

float VxDistance::LineSegmentSquareDistance(const VxRay &line, const VxRay &segment, float *t0, float *t1) {
    const RayPairTerms t = MakeRayPairTerms(line, segment);

    float s0;
    float s1;
    float quad;

    const float segParam = t.d * t.b - t.e * t.a;
    if (t.det < EPSILON || segParam < 0.0f) {
        // Parallel case or segment parameter negative - clamp to 0
        s1 = 0.0f;
        s0 = -(t.d / t.a);
        quad = t.d * s0;
        quad = quad + t.f;
    } else if (segParam > t.det) {
        // Segment parameter > 1 - clamp to 1
        const float dPlusB = t.d + t.b;
        s1 = 1.0f;
        s0 = -(dPlusB / t.a);
        quad = dPlusB * s0 + t.e + t.e + t.f + t.c;
    } else {
        const float invDet = 1.0f / t.det;
        s0 = (t.e * t.b - t.d * t.c) * invDet;
        s1 = invDet * segParam;
        quad = DistanceQuadraticPart(t, s0, s1) + t.f;
    }

    if (t0) *t0 = s0;
    if (t1) *t1 = s1;

    return XAbs(quad);
}

float VxDistance::RayRaySquareDistance(const VxRay &ray0, const VxRay &ray1, float *t0, float *t1) {
    const RayPairTerms t = MakeRayPairTerms(ray0, ray1);

    float s0 = 0.0f;
    float s1 = 0.0f;
    float f = t.f;

    if (t.det < EPSILON) {
        // Parallel case
        s1 = 0.0f;
        if (t.d < 0.0f) {
            s0 = -(t.d / t.a);
            f = f + s0 * t.d;
        } else {
            s0 = 0.0f;
            s1 = -(t.e / t.c);
            f = f + s1 * t.e;
        }
    } else {
        const float ray0Param = t.e * t.b - t.d * t.c;
        const float ray1Param = t.d * t.b - t.e * t.a;

        if (ray0Param < 0.0f) {
            // Clamp ray0 parameter to 0
            s0 = 0.0f;
            if (t.d < 0.0f) {
                // Closest point involves ray1 at t=0
                s1 = 0.0f;
                s0 = -(t.d / t.a);
                f = f + s0 * t.d;
            } else {
                // Optimize ray1 for ray0 at t=0
                if (t.e >= 0.0f) {
                    s1 = 0.0f;
                } else {
                    s1 = -(t.e / t.c);
                    f = f + s1 * t.e;
                }
            }
        } else if (ray1Param < 0.0f) {
            // Clamp ray1 parameter to 0
            s1 = 0.0f;
            if (t.d < 0.0f) {
                s0 = -(t.d / t.a);
                f = f + s0 * t.d;
            } else {
                s0 = 0.0f;
            }
        } else {
            // Both parameters in range
            const float invDet = 1.0f / t.det;
            s0 = ray0Param * invDet;
            s1 = ray1Param * invDet;
            f = f + DistanceQuadraticPart(t, s0, s1);
        }
    }

    if (t0) *t0 = s0;
    if (t1) *t1 = s1;

    return XAbs(f);
}

float VxDistance::RaySegmentSquareDistance(const VxRay &ray, const VxRay &segment, float *t0, float *t1) {
    const RayPairTerms t = MakeRayPairTerms(ray, segment);
    const float a = t.a;
    const float b = t.b;
    const float c = t.c;
    const float d = t.d;
    const float e = t.e;
    float f = t.f;
    const float det = t.det;

    float rayT = 0.0f;
    float segT = 0.0f;

    if (det < EPSILON) {
        // Parallel case
        if (b <= 0.0f) {
            // Evaluate with segment clamped at 1
            segT = 1.0f;
            const float dPlusB = d + b;
            if (dPlusB >= 0.0f) {
                rayT = 0.0f;
                f = f + e + e + c;
            } else {
                rayT = -(dPlusB / a);
                f = f + dPlusB * rayT + e + e + c;
            }
        } else {
            // Evaluate with segment clamped at 0
            segT = 0.0f;
            if (d < 0.0f) {
                rayT = -(d / a);
                f = f + rayT * d;
            } else {
                rayT = 0.0f;
            }
        }
    } else {
        const float rayParam = e * b - d * c;
        const float segParam = d * b - e * a;

        if (segParam < 0.0f) {
            // Clamp segment to 0
            segT = 0.0f;
            if (d < 0.0f) {
                rayT = -(d / a);
                f = f + rayT * d;
            } else {
                rayT = 0.0f;
            }
        } else if (segParam > det) {
            // Clamp segment to 1
            segT = 1.0f;
            if (-b <= d) {
                rayT = 0.0f;
                f = f + e + e + c;
            } else {
                const float dPlusB = d + b;
                rayT = -(dPlusB / a);
                f = f + dPlusB * rayT + e + e + c;
            }
        } else if (rayParam < 0.0f) {
            // Clamp ray to 0 and optimize segment parameter
            rayT = 0.0f;
            if (e >= 0.0f) {
                segT = 0.0f;
            } else if (-e > c) {
                segT = 1.0f;
                f = f + e + e + c;
            } else {
                segT = -(e / c);
                f = f + segT * e;
            }
        } else {
            // Both parameters in range
            const float invDet = 1.0f / det;
            rayT = rayParam * invDet;
            segT = segParam * invDet;
            f = f + DistanceQuadraticPart(t, rayT, segT);
        }
    }

    if (t0) *t0 = rayT;
    if (t1) *t1 = segT;

    return XAbs(f);
}

float VxDistance::SegmentSegmentSquareDistance(const VxRay &segment0, const VxRay &segment1, float *t0, float *t1) {
    const RayPairTerms t = MakeRayPairTerms(segment0, segment1);
    const float a = t.a;
    const float b = t.b;
    const float c = t.c;
    const float d = t.d;
    const float e = t.e;
    float f = t.f;
    const float det = t.det;

    float s0, s1;

    if (det < EPSILON) {
        // Parallel case: choose the minimum endpoint-to-segment distance.
        // This correctly handles collinear overlap (distance = 0) and disjoint parallel segments.

        const VxVector p0 = segment0.m_Origin;
        const VxVector q0 = segment0.m_Origin + segment0.m_Direction;
        const VxVector p1 = segment1.m_Origin;
        const VxVector q1 = segment1.m_Origin + segment1.m_Direction;

        float bestDist = FLT_MAX;
        float bestS0 = 0.0f;
        float bestS1 = 0.0f;

        // p0 -> segment1
        {
            float tParam = 0.0f;
            float distVal = PointSegmentSquareDistance(p0, segment1, &tParam);
            if (distVal < bestDist) {
                bestDist = distVal;
                bestS0 = 0.0f;
                bestS1 = tParam;
            }
        }

        // q0 -> segment1
        {
            float tParam = 0.0f;
            float distVal = PointSegmentSquareDistance(q0, segment1, &tParam);
            if (distVal < bestDist) {
                bestDist = distVal;
                bestS0 = 1.0f;
                bestS1 = tParam;
            }
        }

        // p1 -> segment0
        {
            float tParam = 0.0f;
            float distVal = PointSegmentSquareDistance(p1, segment0, &tParam);
            if (distVal < bestDist) {
                bestDist = distVal;
                bestS0 = tParam;
                bestS1 = 0.0f;
            }
        }

        // q1 -> segment0
        {
            float tParam = 0.0f;
            float distVal = PointSegmentSquareDistance(q1, segment0, &tParam);
            if (distVal < bestDist) {
                bestDist = distVal;
                bestS0 = tParam;
                bestS1 = 1.0f;
            }
        }

        s0 = bestS0;
        s1 = bestS1;
        f = bestDist;
    } else {
        // Non-parallel case
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
    const VxVector diff = point - line.m_Origin;
    const float t = DotProduct(diff, line.m_Direction) / SquareMagnitude(line.m_Direction);

    if (t0) *t0 = t;

    const VxVector d = diff - (line.m_Direction * t);
    return SquareMagnitude(d);
}

float VxDistance::PointRaySquareDistance(const VxVector &point, const VxRay &ray, float *t0) {
    VxVector diff = point - ray.m_Origin;
    const float dotDiffDir = DotProduct(diff, ray.m_Direction);

    float t;
    if (dotDiffDir > 0.0f) {
        t = dotDiffDir / SquareMagnitude(ray.m_Direction);
        diff -= ray.m_Direction * t;
    } else {
        t = 0.0f;
    }

    if (t0) *t0 = t;

    return SquareMagnitude(diff);
}

float VxDistance::PointSegmentSquareDistance(const VxVector &point, const VxRay &segment, float *t0) {
    VxVector diff = point - segment.m_Origin;
    const float dotDiffDir = DotProduct(diff, segment.m_Direction);

    float t;
    if (dotDiffDir > 0.0f) {
        const float dotDirDir = SquareMagnitude(segment.m_Direction);
        if (dotDiffDir < dotDirDir) {
            t = dotDiffDir / dotDirDir;
            diff -= segment.m_Direction * t;
        } else {
            // clamp to segment end
            t = 1.0f;
            diff -= segment.m_Direction;
        }
    } else {
        t = 0.0f;
    }

    if (t0) *t0 = t;

    return SquareMagnitude(diff);
}
