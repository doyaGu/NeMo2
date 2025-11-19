#include "VxDistance.h"

#include "VxRay.h"
#include "VxVector.h"

// Line-Line distance calculations

float VxDistance::LineLineSquareDistance(const VxRay &line0, const VxRay &line1, float *t0, float *t1) {
    // Direction vectors
    const VxVector &dir0 = line0.GetDirection();
    const VxVector &dir1 = line1.GetDirection();

    // Get the segment between the two origins
    VxVector diff = line0.GetOrigin() - line1.GetOrigin();

    float a = DotProduct(dir0, dir0); // Length squared of line0.dir
    float b = DotProduct(dir0, dir1); // Dot product of the two directions
    float c = DotProduct(dir1, dir1); // Length squared of line1.dir
    float d = DotProduct(dir0, diff); // Projection of diff onto line0.dir
    float e = DotProduct(dir1, diff); // Projection of diff onto line1.dir

    float det = a * c - b * b; // Determinant of the system

    // Default parameters
    float s0, s1;

    if (XAbs(det) < EPSILON) {
        // Lines are parallel - project one point onto the other line
        s0 = 0.0f;
        if (XAbs(c) > EPSILON) {
            s1 = e / c;
        } else {
            s1 = 0.0f;
        }
    } else {
        // Compute the parameters of the closest points
        s0 = (b * e - c * d) / det;
        s1 = (a * e - b * d) / det;
    }

    // Store parameters if requested
    if (t0) *t0 = s0;
    if (t1) *t1 = s1;

    // Compute the closest points on the lines
    VxVector closest0, closest1;
    line0.Interpolate(closest0, s0);
    line1.Interpolate(closest1, s1);

    // Return the squared distance between the closest points
    return SquareMagnitude(closest1 - closest0);
}

float VxDistance::LineRaySquareDistance(const VxRay &line, const VxRay &ray, float *t0, float *t1) {
    // Direction vectors
    const VxVector &dir0 = line.GetDirection();
    const VxVector &dir1 = ray.GetDirection();

    // Get the segment between the two origins
    VxVector diff = line.GetOrigin() - ray.GetOrigin();

    float a = DotProduct(dir0, dir0);
    float b = DotProduct(dir0, dir1);
    float c = DotProduct(dir1, dir1);
    float d = DotProduct(dir0, diff);
    float e = DotProduct(dir1, diff);

    float det = a * c - b * b;

    // Default parameters
    float s0, s1;

    if (XAbs(det) < EPSILON) {
        // Lines are parallel
        s0 = 0.0f;
        if (XAbs(c) > EPSILON) {
            s1 = e / c;
        } else {
            s1 = 0.0f;
        }
    } else {
        // Compute the unconstrained parameters
        s0 = (b * e - c * d) / det;
        s1 = (a * e - b * d) / det;
    }

    // For a ray, parameter must be non-negative
    if (s1 < 0.0f) {
        s1 = 0.0f;
        // Recompute s0 with s1 fixed at 0
        if (XAbs(a) > EPSILON) {
            s0 = -d / a;
        } else {
            s0 = 0.0f;
        }
    }

    // Store parameters if requested
    if (t0) *t0 = s0;
    if (t1) *t1 = s1;

    // Compute closest points
    VxVector closest0, closest1;
    line.Interpolate(closest0, s0);
    ray.Interpolate(closest1, s1);

    // Return the squared distance between the closest points
    return SquareMagnitude(closest1 - closest0);
}

float VxDistance::LineSegmentSquareDistance(const VxRay &line, const VxRay &segment, float *t0, float *t1) {
    // Direction vectors
    const VxVector &dir0 = line.GetDirection();
    const VxVector &dir1 = segment.GetDirection();

    // Get the segment between the two origins
    VxVector diff = line.GetOrigin() - segment.GetOrigin();

    float a = DotProduct(dir0, dir0);
    float b = DotProduct(dir0, dir1);
    float c = DotProduct(dir1, dir1);
    float d = DotProduct(dir0, diff);
    float e = DotProduct(dir1, diff);

    float det = a * c - b * b;

    // Default parameters
    float s0, s1;

    if (XAbs(det) < EPSILON) {
        // Lines are parallel
        s0 = 0.0f;
        if (XAbs(c) > EPSILON) {
            s1 = e / c;
        } else {
            s1 = 0.0f;
        }
    } else {
        // Compute the unconstrained parameters
        s0 = (b * e - c * d) / det;
        s1 = (a * e - b * d) / det;
    }

    // For a segment, parameter must be between 0 and 1
    if (s1 < 0.0f) {
        s1 = 0.0f;
        // Recompute s0 with s1 fixed at 0
        if (XAbs(a) > EPSILON) {
            s0 = -d / a;
        } else {
            s0 = 0.0f;
        }
    } else if (s1 > 1.0f) {
        s1 = 1.0f;
        // Recompute s0 with s1 fixed at 1
        if (XAbs(a) > EPSILON) {
            s0 = (b - d) / a;
        } else {
            s0 = 0.0f;
        }
    }

    // Store parameters if requested
    if (t0) *t0 = s0;
    if (t1) *t1 = s1;

    // Compute closest points
    VxVector closest0, closest1;
    line.Interpolate(closest0, s0);
    segment.Interpolate(closest1, s1);

    // Return the squared distance between the closest points
    return SquareMagnitude(closest1 - closest0);
}

float VxDistance::RayRaySquareDistance(const VxRay &ray0, const VxRay &ray1, float *t0, float *t1) {
    // Direction vectors
    const VxVector &dir0 = ray0.GetDirection();
    const VxVector &dir1 = ray1.GetDirection();

    // Get the segment between the two origins
    VxVector diff = ray0.GetOrigin() - ray1.GetOrigin();

    float a = DotProduct(dir0, dir0);
    float b = DotProduct(dir0, dir1);
    float c = DotProduct(dir1, dir1);
    float d = DotProduct(dir0, diff);
    float e = DotProduct(dir1, diff);

    float det = a * c - b * b;

    // Default parameters
    float s0, s1;

    if (XAbs(det) < EPSILON) {
        // Rays are parallel
        s0 = 0.0f;
        if (XAbs(c) > EPSILON) {
            s1 = e / c;
        } else {
            s1 = 0.0f;
        }
    } else {
        // Compute the unconstrained parameters
        s0 = (b * e - c * d) / det;
        s1 = (a * e - b * d) / det;
    }

    // For rays, both parameters must be non-negative
    // Handle constraints systematically
    if (s0 < 0.0f && s1 < 0.0f) {
        // Both parameters negative - closest points are at origins
        s0 = 0.0f;
        s1 = 0.0f;
    } else if (s0 < 0.0f) {
        // s0 is negative, clamp to 0 and recompute s1
        s0 = 0.0f;
        if (XAbs(c) > EPSILON) {
            s1 = e / c;
            if (s1 < 0.0f) s1 = 0.0f;
        } else {
            s1 = 0.0f;
        }
    } else if (s1 < 0.0f) {
        // s1 is negative, clamp to 0 and recompute s0
        s1 = 0.0f;
        if (XAbs(a) > EPSILON) {
            s0 = -d / a;
            if (s0 < 0.0f) s0 = 0.0f;
        } else {
            s0 = 0.0f;
        }
    }

    // Store parameters if requested
    if (t0) *t0 = s0;
    if (t1) *t1 = s1;

    // Compute closest points
    VxVector closest0, closest1;
    ray0.Interpolate(closest0, s0);
    ray1.Interpolate(closest1, s1);

    // Return the squared distance between the closest points
    return SquareMagnitude(closest1 - closest0);
}

float VxDistance::RaySegmentSquareDistance(const VxRay &ray, const VxRay &segment, float *t0, float *t1) {
    // Direction vectors
    const VxVector &dir0 = ray.GetDirection();
    const VxVector &dir1 = segment.GetDirection();

    // Get the segment between the two origins
    VxVector diff = ray.GetOrigin() - segment.GetOrigin();

    float a = DotProduct(dir0, dir0);
    float b = DotProduct(dir0, dir1);
    float c = DotProduct(dir1, dir1);
    float d = DotProduct(dir0, diff);
    float e = DotProduct(dir1, diff);

    float det = a * c - b * b;

    // Default parameters
    float s0, s1;

    if (XAbs(det) < EPSILON) {
        // Lines are parallel
        s0 = 0.0f;
        if (XAbs(c) > EPSILON) {
            s1 = e / c;
        } else {
            s1 = 0.0f;
        }
    } else {
        // Compute the unconstrained parameters
        s0 = (b * e - c * d) / det;
        s1 = (a * e - b * d) / det;
    }

    // Apply constraints: ray.t >= 0, 0 <= segment.t <= 1
    if (s0 < 0.0f) {
        s0 = 0.0f;
        // Recompute s1 with s0 fixed at 0
        if (XAbs(c) > EPSILON) {
            s1 = e / c;
        } else {
            s1 = 0.0f;
        }
    }

    if (s1 < 0.0f) {
        s1 = 0.0f;
        // Recompute s0 with s1 fixed at 0
        if (XAbs(a) > EPSILON) {
            s0 = -d / a;
            if (s0 < 0.0f) s0 = 0.0f;
        } else {
            s0 = 0.0f;
        }
    } else if (s1 > 1.0f) {
        s1 = 1.0f;
        // Recompute s0 with s1 fixed at 1
        if (XAbs(a) > EPSILON) {
            s0 = (b - d) / a;
            if (s0 < 0.0f) s0 = 0.0f;
        } else {
            s0 = 0.0f;
        }
    }

    // Store parameters if requested
    if (t0) *t0 = s0;
    if (t1) *t1 = s1;

    // Compute closest points
    VxVector closest0, closest1;
    ray.Interpolate(closest0, s0);
    segment.Interpolate(closest1, s1);

    // Return the squared distance between the closest points
    return SquareMagnitude(closest1 - closest0);
}

float VxDistance::SegmentSegmentSquareDistance(const VxRay &segment0, const VxRay &segment1, float *t0, float *t1) {
    // Direction vectors
    const VxVector &dir0 = segment0.GetDirection();
    const VxVector &dir1 = segment1.GetDirection();

    // Get the segment between the two origins
    VxVector diff = segment0.GetOrigin() - segment1.GetOrigin();

    float a = DotProduct(dir0, dir0);
    float b = DotProduct(dir0, dir1);
    float c = DotProduct(dir1, dir1);
    float d = DotProduct(dir0, diff);
    float e = DotProduct(dir1, diff);

    float det = a * c - b * b;

    // Default parameters
    float s0, s1;

    if (XAbs(det) < EPSILON) {
        // Lines are parallel
        s0 = 0.0f;
        if (XAbs(c) > EPSILON) {
            s1 = e / c;
        } else {
            s1 = 0.0f;
        }
    } else {
        // Compute the unconstrained parameters
        s0 = (b * e - c * d) / det;
        s1 = (a * e - b * d) / det;
    }

    // For segments, both parameters must be between 0 and 1
    // Apply constraints systematically
    if (s0 < 0.0f) {
        s0 = 0.0f;
        // Recompute s1 with s0 fixed at 0
        if (XAbs(c) > EPSILON) {
            s1 = e / c;
            // Clamp s1 to [0,1]
            if (s1 < 0.0f) s1 = 0.0f;
            else if (s1 > 1.0f) s1 = 1.0f;
        } else {
            s1 = 0.0f;
        }
    } else if (s0 > 1.0f) {
        s0 = 1.0f;
        // Recompute s1 with s0 fixed at 1
        if (XAbs(c) > EPSILON) {
            s1 = (e + b) / c;
            // Clamp s1 to [0,1]
            if (s1 < 0.0f) s1 = 0.0f;
            else if (s1 > 1.0f) s1 = 1.0f;
        } else {
            s1 = 0.0f;
        }
    } else {
        // s0 is in [0,1], check s1
        if (s1 < 0.0f) {
            s1 = 0.0f;
            // Recompute s0 with s1 fixed at 0
            if (XAbs(a) > EPSILON) {
                s0 = -d / a;
                // Clamp s0 to [0,1]
                if (s0 < 0.0f) s0 = 0.0f;
                else if (s0 > 1.0f) s0 = 1.0f;
            } else {
                s0 = 0.0f;
            }
        } else if (s1 > 1.0f) {
            s1 = 1.0f;
            // Recompute s0 with s1 fixed at 1
            if (XAbs(a) > EPSILON) {
                s0 = (b - d) / a;
                // Clamp s0 to [0,1]
                if (s0 < 0.0f) s0 = 0.0f;
                else if (s0 > 1.0f) s0 = 1.0f;
            } else {
                s0 = 0.0f;
            }
        }
        // If both s0 and s1 are in range, we're done
    }

    // Store parameters if requested
    if (t0) *t0 = s0;
    if (t1) *t1 = s1;

    // Compute closest points
    VxVector closest0, closest1;
    segment0.Interpolate(closest0, s0);
    segment1.Interpolate(closest1, s1);

    // Return the squared distance between the closest points
    return SquareMagnitude(closest1 - closest0);
}

// Point-Line distance calculations

float VxDistance::PointLineSquareDistance(const VxVector &point, const VxRay &line, float *t0) {
    // Direction vector
    const VxVector &dir = line.GetDirection();

    // Vector from line origin to point
    VxVector diff = point - line.GetOrigin();

    // Dot products
    float a = DotProduct(dir, dir);  // Length squared of line.dir
    float b = DotProduct(dir, diff); // Projection of diff onto line.dir

    // Parameter of the closest point on the line
    float s = 0.0f;
    if (XAbs(a) > EPSILON) {
        s = b / a;
    }

    // Store parameter if requested
    if (t0) *t0 = s;

    // Compute closest point on the line
    VxVector closest;
    line.Interpolate(closest, s);

    // Return the squared distance to the closest point
    return SquareMagnitude(point - closest);
}

float VxDistance::PointRaySquareDistance(const VxVector &point, const VxRay &ray, float *t0) {
    // Direction vector
    const VxVector &dir = ray.GetDirection();

    // Vector from ray origin to point
    VxVector diff = point - ray.GetOrigin();

    // Dot products
    float a = DotProduct(dir, dir);  // Length squared of ray.dir
    float b = DotProduct(dir, diff); // Projection of diff onto ray.dir

    // Parameter of the closest point on the ray (constrained to be >= 0)
    float s = 0.0f;
    if (XAbs(a) > EPSILON) {
        s = b / a;
        if (s < 0.0f) s = 0.0f;
    }

    // Store parameter if requested
    if (t0) *t0 = s;

    // Compute closest point on the ray
    VxVector closest;
    ray.Interpolate(closest, s);

    // Return the squared distance to the closest point
    return SquareMagnitude(point - closest);
}

float VxDistance::PointSegmentSquareDistance(const VxVector &point, const VxRay &segment, float *t0) {
    // Direction vector
    const VxVector &dir = segment.GetDirection();

    // Vector from segment origin to point
    VxVector diff = point - segment.GetOrigin();

    // Dot products
    float a = DotProduct(dir, dir);  // Length squared of segment.dir
    float b = DotProduct(dir, diff); // Projection of diff onto segment.dir

    // Parameter of the closest point on the segment (constrained to be between 0 and 1)
    float s = 0.0f;
    if (XAbs(a) > EPSILON) {
        s = b / a;
        if (s < 0.0f) s = 0.0f;
        else if (s > 1.0f) s = 1.0f;
    }

    // Store parameter if requested
    if (t0) *t0 = s;

    // Compute the closest point on the segment
    VxVector closest;
    segment.Interpolate(closest, s);

    // Return the squared distance to the closest point
    return SquareMagnitude(point - closest);
}
