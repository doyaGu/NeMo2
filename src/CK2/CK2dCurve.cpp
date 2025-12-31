#include "CK2dCurve.h"
#include "CKStateChunk.h"

CKERROR CK2dCurve::GetPos(float step, Vx2DVector *pos) {
    if (!pos) return CKERR_INVALIDPARAMETER;
    if (m_ControlPoints.Size() < 2) return CKERR_INVALIDPARAMETER;

    // Clamp step value to [0, 1] range
    float clampedStep = (step > 1.0f) ? 1.0f : (step < 0.0f) ? 0.0f : step;
    float targetLength = clampedStep * m_Length;

    // Find the segment containing the target length
    int segmentIndex = 0;
    for (; segmentIndex < m_ControlPoints.Size() - 1; ++segmentIndex) {
        if (m_ControlPoints[segmentIndex + 1].m_Length > targetLength) {
            break;
        }
    }

    const CK2dCurvePoint &startPt = m_ControlPoints[segmentIndex];
    const CK2dCurvePoint &endPt = m_ControlPoints[segmentIndex + 1];

    // Calculate normalized t value within the segment
    float segmentLength = endPt.m_Length - startPt.m_Length;
    float t = (segmentLength > 0.0f) ? (targetLength - startPt.m_Length) / segmentLength : 0.0f;

    if (startPt.IsLinear()) {
        // Linear interpolation
        pos->x = startPt.m_Position.x + (endPt.m_Position.x - startPt.m_Position.x) * t;
        pos->y = startPt.m_Position.y + (endPt.m_Position.y - startPt.m_Position.y) * t;
    } else {
        // Cubic Hermite spline interpolation
        float t2 = t * t;
        float t3 = t2 * t;

        // Hermite basis functions
        float h1 = 2.0f * t3 - 3.0f * t2 + 1.0f;
        float h2 = -2.0f * t3 + 3.0f * t2;
        float h3 = t3 - 2.0f * t2 + t;
        float h4 = t3 - t2;

        // Calculate interpolated position
        pos->x = h1 * startPt.m_Position.x +
            h2 * endPt.m_Position.x +
            h3 * startPt.m_OutTang.x +
            h4 * endPt.m_InTang.x;

        pos->y = h1 * startPt.m_Position.y +
            h2 * endPt.m_Position.y +
            h3 * startPt.m_OutTang.y +
            h4 * endPt.m_InTang.y;
    }

    return CK_OK;
}

float CK2dCurve::GetY(float x) {
    const int MAX_ITERATIONS = 1000;
    const float TOLERANCE = 1e-5f;

    if (m_ControlPoints.Size() < 2)
        return 0.0f;

    // Clamp input to valid range [0, 1]
    x = (x > 1.0f) ? 1.0f : (x < 0.0f) ? 0.0f : x;

    // Find containing segment
    int segment = 0;
    for (; segment < m_ControlPoints.Size() - 1; ++segment) {
        if (m_ControlPoints[segment + 1].m_Position.x >= x) {
            break;
        }
    }

    const CK2dCurvePoint &start = m_ControlPoints[segment];
    const CK2dCurvePoint &end = m_ControlPoints[segment + 1];

    // Handle exact point matches
    if (x == start.m_Position.x) return start.m_Position.y;
    if (x == end.m_Position.x) return end.m_Position.y;

    // Linear segment calculation
    if (start.IsLinear()) {
        float t = (x - start.m_Position.x) / (end.m_Position.x - start.m_Position.x);
        return start.m_Position.y + t * (end.m_Position.y - start.m_Position.y);
    }

    // Cubic spline parameter estimation using bisection method
    float tMin = 0.0f;
    float tMax = 1.0f;
    float t = 0.5f;
    float currentX = 0.0f;
    float y = 0.0f;

    for (int i = 0; i < MAX_ITERATIONS; ++i) {
        // Calculate cubic Hermite interpolation
        float t2 = t * t;
        float t3 = t2 * t;

        // Hermite basis functions
        float h1 = 2.0f * t3 - 3.0f * t2 + 1.0f;
        float h2 = -2.0f * t3 + 3.0f * t2;
        float h3 = t3 - 2.0f * t2 + t;
        float h4 = t3 - t2;

        // Calculate X and Y coordinates
        currentX = h1 * start.m_Position.x +
            h2 * end.m_Position.x +
            h3 * start.m_OutTang.x +
            h4 * end.m_InTang.x;

        y = h1 * start.m_Position.y +
            h2 * end.m_Position.y +
            h3 * start.m_OutTang.y +
            h4 * end.m_InTang.y;

        // Check convergence
        if (fabs(currentX - x) < TOLERANCE) break;

        // Adjust bisection interval
        if (currentX < x) {
            tMin = t;
        } else {
            tMax = t;
        }
        t = (tMin + tMax) * 0.5f;
    }

    // Clamp result to [0, 1] as final safeguard
    return (y > 1.0f) ? 1.0f : (y < 0.0f) ? 0.0f : y;
}

void CK2dCurve::AddControlPoint(const Vx2DVector &pos) {
    CK2dCurvePoint pt;
    pt.m_Curve = this;
    pt.m_Position = pos;
    m_ControlPoints.PushBack(pt);
    pt.NotifyUpdate();
}

void CK2dCurve::Update() {
    const int SAMPLES_PER_SEGMENT = 100;
    const float SAMPLE_STEP = 1.0f / SAMPLES_PER_SEGMENT;
    const unsigned char LINEAR_FLAG = 0x02;

    // Sort control points by X position
    m_ControlPoints.BubbleSort(CK2dCurvePoint::CurvePointSortFunc);

    // Update tangent vectors and intermediate positions
    UpdatePointsAndTangents();

    const int pointCount = m_ControlPoints.Size();
    if (pointCount < 2)
        return;

    // Initialize length tracking
    m_Length = 0.0f;
    m_ControlPoints[0].m_Length = 0.0f;

    // Calculate length for each curve segment
    for (int i = 0; i < pointCount - 1; ++i) {
        CK2dCurvePoint &current = m_ControlPoints[i];
        CK2dCurvePoint &next = m_ControlPoints[i + 1];

        if (current.m_Flags & LINEAR_FLAG) {
            // Straight line segment calculation
            Vx2DVector delta(next.m_Position - current.m_Position);
            m_Length += delta.Magnitude();
        } else {
            // Cubic spline length approximation
            Vx2DVector prevPos = current.m_Position;
            for (int s = 1; s <= SAMPLES_PER_SEGMENT; ++s) {
                float t = s * SAMPLE_STEP;
                float t2 = t * t;
                float t3 = t2 * t;

                // Cubic interpolation coefficients
                float h1 = 2 * t3 - 3 * t2 + 1;
                float h2 = -2 * t3 + 3 * t2;
                float h3 = t3 - 2 * t2 + t;
                float h4 = t3 - t2;

                // Calculate interpolated position
                Vx2DVector position =
                    current.m_Position * h1 +
                    next.m_Position * h2 +
                    current.m_OutTang * h3 +
                    next.m_InTang * h4;

                // Accumulate segment length
                Vx2DVector delta(position - prevPos);
                m_Length += delta.Magnitude();
                prevPos = position;
            }
        }

        // Store accumulated length in next point
        if (i < pointCount - 1) {
            next.m_Length = m_Length;
        }
    }
}

CK2dCurve::CK2dCurve() {
    m_Length = 0.0f;
    FittingCoef = 0.0f;
    AddControlPoint(Vx2DVector(0.0f, 0.0f));
    AddControlPoint(Vx2DVector(1.0f, 1.0f));
}

CKStateChunk *CK2dCurve::Dump() {
    CKStateChunk *chunk = CreateCKStateChunk(CKCHNK_2DCURVE, nullptr);
    chunk->StartWrite();
    chunk->SetDataVersion(CHUNK_DEV_2_1);
    chunk->WriteIdentifier(0x100);

    XSObjectArray objects;
    objects.Resize(m_ControlPoints.Size());
    objects.Fill(0);
    objects.Save(chunk, nullptr);

    chunk->WriteFloat(FittingCoef);
    for (XArray<CK2dCurvePoint>::Iterator it = m_ControlPoints.Begin(); it != m_ControlPoints.End(); it++) {
        chunk->WriteDword(it->m_Flags | 0x10000000);
        chunk->WriteFloat(it->m_Position.x);
        chunk->WriteFloat(it->m_Position.y);
        chunk->WriteFloat(it->m_Tension);
        chunk->WriteFloat(it->m_Continuity);
        chunk->WriteFloat(it->m_Bias);
        chunk->WriteFloat(0.0f);
        chunk->WriteFloat(0.0f);
        chunk->WriteFloat(it->m_InTang.x);
        chunk->WriteFloat(it->m_InTang.y);
        chunk->WriteFloat(it->m_OutTang.x);
        chunk->WriteFloat(it->m_OutTang.y);
    }

    chunk->CloseChunk();
    return chunk;
}

CKERROR CK2dCurve::Read(CKStateChunk *chunk) {
    if (!chunk)
        return CKERR_INVALIDPARAMETER;

    chunk->StartRead();

    // Handle legacy format versions
    if (chunk->GetDataVersion() < 1) {
        // Read fitting coefficient if present
        if (chunk->SeekIdentifier(CK_STATESAVE_2DCURVEFITCOEFF)) {
            FittingCoef = chunk->ReadFloat();
        }

        // Read control points array
        if (chunk->SeekIdentifier(CK_STATESAVE_2DCURVESAVEPOINTS)) {
            const int pointCount = chunk->ReadInt();
            m_ControlPoints.Resize(pointCount);

            for (int i = 0; i < pointCount; ++i) {
                // Read control point data
                chunk->ReadObjectID(); // Legacy object ID (ignored)
                CKStateChunk *subChunk = chunk->ReadSubChunk();

                m_ControlPoints[i].m_Curve = this;
                m_ControlPoints[i].Read(subChunk);

                if (subChunk) {
                    delete subChunk;
                }
            }
        }
    }

    // Modern format handling (version >= 1)
    if (chunk->SeekIdentifier(CK_STATESAVE_2DCURVENEWDATA)) {
        // Read control points array
        const XObjectArray &objArray = chunk->ReadXObjectArray();
        int pointCount = objArray.Size();
        m_ControlPoints.Resize(pointCount);

        // Read curve parameters
        FittingCoef = chunk->ReadFloat();

        // Read control point data
        for (int i = 0; i < pointCount; ++i) {
            CK2dCurvePoint &point = m_ControlPoints[i];
            point.m_Curve = this;

            // Read point flags
            point.m_Flags = chunk->ReadDword();

            if (point.m_Flags & 0x10000000) {
                // Clear legacy flags
                point.m_Flags &= ~0x10000000;

                // Read position and spline parameters
                point.m_Position.x = chunk->ReadFloat();
                point.m_Position.y = chunk->ReadFloat();
                point.m_Tension = chunk->ReadFloat();
                point.m_Continuity = chunk->ReadFloat();
                point.m_Bias = chunk->ReadFloat();

                // Read legacy values (no longer used)
                chunk->ReadFloat(); // Was m_Length
                chunk->ReadFloat(); // Was m_Time

                // Read tangent vectors
                point.m_InTang.x = chunk->ReadFloat();
                point.m_InTang.y = chunk->ReadFloat();
                point.m_OutTang.x = chunk->ReadFloat();
                point.m_OutTang.y = chunk->ReadFloat();
            }
        }
    }

    // Update curve calculations
    Update();
    return CK_OK;
}

void CK2dCurve::UpdatePointsAndTangents() {
    int pointCount = m_ControlPoints.Size();

    if (pointCount < 2)
        return;

    // Process first control point
    CK2dCurvePoint &firstPt = m_ControlPoints[0];
    CK2dCurvePoint &secondPt = m_ControlPoints[1];

    float deltaX = secondPt.m_Position.x - firstPt.m_Position.x;
    float deltaY = secondPt.m_Position.y - firstPt.m_Position.y;

    if (!firstPt.IsTCB()) {
        // Calculate tangents using TCB parameters
        float t = 1.0f - firstPt.m_Tension;
        float c = 1.0f - firstPt.m_Continuity;
        float b = 1.0f - firstPt.m_Bias;

        // Out tangent calculation
        float outFactor = c * t * b;
        firstPt.m_OutTang.x = (deltaX * outFactor) * 0.5f;
        firstPt.m_OutTang.y = (deltaY * outFactor) * 0.5f;

        // In tangent calculation
        float inFactor = (c + 1.0f) * t * b;
        firstPt.m_InTang.x = (deltaX * inFactor) * 0.5f;
        firstPt.m_InTang.y = (deltaY * inFactor) * 0.5f;
    }

    // Prevent division by zero
    if (fabs(firstPt.m_InTang.x) < EPSILON) {
        float scale = EPSILON / firstPt.m_InTang.x;
        firstPt.m_InTang *= scale;
    }

    if (fabs(firstPt.m_OutTang.x) < EPSILON) {
        float scale = deltaX / firstPt.m_OutTang.x;
        firstPt.m_OutTang *= scale;
    }

    // Calculate fitted position
    float midX = (firstPt.m_Position.x + secondPt.m_Position.x) * 0.5f;
    firstPt.m_RCurvePos.x = firstPt.m_Position.x + (midX - firstPt.m_Position.x) * FittingCoef;
    firstPt.m_RCurvePos.y = firstPt.m_Position.y +
        ((firstPt.m_Position.y + secondPt.m_Position.y) * 0.5f - firstPt.m_Position.y) * FittingCoef;

    // Process intermediate points
    for (int i = 1; i < pointCount - 1; ++i) {
        CK2dCurvePoint &prevPt = m_ControlPoints[i - 1];
        CK2dCurvePoint &currPt = m_ControlPoints[i];
        CK2dCurvePoint &nextPt = m_ControlPoints[i + 1];

        float prevDeltaX = currPt.m_Position.x - prevPt.m_Position.x;
        float prevDeltaY = currPt.m_Position.y - prevPt.m_Position.y;
        float nextDeltaX = nextPt.m_Position.x - currPt.m_Position.x;
        float nextDeltaY = nextPt.m_Position.y - currPt.m_Position.y;

        if (!currPt.IsTCB()) {
            // TCB calculations
            float t = 1.0f - currPt.m_Tension;
            float c = 1.0f - currPt.m_Continuity;
            float b = 1.0f - currPt.m_Bias;

            // Out tangent
            float outFactor = c * t * (b + 1.0f);
            currPt.m_OutTang.x = (nextDeltaX * outFactor + prevDeltaX * c * t * b) * 0.5f;
            currPt.m_OutTang.y = (nextDeltaY * outFactor + prevDeltaY * c * t * b) * 0.5f;

            // In tangent
            float inFactor = (c + 1.0f) * t * b;
            currPt.m_InTang.x = (prevDeltaX * inFactor + nextDeltaX * (c + 1.0f) * t * (b + 1.0f)) * 0.5f;
            currPt.m_InTang.y = (prevDeltaY * inFactor + nextDeltaY * (c + 1.0f) * t * (b + 1.0f)) * 0.5f;
        }

        // Normalize tangents
        if (fabs(prevDeltaX) < EPSILON) prevDeltaX = EPSILON;
        if (fabs(currPt.m_InTang.x) < EPSILON) {
            float scale = prevDeltaX / currPt.m_InTang.x;
            currPt.m_InTang *= scale;
        }

        if (fabs(nextDeltaX) < EPSILON) nextDeltaX = EPSILON;
        if (fabs(currPt.m_OutTang.x) < EPSILON) {
            float scale = nextDeltaX / currPt.m_OutTang.x;
            currPt.m_OutTang *= scale;
        }

        // Update fitted position
        float midX = (prevPt.m_Position.x + nextPt.m_Position.x) * 0.5f;
        currPt.m_RCurvePos.x = currPt.m_Position.x + (midX - currPt.m_Position.x) * FittingCoef;
        currPt.m_RCurvePos.y = currPt.m_Position.y +
            ((prevPt.m_Position.y + nextPt.m_Position.y) * 0.5f - currPt.m_Position.y) * FittingCoef;
    }

    // Process last control point
    CK2dCurvePoint &lastPt = m_ControlPoints[pointCount - 1];
    CK2dCurvePoint &prevLastPt = m_ControlPoints[pointCount - 2];

    float lastDeltaX = lastPt.m_Position.x - prevLastPt.m_Position.x;
    float lastDeltaY = lastPt.m_Position.y - prevLastPt.m_Position.y;

    if (!lastPt.IsTCB()) {
        float t = 1.0f - lastPt.m_Tension;
        float c = 1.0f - lastPt.m_Continuity;
        float b = 1.0f - lastPt.m_Bias;

        // In tangent calculation
        float inFactor = (c + 1.0f) * t * (b + 1.0f);
        lastPt.m_InTang.x = (lastDeltaX * inFactor) * 0.5f;
        lastPt.m_InTang.y = (lastDeltaY * inFactor) * 0.5f;

        // Out tangent calculation
        float outFactor = c * t * b;
        lastPt.m_OutTang.x = (lastDeltaX * outFactor) * 0.5f;
        lastPt.m_OutTang.y = (lastDeltaY * outFactor) * 0.5f;
    }

    // Normalize last point tangents
    if (fabs(lastDeltaX) < EPSILON) lastDeltaX = EPSILON;
    if (fabs(lastPt.m_InTang.x) < EPSILON) {
        float scale = lastDeltaX / lastPt.m_InTang.x;
        lastPt.m_InTang *= scale;
    }

    // Calculate final fitted position
    float lastMidX = (prevLastPt.m_Position.x + lastPt.m_Position.x) * 0.5f;
    lastPt.m_RCurvePos.x = lastPt.m_Position.x + (lastMidX - lastPt.m_Position.x) * FittingCoef;
    lastPt.m_RCurvePos.y = lastPt.m_Position.y +
        ((prevLastPt.m_Position.y + lastPt.m_Position.y) * 0.5f - lastPt.m_Position.y) * FittingCoef;
}
