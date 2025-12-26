#include "VxRect.h"

void VxRect::Transform(const VxRect &destScreen, const VxRect &srcScreen) {
    float srcScreenInvWidth = 1.0f / (srcScreen.right - srcScreen.left);
    float srcScreenInvHeight = 1.0f / (srcScreen.bottom - srcScreen.top);

    VxRect rect(
        (left - srcScreen.left) * srcScreenInvWidth,
        (top - srcScreen.top) * srcScreenInvHeight,
        (right - srcScreen.right) * srcScreenInvWidth,
        (bottom - srcScreen.bottom) * srcScreenInvHeight);

    float destScreenWidth = destScreen.right - destScreen.left;
    float destScreenHeight = destScreen.bottom - destScreen.top;

    left = rect.left * destScreenWidth + destScreen.left;
    top = rect.top * destScreenHeight + destScreen.top;
    right = rect.right * destScreenWidth + destScreen.right;
    bottom = rect.bottom * destScreenHeight + destScreen.bottom;
}

void VxRect::Transform(const Vx2DVector &destScreenSize, const Vx2DVector &srcScreenSize) {
    float srcScreenInvWidth = 1.0f / srcScreenSize.x;
    float srcScreenInvHeight = 1.0f / srcScreenSize.y;

    left *= srcScreenInvWidth * destScreenSize.x;
    top *= srcScreenInvHeight * destScreenSize.y;
    right *= srcScreenInvWidth * destScreenSize.x;
    bottom *= srcScreenInvHeight * destScreenSize.y;
}

void VxRect::TransformToHomogeneous(const VxRect &screen) {
    float screenInvWidth = 1.0f / (screen.right - screen.left);
    float screenInvHeight = 1.0f / (screen.bottom - screen.top);
    float width = right - left;
    float height = bottom - top;

    left = (left - screen.left) * screenInvWidth;
    top = (top - screen.top) * screenInvHeight;
    right = left + width * screenInvWidth;
    bottom = top + height * screenInvHeight;
}

void VxRect::TransformFromHomogeneous(const VxRect &screen) {
    float screenWidth = screen.right - screen.left;
    float screenHeight = screen.bottom - screen.top;
    float width = right - left;
    float height = bottom - top;

    left = screen.left + left * screenWidth;
    top = screen.top + top * screenHeight;
    right = left + width * screenWidth;
    bottom = top + height * screenHeight;
}
