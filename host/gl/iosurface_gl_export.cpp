// Copyright (C) 2026 The Android Open Source Project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifdef __APPLE__

#include "iosurface_gl_export.h"

#include <CoreFoundation/CoreFoundation.h>
#include <IOSurface/IOSurface.h>
#include <OpenGL/CGLCurrent.h>
#include <OpenGL/CGLIOSurface.h>

#include "borrowed_image_gl.h"
#include "OpenGLESDispatch/DispatchTables.h"
#include "gfxstream/common/logging.h"
#include "gfxstream/host/iosurface_export.h"

#ifndef GL_TEXTURE_RECTANGLE
#define GL_TEXTURE_RECTANGLE 0x84F5
#endif

#ifndef GL_TEXTURE_BINDING_RECTANGLE
#define GL_TEXTURE_BINDING_RECTANGLE 0x84F6
#endif

#ifndef GL_BGRA
#define GL_BGRA 0x80E1
#endif

#ifndef GL_UNSIGNED_INT_8_8_8_8_REV
#define GL_UNSIGNED_INT_8_8_8_8_REV 0x8367
#endif

namespace gfxstream {
namespace host {
namespace gl {

IosurfaceGlExportSink::IosurfaceGlExportSink(uint32_t displayId) : mDisplayId(displayId) {
    // Touch the shared producer once so channel setup logs appear near sink
    // creation instead of at first frame.
    FrameChannel::sharedProducer();
}

IosurfaceGlExportSink::~IosurfaceGlExportSink() { destroyTargets(); }

void IosurfaceGlExportSink::exportColorBuffer(ColorBuffer* colorBuffer, uint32_t targetWidth,
                                              uint32_t targetHeight) {
    FrameChannel* channel = FrameChannel::sharedProducer();
    uint64_t generation = 0;
    if (!channel || !channel->captureGeneration(mDisplayId, &generation)) {
        return;
    }
    exportColorBuffer(colorBuffer, targetWidth, targetHeight, generation);
}

void IosurfaceGlExportSink::exportColorBuffer(ColorBuffer* colorBuffer, uint32_t targetWidth,
                                              uint32_t targetHeight, uint64_t generation) {
    if (!gfxstream::host::isIosurfaceDisplayExportEnabled(mDisplayId) || !colorBuffer) {
        return;
    }
    const uint32_t sourceWidth = colorBuffer->getWidth();
    const uint32_t sourceHeight = colorBuffer->getHeight();
    const uint32_t width = targetWidth ? targetWidth : sourceWidth;
    const uint32_t height = targetHeight ? targetHeight : sourceHeight;
    if (sourceWidth == 0 || sourceHeight == 0 || width == 0 || height == 0) {
        return;
    }
    if (!ensureTargets(width, height)) {
        return;
    }
    blitColorBuffer(*colorBuffer, sourceWidth, sourceHeight, generation);
}

bool IosurfaceGlExportSink::ensureTargets(uint32_t width, uint32_t height) {
    if (mWidth == width && mHeight == height && targetsReady()) {
        return true;
    }

    destroyTargets();

    CGLContextObj context = CGLGetCurrentContext();
    if (!context) {
        GFXSTREAM_ERROR("MACMU_IOSURFACE_EXPORT display %u has no current CGL context.",
                        mDisplayId);
        return false;
    }

    mWidth = width;
    mHeight = height;
    mNextTargetIndex = 0;
    for (size_t i = 0; i < kTargetCount; ++i) {
        if (!createTarget(mTargets[i], context, width, height, i)) {
            destroyTargets();
            return false;
        }
    }
    GFXSTREAM_INFO(
        "MACMU_IOSURFACE_EXPORT GL display %u GPU-copy using %zu IOSurface buffers size=%ux%u",
        mDisplayId, kTargetCount, mWidth, mHeight);
    return true;
}

bool IosurfaceGlExportSink::createTarget(PresentTarget& target, void* cglContext, uint32_t width,
                                         uint32_t height, size_t index) {
    IOSurfaceRef surface =
        static_cast<IOSurfaceRef>(gfxstream::host::createBgra8Iosurface(width, height));
    if (!surface) {
        GFXSTREAM_ERROR("MACMU_IOSURFACE_EXPORT failed to create IOSurface.");
        return false;
    }

    GLint previousRectangleTexture = 0;
    ::glGetIntegerv(GL_TEXTURE_BINDING_RECTANGLE, &previousRectangleTexture);

    ::glGenTextures(1, &target.texture);
    ::glBindTexture(GL_TEXTURE_RECTANGLE, target.texture);
    const CGLError error = CGLTexImageIOSurface2D(static_cast<CGLContextObj>(cglContext),
                                                  GL_TEXTURE_RECTANGLE, GL_RGBA, width, height,
                                                  GL_BGRA, GL_UNSIGNED_INT_8_8_8_8_REV, surface, 0);
    ::glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    ::glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    ::glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    ::glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    ::glBindTexture(GL_TEXTURE_RECTANGLE, previousRectangleTexture);

    if (error != kCGLNoError) {
        GFXSTREAM_ERROR("MACMU_IOSURFACE_EXPORT failed to bind IOSurface texture: %d", error);
        CFRelease(surface);
        return false;
    }

    GLint previousHostDrawFbo = 0;
    ::glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &previousHostDrawFbo);
    ::glGenFramebuffers(1, &target.drawFbo);
    ::glBindFramebuffer(GL_DRAW_FRAMEBUFFER, target.drawFbo);
    ::glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_RECTANGLE,
                             target.texture, 0);
    const GLenum drawStatus = ::glCheckFramebufferStatus(GL_DRAW_FRAMEBUFFER);
    ::glBindFramebuffer(GL_DRAW_FRAMEBUFFER, previousHostDrawFbo);
    if (drawStatus != GL_FRAMEBUFFER_COMPLETE) {
        GFXSTREAM_ERROR("MACMU_IOSURFACE_EXPORT IOSurface framebuffer incomplete: 0x%x.",
                        drawStatus);
        CFRelease(surface);
        return false;
    }

    target.surface = surface;
    target.surfaceId = static_cast<uint32_t>(IOSurfaceGetID(surface));
    GFXSTREAM_INFO(
        "MACMU_IOSURFACE_EXPORT GL display %u GPU-copy IOSurface[%zu] id=%u size=%ux%u",
        mDisplayId, index, target.surfaceId, width, height);
    return true;
}

bool IosurfaceGlExportSink::targetsReady() const {
    for (const auto& target : mTargets) {
        if (!target.surface || target.surfaceId == 0 || target.texture == 0 ||
            target.drawFbo == 0) {
            return false;
        }
    }
    return true;
}

void IosurfaceGlExportSink::blitColorBuffer(ColorBuffer& colorBuffer, uint32_t sourceWidth,
                                            uint32_t sourceHeight, uint64_t generation) {
    if (!s_gles2.glBlitFramebuffer) {
        GFXSTREAM_ERROR("MACMU_IOSURFACE_EXPORT GL requires glBlitFramebuffer.");
        return;
    }
    PresentTarget& target = mTargets[mNextTargetIndex];

    // Borrow the ColorBuffer's live GL texture directly (same mechanism the GL
    // compositor uses); this stays valid on the post worker thread, unlike the
    // EGLImage rebinding path which requires a render thread.
    auto borrowed = colorBuffer.borrowForDisplay(ColorBuffer::UsedApi::kGl);
    if (!borrowed) {
        GFXSTREAM_ERROR("MACMU_IOSURFACE_EXPORT GL failed to borrow ColorBuffer texture.");
        return;
    }
    const auto* borrowedGl = static_cast<const BorrowedImageInfoGl*>(borrowed.get());
    if (borrowedGl->texture == 0) {
        GFXSTREAM_ERROR("MACMU_IOSURFACE_EXPORT GL borrowed ColorBuffer has no texture.");
        return;
    }

    GLint previousReadFbo = 0;
    GLint previousHostDrawFbo = 0;
    s_gles2.glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &previousReadFbo);
    ::glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &previousHostDrawFbo);

    if (!mReadFbo) {
        s_gles2.glGenFramebuffers(1, &mReadFbo);
    }

    s_gles2.glBindFramebuffer(GL_READ_FRAMEBUFFER, mReadFbo);
    s_gles2.glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                                   borrowedGl->texture, 0);
    ::glBindFramebuffer(GL_DRAW_FRAMEBUFFER, target.drawFbo);

    const GLenum readStatus = s_gles2.glCheckFramebufferStatus(GL_READ_FRAMEBUFFER);
    if (readStatus != GL_FRAMEBUFFER_COMPLETE) {
        GFXSTREAM_ERROR("MACMU_IOSURFACE_EXPORT GL read framebuffer incomplete: 0x%x.",
                        readStatus);
        restoreGlState(previousReadFbo, previousHostDrawFbo);
        return;
    }

    ::glBlitFramebuffer(0, 0, sourceWidth, sourceHeight, 0, 0, mWidth, mHeight,
                        GL_COLOR_BUFFER_BIT, GL_NEAREST);
    if (borrowedGl->onCommandsIssued) {
        borrowedGl->onCommandsIssued();
    }
    // The consumer maps the IOSurface as soon as the metadata is published, so
    // the copy must have fully landed in the surface before that.
    ::glFinish();

    ++mFrameNumber;
    publishMetadata(target, generation);
    mNextTargetIndex = (mNextTargetIndex + 1) % kTargetCount;
    restoreGlState(previousReadFbo, previousHostDrawFbo);
}

void IosurfaceGlExportSink::restoreGlState(GLint readFbo, GLint hostDrawFbo) {
    s_gles2.glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 0, 0);
    s_gles2.glBindFramebuffer(GL_READ_FRAMEBUFFER, readFbo);
    ::glBindFramebuffer(GL_DRAW_FRAMEBUFFER, hostDrawFbo);
}

void IosurfaceGlExportSink::publishMetadata(const PresentTarget& target, uint64_t generation) {
    if (!target.surface) {
        return;
    }
    // Only the shared-memory + socket doorbell channel is supported. If it is
    // unavailable there is no fallback, so drop the frame and log once so a
    // black screen stays diagnosable.
    if (FrameChannel* channel = FrameChannel::sharedProducer()) {
        channel->publish(mDisplayId, target.surfaceId, mWidth, mHeight, /*flags=*/0,
                         mFrameNumber, generation);
        return;
    }
    static bool logged = false;
    if (!logged) {
        logged = true;
        GFXSTREAM_ERROR("MACMU_IOSURFACE_EXPORT GL frame channel unavailable; dropping frames.");
    }
}

void IosurfaceGlExportSink::destroyTargets() {
    if (mReadFbo) {
        s_gles2.glDeleteFramebuffers(1, &mReadFbo);
        mReadFbo = 0;
    }
    for (auto& target : mTargets) {
        if (target.drawFbo) {
            ::glDeleteFramebuffers(1, &target.drawFbo);
            target.drawFbo = 0;
        }
        if (target.texture) {
            ::glDeleteTextures(1, &target.texture);
            target.texture = 0;
        }
        if (target.surface) {
            CFRelease(static_cast<IOSurfaceRef>(target.surface));
            target.surface = nullptr;
        }
        target.surfaceId = 0;
    }
    mWidth = 0;
    mHeight = 0;
    mNextTargetIndex = 0;
}

}  // namespace gl
}  // namespace host
}  // namespace gfxstream

#endif  // __APPLE__
