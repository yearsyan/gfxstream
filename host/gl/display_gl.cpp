// Copyright (C) 2022 The Android Open Source Project
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

#include "display_gl.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

#include "display_surface_gl.h"
#include "OpenGLESDispatch/DispatchTables.h"
#include "OpenGLESDispatch/EGLDispatch.h"
#include "texture_draw.h"
#include "gfxstream/common/logging.h"
#include "gfxstream/host/display_operations.h"

#ifdef __APPLE__
#include <CoreFoundation/CoreFoundation.h>
#include <IOSurface/IOSurface.h>
#include <OpenGL/CGLCurrent.h>
#include <OpenGL/CGLIOSurface.h>
#endif

namespace gfxstream {
namespace host {
namespace gl {
namespace {

std::shared_future<void> getCompletedFuture() {
    std::shared_future<void> completedFuture =
        std::async(std::launch::deferred, [] {}).share();
    completedFuture.wait();
    return completedFuture;
}

bool isIosurfaceExportEnabled() {
    const char* value = std::getenv("MACMU_IOSURFACE_EXPORT");
    if (!value) {
        value = std::getenv("AEMU_IOSURFACE_EXPORT");
    }
    if (!value) {
        return false;
    }
    const std::string enabled(value);
    return enabled == "1" || enabled == "true" || enabled == "TRUE" || enabled == "yes" ||
           enabled == "YES";
}

}  // namespace

#ifdef __APPLE__

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

class IosurfaceGlDisplaySink {
  public:
    IosurfaceGlDisplaySink() {
        if (const char* path = std::getenv("MACMU_IOSURFACE_EXPORT_PATH")) {
            mMetadataPath = path;
        } else if (const char* path = std::getenv("AEMU_IOSURFACE_EXPORT_PATH")) {
            mMetadataPath = path;
        } else {
            mMetadataPath = "/tmp/macmu-iosurface.json";
        }
    }

    ~IosurfaceGlDisplaySink() { destroyTarget(); }

    void post(const DisplayGl::Post& post) {
        if (post.layers.empty()) {
            return;
        }
        if (post.layers.size() > 1 && !mLoggedMultiLayerWarning) {
            GFXSTREAM_WARNING(
                "MACMU_IOSURFACE_EXPORT GL prototype exports only the first display layer.");
            mLoggedMultiLayerWarning = true;
        }
        if (post.colorTransform.has_value() && !mLoggedTransformWarning) {
            GFXSTREAM_WARNING("MACMU_IOSURFACE_EXPORT GL prototype ignores color transform.");
            mLoggedTransformWarning = true;
        }

        const auto& layer = post.layers[0];
        ColorBuffer* colorBuffer = layer.colorBuffer;
        if (!colorBuffer) {
            return;
        }
        if ((layer.layerOptions.has_value() || layer.overlayOptions.has_value() ||
             layer.colorTransform.has_value()) &&
            !mLoggedTransformWarning) {
            GFXSTREAM_WARNING(
                "MACMU_IOSURFACE_EXPORT GL prototype exports the raw first ColorBuffer.");
            mLoggedTransformWarning = true;
        }

        const uint32_t sourceWidth = colorBuffer->getWidth();
        const uint32_t sourceHeight = colorBuffer->getHeight();
        const uint32_t targetWidth = post.frameWidth ? post.frameWidth : sourceWidth;
        const uint32_t targetHeight = post.frameHeight ? post.frameHeight : sourceHeight;
        if (sourceWidth == 0 || sourceHeight == 0 || targetWidth == 0 || targetHeight == 0) {
            return;
        }
        if (!ensureTarget(targetWidth, targetHeight)) {
            return;
        }
        blitColorBuffer(*colorBuffer, sourceWidth, sourceHeight);
    }

  private:
    static void setDictionaryNumber(CFMutableDictionaryRef dictionary, CFStringRef key,
                                    int64_t value) {
        CFNumberRef number = CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt64Type, &value);
        if (number) {
            CFDictionarySetValue(dictionary, key, number);
            CFRelease(number);
        }
    }

    bool ensureTarget(uint32_t width, uint32_t height) {
        if (mSurface && mWidth == width && mHeight == height) {
            return true;
        }

        destroyTarget();

        CGLContextObj context = CGLGetCurrentContext();
        if (!context) {
            GFXSTREAM_ERROR("MACMU_IOSURFACE_EXPORT has no current CGL context.");
            return false;
        }

        const int64_t bytesPerElement = 4;
        const int64_t bytesPerRow = static_cast<int64_t>(width) * bytesPerElement;
        const int64_t allocSize = bytesPerRow * static_cast<int64_t>(height);
        constexpr int64_t kPixelFormatBGRA = 0x42475241;  // 'BGRA'

        CFMutableDictionaryRef properties =
            CFDictionaryCreateMutable(kCFAllocatorDefault, 0, &kCFTypeDictionaryKeyCallBacks,
                                      &kCFTypeDictionaryValueCallBacks);
        if (!properties) {
            GFXSTREAM_ERROR("MACMU_IOSURFACE_EXPORT failed to create IOSurface properties.");
            return false;
        }
        setDictionaryNumber(properties, kIOSurfaceWidth, width);
        setDictionaryNumber(properties, kIOSurfaceHeight, height);
        setDictionaryNumber(properties, kIOSurfaceBytesPerElement, bytesPerElement);
        setDictionaryNumber(properties, kIOSurfaceBytesPerRow, bytesPerRow);
        setDictionaryNumber(properties, kIOSurfaceAllocSize, allocSize);
        setDictionaryNumber(properties, kIOSurfacePixelFormat, kPixelFormatBGRA);
        CFDictionarySetValue(properties, kIOSurfaceIsGlobal, kCFBooleanTrue);

        IOSurfaceRef surface = IOSurfaceCreate(properties);
        CFRelease(properties);
        if (!surface) {
            GFXSTREAM_ERROR("MACMU_IOSURFACE_EXPORT failed to create IOSurface.");
            return false;
        }

        if (!mSurfaceTexture) {
            ::glGenTextures(1, &mSurfaceTexture);
        }
        if (!mReadFbo) {
            s_gles2.glGenFramebuffers(1, &mReadFbo);
        }
        if (!mDrawFbo) {
            ::glGenFramebuffers(1, &mDrawFbo);
        }
        if (!mSourceTexture) {
            s_gles2.glGenTextures(1, &mSourceTexture);
        }

        GLint previousRectangleTexture = 0;
        ::glGetIntegerv(GL_TEXTURE_BINDING_RECTANGLE, &previousRectangleTexture);

        ::glBindTexture(GL_TEXTURE_RECTANGLE, mSurfaceTexture);
        CGLError error = CGLTexImageIOSurface2D(context, GL_TEXTURE_RECTANGLE, GL_RGBA, width,
                                               height, GL_BGRA,
                                               GL_UNSIGNED_INT_8_8_8_8_REV, surface, 0);
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

        mSurface = surface;
        mSurfaceId = IOSurfaceGetID(mSurface);
        mWidth = width;
        mHeight = height;
        GFXSTREAM_INFO("MACMU_IOSURFACE_EXPORT GL publishing IOSurface id=%u size=%ux%u to %s",
                       static_cast<uint32_t>(mSurfaceId), mWidth, mHeight, mMetadataPath.c_str());
        publishMetadata();
        return true;
    }

    void blitColorBuffer(ColorBuffer& colorBuffer, uint32_t sourceWidth, uint32_t sourceHeight) {
        if (!s_gles2.glBlitFramebuffer) {
            GFXSTREAM_ERROR("MACMU_IOSURFACE_EXPORT GL requires glBlitFramebuffer.");
            return;
        }

        GLint previousActiveTexture = GL_TEXTURE0;
        GLint previousTexture2d = 0;
        GLint previousReadFbo = 0;
        GLint previousHostTextureRectangle = 0;
        GLint previousHostDrawFbo = 0;
        s_gles2.glGetIntegerv(GL_ACTIVE_TEXTURE, &previousActiveTexture);
        s_gles2.glActiveTexture(GL_TEXTURE0);
        s_gles2.glGetIntegerv(GL_TEXTURE_BINDING_2D, &previousTexture2d);
        s_gles2.glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &previousReadFbo);
        ::glGetIntegerv(GL_TEXTURE_BINDING_RECTANGLE, &previousHostTextureRectangle);
        ::glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &previousHostDrawFbo);

        s_gles2.glBindTexture(GL_TEXTURE_2D, mSourceTexture);
        s_gles2.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        s_gles2.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        s_gles2.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        s_gles2.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        if (!colorBuffer.glOpBindToTexture()) {
            GFXSTREAM_ERROR("MACMU_IOSURFACE_EXPORT GL failed to bind ColorBuffer texture.");
            restoreGlState(previousActiveTexture, previousTexture2d, previousReadFbo,
                           previousHostTextureRectangle, previousHostDrawFbo);
            return;
        }

        s_gles2.glBindFramebuffer(GL_READ_FRAMEBUFFER, mReadFbo);
        s_gles2.glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                                       mSourceTexture, 0);
        ::glBindTexture(GL_TEXTURE_RECTANGLE, mSurfaceTexture);
        ::glBindFramebuffer(GL_DRAW_FRAMEBUFFER, mDrawFbo);
        ::glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_RECTANGLE,
                                 mSurfaceTexture, 0);

        const GLenum readStatus = s_gles2.glCheckFramebufferStatus(GL_READ_FRAMEBUFFER);
        const GLenum drawStatus = ::glCheckFramebufferStatus(GL_DRAW_FRAMEBUFFER);
        if (readStatus != GL_FRAMEBUFFER_COMPLETE || drawStatus != GL_FRAMEBUFFER_COMPLETE) {
            GFXSTREAM_ERROR(
                "MACMU_IOSURFACE_EXPORT GL framebuffer incomplete read=0x%x draw=0x%x.",
                readStatus, drawStatus);
            restoreGlState(previousActiveTexture, previousTexture2d, previousReadFbo,
                           previousHostTextureRectangle, previousHostDrawFbo);
            return;
        }

        ::glBlitFramebuffer(0, 0, sourceWidth, sourceHeight, 0, 0, mWidth, mHeight,
                            GL_COLOR_BUFFER_BIT, GL_NEAREST);
        ::glFlush();

        ++mFrameNumber;
        publishMetadata();
        restoreGlState(previousActiveTexture, previousTexture2d, previousReadFbo,
                       previousHostTextureRectangle, previousHostDrawFbo);
    }

    void restoreGlState(GLint activeTexture, GLint texture2d, GLint readFbo,
                        GLint hostTextureRectangle, GLint hostDrawFbo) {
        s_gles2.glBindFramebuffer(GL_READ_FRAMEBUFFER, readFbo);
        s_gles2.glBindTexture(GL_TEXTURE_2D, texture2d);
        ::glBindFramebuffer(GL_DRAW_FRAMEBUFFER, hostDrawFbo);
        ::glBindTexture(GL_TEXTURE_RECTANGLE, hostTextureRectangle);
        s_gles2.glActiveTexture(activeTexture);
    }

    void publishMetadata() {
        if (!mSurface) {
            return;
        }
        const auto now = std::chrono::steady_clock::now().time_since_epoch();
        const auto nowNs = std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
        const std::string tmpPath = mMetadataPath + ".tmp";
        std::ofstream out(tmpPath, std::ios::out | std::ios::trunc);
        if (!out) {
            GFXSTREAM_WARNING("MACMU_IOSURFACE_EXPORT could not write metadata file %s",
                              mMetadataPath.c_str());
            return;
        }
        out << "{\n"
            << "  \"iosurface_id\": " << static_cast<uint32_t>(mSurfaceId) << ",\n"
            << "  \"width\": " << mWidth << ",\n"
            << "  \"height\": " << mHeight << ",\n"
            << "  \"pixel_format\": \"BGRA8Unorm\",\n"
            << "  \"frame\": " << mFrameNumber << ",\n"
            << "  \"timestamp_ns\": " << nowNs << "\n"
            << "}\n";
        out.close();
        std::rename(tmpPath.c_str(), mMetadataPath.c_str());
    }

    void destroyTarget() {
        if (mReadFbo) {
            s_gles2.glDeleteFramebuffers(1, &mReadFbo);
            mReadFbo = 0;
        }
        if (mDrawFbo) {
            ::glDeleteFramebuffers(1, &mDrawFbo);
            mDrawFbo = 0;
        }
        if (mSourceTexture) {
            s_gles2.glDeleteTextures(1, &mSourceTexture);
            mSourceTexture = 0;
        }
        if (mSurfaceTexture) {
            ::glDeleteTextures(1, &mSurfaceTexture);
            mSurfaceTexture = 0;
        }
        if (mSurface) {
            CFRelease(mSurface);
            mSurface = nullptr;
        }
        mSurfaceId = 0;
        mWidth = 0;
        mHeight = 0;
    }

    std::string mMetadataPath;
    IOSurfaceRef mSurface = nullptr;
    IOSurfaceID mSurfaceId = 0;
    uint32_t mWidth = 0;
    uint32_t mHeight = 0;
    GLuint mSurfaceTexture = 0;
    GLuint mSourceTexture = 0;
    GLuint mReadFbo = 0;
    GLuint mDrawFbo = 0;
    uint64_t mFrameNumber = 0;
    bool mLoggedMultiLayerWarning = false;
    bool mLoggedTransformWarning = false;
};

#else

class IosurfaceGlDisplaySink {};

#endif  // __APPLE__

DisplayGl::DisplayGl(TextureDraw* textureDraw) : mTextureDraw(textureDraw) {}

DisplayGl::~DisplayGl() = default;

std::shared_future<void> DisplayGl::post(const Post& post) {
    if (isIosurfaceExportEnabled()) {
#ifdef __APPLE__
        if (!mIosurfaceSink) {
            mIosurfaceSink = std::make_unique<IosurfaceGlDisplaySink>();
        }
        mIosurfaceSink->post(post);
#else
        static bool logged = false;
        if (!logged) {
            GFXSTREAM_ERROR("MACMU_IOSURFACE_EXPORT is only supported on macOS.");
            logged = true;
        }
#endif
    }

    const auto* surface = getBoundSurface();
    if (!surface) {
        return getCompletedFuture();
    }
    if (post.layers.empty()) {
        clear();
        return getCompletedFuture();
    }
    const auto* surfaceGl = static_cast<const DisplaySurfaceGl*>(surface->getImpl());

    bool hasDrawLayer = false;
    for (const PostLayer& layer : post.layers) {
        if (layer.layerOptions) {
            if (!hasDrawLayer) {
                mTextureDraw->prepareForDrawLayer();
                hasDrawLayer = true;
            }
            layer.colorBuffer->glOpPostLayer(*layer.layerOptions, post.frameWidth,
                                             post.frameHeight, post.colorTransform);
        } else if (layer.overlayOptions) {
            if (hasDrawLayer) {
                GFXSTREAM_ERROR("Cannot mix colorBuffer.postLayer with postWithOverlay!");
            }

            layer.colorBuffer->glOpPostViewportScaledWithOverlay(
                layer.overlayOptions->rotation,
                layer.overlayOptions->dx, layer.overlayOptions->dy,
                layer.overlayOptions->scaleX, layer.overlayOptions->scaleY,
                layer.colorTransform);
        }
    }
    if (hasDrawLayer) {
        mTextureDraw->cleanupForDrawLayer();
    }

    s_egl.eglSwapBuffers(surfaceGl->mDisplay, surfaceGl->mSurface);

    return getCompletedFuture();
}

void DisplayGl::viewport(int width, int height) {
    mViewportWidth = width;
    mViewportHeight = height;
    s_gles2.glViewport(0, 0, mViewportWidth, mViewportHeight);
}

void DisplayGl::clear() {
    const auto* surface = getBoundSurface();
    if (!surface) {
        return;
    }
    const auto* surfaceGl = static_cast<const DisplaySurfaceGl*>(surface->getImpl());
#ifndef __linux__
    s_gles2.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
    s_egl.eglSwapBuffers(surfaceGl->mDisplay, surfaceGl->mSurface);
#else
    (void)surfaceGl;
#endif
}

}  // namespace gl
}  // namespace host
}  // namespace gfxstream
