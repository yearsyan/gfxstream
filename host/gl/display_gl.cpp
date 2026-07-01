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

#include <array>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "borrowed_image_gl.h"
#include "display_surface_gl.h"
#include "OpenGLESDispatch/DispatchTables.h"
#include "OpenGLESDispatch/EGLDispatch.h"
#include "texture_draw.h"
#include "gfxstream/common/logging.h"
#include "gfxstream/host/display_operations.h"
#include "gfxstream/host/iosurface_export.h"
#include "gfxstream/system/System.h"

#ifdef __APPLE__
#include "iosurface_gl_export.h"
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

using gfxstream::host::isIosurfaceExportEnabled;

}  // namespace

#ifdef __APPLE__

// Thin display-0 adapter over the shared per-display GL export sink: DisplayGl
// presents only the primary display, so this publishes to frame-channel
// slot 0. Secondary displays are exported from PostWorkerGl after their
// per-display composition.
class IosurfaceGlDisplaySink {
  public:
    IosurfaceGlDisplaySink() : mSink(/*displayId=*/0) {}

    void post(const DisplayGl::Post& post) {
        if (post.layers.empty()) {
            return;
        }
        if (post.layers.size() > 1 && !mLoggedMultiLayerWarning) {
            GFXSTREAM_WARNING(
                "MACMU_IOSURFACE_EXPORT GL sink exports only the first display layer.");
            mLoggedMultiLayerWarning = true;
        }
        if (post.colorTransform.has_value() && !mLoggedTransformWarning) {
            GFXSTREAM_WARNING("MACMU_IOSURFACE_EXPORT GL sink ignores color transform.");
            mLoggedTransformWarning = true;
        }

        const auto& layer = post.layers[0];
        uint32_t targetWidth = layer.colorBuffer ? layer.colorBuffer->getWidth() : 0;
        uint32_t targetHeight = layer.colorBuffer ? layer.colorBuffer->getHeight() : 0;
        if (layer.layerOptions) {
            const auto& frame = layer.layerOptions->displayFrame;
            const int frameWidth = frame.right - frame.left;
            const int frameHeight = frame.bottom - frame.top;
            if (frameWidth > 0 && frameHeight > 0) {
                targetWidth = static_cast<uint32_t>(frameWidth);
                targetHeight = static_cast<uint32_t>(frameHeight);
            }
        }
        mSink.exportColorBuffer(layer.colorBuffer, targetWidth, targetHeight);
    }

  private:
    IosurfaceGlExportSink mSink;
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
