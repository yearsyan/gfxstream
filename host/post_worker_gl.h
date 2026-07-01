/*
 * Copyright (C) 2023 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <future>
#include <map>
#include <memory>
#include <optional>

#include "gfxstream/host/display_surface_user.h"
#include "post_worker.h"
#include "host/gl/display_gl.h"
#include "host/gl/emulation_gl.h"

#ifdef __APPLE__
#include "host/gl/iosurface_gl_export.h"
#endif

namespace gfxstream {
namespace host {
namespace gl {

class DisplayGl;
class EmulationGl;
class RecursiveScopedContextBind;

class PostWorkerGl : public PostWorker, public DisplaySurfaceUser {
   public:
    PostWorkerGl(bool mainThreadPostingOnly, FrameBuffer* fb, Compositor* compositor,
                 gl::DisplayGl* displayGl, gl::EmulationGl* emulationGl);

    protected:
    std::shared_future<void> postImpl(
        std::shared_ptr<ColorBuffer> cb, HandleType cbHandle,
        const std::optional<std::array<float, 16>>& colorTransform) override;
    void viewportImpl(int width, int height) override;
    void clearImpl() override;
    void exitImpl() override;
    std::shared_future<void> composeImpl(
        const FlatComposeRequest& composeRequest,
        const Post::ColorBufferRefMap& colorBufferRefs) override;
    ColorBuffer::UsedApi getColorBufferUsedApi() const override { return ColorBuffer::UsedApi::kGl; }

    void bindToSurfaceImpl(DisplaySurface* surface) override {}
    void surfaceUpdated(DisplaySurface* surface) override {}
    void unbindFromSurfaceImpl() override {}

   private:
    void setupContext();
    gl::DisplayGl::PostLayer postWithOverlay(
        ColorBuffer* cb, const std::optional<std::array<float, 16>>& colorTransform);
    void exportComposedDisplay(const FlatComposeRequest& composeRequest,
                               const Post::ColorBufferRefMap& colorBufferRefs);

   protected:
    void exportDisplayImpl(uint32_t displayId) override;

   private:
    // TODO(b/233939967): conslidate DisplayGl and DisplayVk into
    // `Display* const m_display`.
    gl::DisplayGl* const m_displayGl;

    int m_viewportWidth = 0;
    int m_viewportHeight = 0;

    bool mContextBound = false;
    std::unique_ptr<DisplaySurface> mFakeWindowSurface = nullptr;
    gl::EmulationGl* mEmulationGl;

#ifdef __APPLE__
    // MacMu per-display IOSurface export: one sink per secondary display,
    // publishing to that display's frame-channel slot right after its
    // composition lands in the display's target ColorBuffer. Display 0 is
    // exported by DisplayGl at final present instead. Post-thread only.
    std::map<uint32_t, std::unique_ptr<IosurfaceGlExportSink>> mIosurfaceComposeSinks;
#endif
};

}  // namespace gl
}  // namespace host
}  // namespace gfxstream
