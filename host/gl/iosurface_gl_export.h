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

// Per-display GL IOSurface export sink for MacMu.
//
// One instance exports one display's frames: it owns a small ring of
// IOSurface-backed GL framebuffers, GPU-blits a source ColorBuffer into the
// next ring entry, and publishes the entry's IOSurface id to the display's
// frame-channel slot (see gfxstream/host/iosurface_export.h).
//
// Callers must invoke exportColorBuffer() on a thread with a current GL
// context (the post worker thread): DisplayGl uses it at final present for
// display 0, PostWorkerGl uses it after per-display composition for
// display ids != 0.

#pragma once

#ifdef __APPLE__

#include <GLES2/gl2.h>

#include <array>
#include <cstdint>

#include "color_buffer.h"

namespace gfxstream {
namespace host {
namespace gl {

class IosurfaceGlExportSink {
  public:
    explicit IosurfaceGlExportSink(uint32_t displayId);
    ~IosurfaceGlExportSink();

    IosurfaceGlExportSink(const IosurfaceGlExportSink&) = delete;
    IosurfaceGlExportSink& operator=(const IosurfaceGlExportSink&) = delete;

    // GPU-blit |colorBuffer| into the next IOSurface ring entry sized
    // |targetWidth| x |targetHeight| (recreated on size change) and publish it
    // to this display's frame-channel slot. Requires a current GL context.
    void exportColorBuffer(ColorBuffer* colorBuffer, uint32_t targetWidth, uint32_t targetHeight);
    // Variant for callers that capture the display lifecycle before selecting
    // |colorBuffer|. This prevents a retained buffer from the previous display
    // instance from being published after the id is reused.
    void exportColorBuffer(ColorBuffer* colorBuffer, uint32_t targetWidth, uint32_t targetHeight,
                           uint64_t generation);

   private:
    // The consumer (shell) samples the newest published IOSurface while the
    // next frame is being produced, so rotate between multiple targets to keep
    // writes off the surface currently on screen.
    static constexpr size_t kTargetCount = 3;

    struct PresentTarget {
        void* surface = nullptr;  // IOSurfaceRef, kept opaque here (see the
                                  // ALIGN macro note in iosurface_export.h)
        uint32_t surfaceId = 0;
        GLuint texture = 0;  // host GL name, GL_TEXTURE_RECTANGLE
        GLuint drawFbo = 0;  // host GL name
    };

    bool ensureTargets(uint32_t width, uint32_t height);
    bool createTarget(PresentTarget& target, void* cglContext, uint32_t width, uint32_t height,
                      size_t index);
    bool targetsReady() const;
    void blitColorBuffer(ColorBuffer& colorBuffer, uint32_t sourceWidth, uint32_t sourceHeight,
                         uint64_t generation);
    void restoreGlState(GLint readFbo, GLint hostDrawFbo);
    void publishMetadata(const PresentTarget& target, uint64_t generation);
    void destroyTargets();

    const uint32_t mDisplayId;
    std::array<PresentTarget, kTargetCount> mTargets;
    size_t mNextTargetIndex = 0;
    uint32_t mWidth = 0;
    uint32_t mHeight = 0;
    GLuint mReadFbo = 0;
    uint64_t mFrameNumber = 0;
};

}  // namespace gl
}  // namespace host
}  // namespace gfxstream

#endif  // __APPLE__
