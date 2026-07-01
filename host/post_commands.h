// Copyright 2025 The Android Open Source Project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either expresso or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <array>
#include <functional>
#include <future>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

#include "gfxstream/host/display_operations.h"
#include "gfxstream/host/gfxstream_format.h"
#include "handle.h"
#include "render-utils/Renderer.h"

namespace gfxstream {
namespace host {

class ColorBuffer;

// Posting
enum class PostCmd {
    Post = 0,
    Viewport = 1,
    Compose = 2,
    Clear = 3,
    Screenshot = 4,
    Exit = 5,
    Block = 6,
    // MacMu: export one display's bound ColorBuffer to its IOSurface
    // frame-channel slot (fired per guest frame via display buffer re-binds).
    MacMuExportDisplay = 7,
};

struct Post {
    using ColorBufferRefMap = std::unordered_map<HandleType, std::shared_ptr<ColorBuffer>>;

    struct Block {
        // schduledSignal will be set when the block task is scheduled.
        std::promise<void> scheduledSignal;
        // The block task won't stop until continueSignal is ready.
        std::future<void> continueSignal;
    };
    using CompletionCallback =
        std::function<void(std::shared_future<void> waitForGpu)>;
    PostCmd cmd;
    int composeVersion;
    std::vector<char> composeBuffer;
    std::unique_ptr<CompletionCallback> completionCallback = nullptr;
    std::unique_ptr<Block> block = nullptr;
    HandleType cbHandle = 0;
    std::shared_ptr<ColorBuffer> cbRef = nullptr;
    ColorBufferRefMap colorBufferRefs;
    std::optional<std::array<float, 16>> colorTransform;

    //TODO: remove union here and separate into message structures
    union {
        ColorBuffer* cb;
        struct {
            int width;
            int height;
        } viewport;
        struct {
            uint32_t displayId;
        } exportDisplay;
        struct {
            ColorBuffer* cb;
            int screenwidth;
            int screenheight;
            int rotation;
            GfxstreamFormat pixelsFormat;
            void* pixels;
            Rect rect;
        } screenshot;
    };
};

}  // namespace host
}  // namespace gfxstream
