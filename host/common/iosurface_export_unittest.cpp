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

#include "gfxstream/host/iosurface_export.h"

#include <sys/socket.h>
#include <unistd.h>

#include <cstdlib>
#include <iostream>
#include <string>

namespace gfxstream {
namespace host {
namespace {

class ScopedFrameChannelEnvironment {
   public:
    ScopedFrameChannelEnvironment() {
        if (socketpair(AF_UNIX, SOCK_DGRAM, 0, mDoorbellFds) != 0) {
            return;
        }
        setenv("MACMU_IOSURFACE_EXPORT", "1", 1);
        const std::string fd = std::to_string(mDoorbellFds[1]);
        setenv("MACMU_FRAME_DOORBELL_FD", fd.c_str(), 1);
        resetIosurfaceDisplayExportSubscriptions();
    }

    ~ScopedFrameChannelEnvironment() {
        resetIosurfaceDisplayExportSubscriptions();
        unsetenv("MACMU_FRAME_DOORBELL_FD");
        unsetenv("MACMU_IOSURFACE_EXPORT");
        for (int& fd : mDoorbellFds) {
            if (fd >= 0) {
                close(fd);
                fd = -1;
            }
        }
    }

    bool valid() const { return mDoorbellFds[0] >= 0 && mDoorbellFds[1] >= 0; }

   private:
    int mDoorbellFds[2] = {-1, -1};
};

bool require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "gfxstream_iosurface_export_unittests: FAIL: " << message << '\n';
        return false;
    }
    return true;
}

bool runDisplayReuseTest() {
    ScopedFrameChannelEnvironment environment;
    if (!require(environment.valid(), "failed to create frame-channel doorbell")) return false;

    const uint32_t wrapperPid = static_cast<uint32_t>(getpid());
    auto consumer = FrameChannel::createConsumer(wrapperPid);
    if (!require(consumer && consumer->valid(), "failed to create frame-channel consumer")) {
        return false;
    }
    auto producer = FrameChannel::createProducer(wrapperPid);
    if (!require(producer && producer->valid(), "failed to create frame-channel producer")) {
        return false;
    }

    constexpr uint32_t kDisplayId = 1;
    setIosurfaceDisplayExportEnabled(kDisplayId, true);

    uint64_t generationA = 0;
    if (!require(producer->captureGeneration(kDisplayId, &generationA),
                 "failed to capture application A generation")) {
        return false;
    }
    producer->publish(kDisplayId, /*iosurfaceId=*/101, /*width=*/800, /*height=*/600,
                      /*flags=*/0, /*frame=*/7, generationA);

    IosurfaceFrameMetadata metadata = {};
    if (!require(consumer->read(kDisplayId, &metadata), "application A frame was not published") ||
        !require(metadata.iosurfaceId == 101u, "application A IOSurface id mismatch") ||
        !require(metadata.frame == 7u, "application A frame number mismatch")) {
        return false;
    }

    setIosurfaceDisplayExportEnabled(kDisplayId, false);
    producer->clear(kDisplayId);
    if (!require(!consumer->read(kDisplayId, &metadata),
                 "cleared display slot remained readable")) {
        return false;
    }

    setIosurfaceDisplayExportEnabled(kDisplayId, true);
    uint64_t generationB = 0;
    if (!require(producer->captureGeneration(kDisplayId, &generationB),
                 "failed to capture application B generation") ||
        !require(generationA != generationB, "display lifecycle generation did not advance")) {
        return false;
    }

    // A GPU export that began for application A before removal must not
    // repopulate the slot after application B reuses the display id.
    producer->publish(kDisplayId, /*iosurfaceId=*/102, /*width=*/800, /*height=*/600,
                      /*flags=*/0, /*frame=*/8, generationA);
    if (!require(!consumer->read(kDisplayId, &metadata),
                 "late application A frame repopulated the reused slot")) {
        return false;
    }

    producer->publish(kDisplayId, /*iosurfaceId=*/201, /*width=*/1080, /*height=*/1920,
                      /*flags=*/0, /*frame=*/1, generationB);
    if (!require(consumer->read(kDisplayId, &metadata), "application B frame was not published") ||
        !require(metadata.iosurfaceId == 201u, "application B IOSurface id mismatch") ||
        !require(metadata.frame == 1u, "application B frame number mismatch") ||
        !require(metadata.width == 1080u && metadata.height == 1920u,
                 "application B dimensions mismatch")) {
        return false;
    }
    return true;
}

}  // namespace
}  // namespace host
}  // namespace gfxstream

int main() {
    if (!gfxstream::host::runDisplayReuseTest()) {
        return 1;
    }
    std::cout << "gfxstream_iosurface_export_unittests: PASS\n";
    return 0;
}
