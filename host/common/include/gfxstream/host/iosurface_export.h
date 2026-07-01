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

// Cross-process frame control plane shared by the gfxstream producer (running
// inside the qemu process) and the MacMu Metal shell (a separate process).
//
// Protocol v2: the shared-memory object holds a fixed table of per-display
// slots (slot index == Android display id). Each slot is an independent
// odd/even seqlock so any display can publish without touching the others.
// The IOSurface itself carries the shared GPU pixels across the process
// boundary; the slot only publishes per-frame metadata (iosurface id / size /
// frame number). A single inherited Unix datagram socket doorbell wakes the
// consumer for all displays; the doorbell payload is advisory and the consumer
// re-scans the slot table on wake.
//
// The struct layout is duplicated in shell/core/frame_consumer.cpp (MIT side).
// Any layout change must update both copies and bump kFrameShmVersion. See
// docs/FRAME_CHANNEL_V2_CONTROL_PLANE.md.
//
// When the channel cannot be established the producer logs once and drops
// frames (no fallback); the shell and qemu are expected to ship in lockstep.

#ifndef GFXSTREAM_HOST_IOSURFACE_EXPORT_H
#define GFXSTREAM_HOST_IOSURFACE_EXPORT_H

#include <cstdint>
#include <memory>

// NOTE: this header deliberately does NOT include <IOSurface/IOSurface.h>.
// That system header transitively pulls in <sys/param.h>/<arm/param.h>, which
// defines a single-argument `ALIGN(p)` macro that collides with gfxstream's
// own two-argument `ALIGN(x, a)` macro (see gfxstream/Macros.h) and breaks
// translation units that include this header after Macros.h. Callers that need
// the concrete IOSurfaceRef type already include <IOSurface/IOSurface.h>
// themselves; the cross-process helpers below only need an opaque handle.

namespace gfxstream {
namespace host {

// True when MACMU_IOSURFACE_EXPORT / AEMU_IOSURFACE_EXPORT is set to a truthy
// value. Replaces the four previously-duplicated getenv checks.
bool isIosurfaceExportEnabled();

// The fixed slot table size. Android multi-display ids are small integers
// (the ecosystem cap is ~11 displays); display ids >= kFrameSlotCount are
// rejected at the control plane and never reach the frame channel.
inline constexpr uint32_t kFrameSlotCount = 16;
inline constexpr uint32_t kFrameShmVersion = 2;

// Per-display demand from the shell. Export is intentionally opt-in so an
// invisible/closed display does not force a GPU copy and cross-process wakeup
// for every guest frame.
bool isIosurfaceDisplayExportEnabled(uint32_t displayId);
void setIosurfaceDisplayExportEnabled(uint32_t displayId, bool enabled);
void resetIosurfaceDisplayExportSubscriptions();

#ifdef __APPLE__

// Create a BGRA8 IOSurface of the given size with global export enabled.
// Returns a +1 retained IOSurfaceRef as an opaque void* (caller casts back to
// IOSurfaceRef and CFReleases it), or nullptr on failure.
void* createBgra8Iosurface(uint32_t width, uint32_t height);

#endif  // __APPLE__

// Per-display slot flags.
inline constexpr uint32_t kFrameSlotFlagPrimary = 1u << 0;

// In-process snapshot of one display slot's metadata.
struct IosurfaceFrameMetadata {
    uint32_t displayId;
    uint32_t iosurfaceId;
    uint32_t width;
    uint32_t height;
    uint32_t flags;
    uint64_t frame;
    uint64_t timestampNs;
};

// Cross-process frame channel.
//
// Naming: the channel is keyed off the wrapper (shell) pid, which the shell
// already exports to qemu via ANDROID_EMULATOR_WRAPPER_PID. The shared-memory
// object is "macmu.frame.<pid>". The doorbell is a socketpair endpoint
// inherited from the shell and advertised through MACMU_FRAME_DOORBELL_FD.
//
// Lifetime:
//   * the shell creates the shm object and socketpair before launching qemu.
//   * createProducer() is called from gfxstream (inside qemu) on first publish.
//     It opens the existing shm, validates magic/version, and uses the
//     inherited doorbell fd.
//   * If either side cannot be set up, the channel reports !valid() and callers
//     must skip publishing over this channel.
class FrameChannel {
   public:
    FrameChannel();
    ~FrameChannel();

    FrameChannel(const FrameChannel&) = delete;
    FrameChannel& operator=(const FrameChannel&) = delete;

    // Legacy/test consumer helper. The shipping shell uses its MIT-local
    // FrameConsumer so it can pass the socketpair fd through posix_spawn.
    // Returns an invalid-but-non-null channel (valid() == false) on failure.
    static std::unique_ptr<FrameChannel> createConsumer(uint32_t wrapperPid);

    // Build the producer side (gfxstream inside qemu). Looks up the consumer's
    // resources created with the same wrapperPid.
    static std::unique_ptr<FrameChannel> createProducer(uint32_t wrapperPid);

    // Process-wide shared producer, created lazily from
    // ANDROID_EMULATOR_WRAPPER_PID. Multiple sinks (DisplayGl slot 0, DisplayVk
    // slot 0, per-display compose sinks) publish through the same channel.
    // Returns nullptr when the wrapper pid is unset or setup failed.
    static FrameChannel* sharedProducer();

    bool valid() const;

    // Producer: publish the latest frame metadata for |displayId| under that
    // slot's seqlock and ring the consumer's doorbell (best-effort; a dropped
    // notification is harmless because the consumer re-scans on any wake and
    // does a final poll at timeout).
    void publish(uint32_t displayId, uint32_t iosurfaceId, uint32_t width, uint32_t height,
                 uint32_t flags, uint64_t frame);
    // Generation-aware publish for GPU work that can overlap display
    // removal/reuse. A result from an older display generation is discarded.
    void publish(uint32_t displayId, uint32_t iosurfaceId, uint32_t width, uint32_t height,
                 uint32_t flags, uint64_t frame, uint64_t generation);

    // Atomically confirm that export is enabled and capture the current slot
    // generation before starting GPU work.
    bool captureGeneration(uint32_t displayId, uint64_t* outGeneration);
    // clear() advances the slot generation and resets frame metadata so work
    // carrying an older captured generation cannot publish after id reuse.
    void clear(uint32_t displayId);

    // Consumer: read the latest metadata for |displayId| into |out|. Returns
    // false if the channel is invalid, the display id is out of range, or no
    // frame has ever been published for that display.
    bool read(uint32_t displayId, IosurfaceFrameMetadata* out);

    // Consumer: block until display |displayId| has a frame newer than
    // |lastFrame| or |timeoutMs| elapses. Note: the doorbell is shared across
    // displays, so this can wake spuriously for other displays' frames; it
    // re-checks and keeps waiting until the deadline.
    bool waitForFrame(uint32_t displayId, uint64_t lastFrame, uint64_t timeoutMs,
                      IosurfaceFrameMetadata* out);

   private:
    struct Impl;
    std::unique_ptr<Impl> mImpl;
};

}  // namespace host
}  // namespace gfxstream

#endif  // GFXSTREAM_HOST_IOSURFACE_EXPORT_H
