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

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

#include "gfxstream/common/logging.h"
#include "gfxstream/system/System.h"

#ifdef __APPLE__
#include <CoreFoundation/CoreFoundation.h>
#include <IOSurface/IOSurface.h>
#include <poll.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace gfxstream {
namespace host {

namespace {

std::atomic<uint32_t> sIosurfaceDisplaySubscriptions{0};

}  // namespace

bool isIosurfaceExportEnabled() {
    const std::string value = gfxstream::base::getEnvironmentVariable("MACMU_IOSURFACE_EXPORT");
    const std::string enabled =
        value.empty() ? gfxstream::base::getEnvironmentVariable("AEMU_IOSURFACE_EXPORT") : value;
    return enabled == "1" || enabled == "true" || enabled == "TRUE" || enabled == "yes" ||
           enabled == "YES";
}

bool isIosurfaceDisplayExportEnabled(uint32_t displayId) {
    if (!isIosurfaceExportEnabled() || displayId >= kFrameSlotCount) {
        return false;
    }
    const uint32_t mask = uint32_t{1} << displayId;
    return (sIosurfaceDisplaySubscriptions.load(std::memory_order_acquire) & mask) != 0;
}

void setIosurfaceDisplayExportEnabled(uint32_t displayId, bool enabled) {
    if (displayId >= kFrameSlotCount) {
        return;
    }
    const uint32_t mask = uint32_t{1} << displayId;
    if (enabled) {
        sIosurfaceDisplaySubscriptions.fetch_or(mask, std::memory_order_release);
    } else {
        sIosurfaceDisplaySubscriptions.fetch_and(~mask, std::memory_order_release);
    }
}

void resetIosurfaceDisplayExportSubscriptions() {
    sIosurfaceDisplaySubscriptions.store(0, std::memory_order_release);
}

#ifdef __APPLE__

namespace {

void setDictionaryNumber(CFMutableDictionaryRef dictionary, CFStringRef key, int64_t value) {
    CFNumberRef number = CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt64Type, &value);
    if (number) {
        CFDictionarySetValue(dictionary, key, number);
        CFRelease(number);
    }
}

}  // namespace

void* createBgra8Iosurface(uint32_t width, uint32_t height) {
    if (width == 0 || height == 0) {
        return nullptr;
    }
    const int64_t bytesPerElement = 4;
    const int64_t minBytesPerRow = static_cast<int64_t>(width) * bytesPerElement;
    const int64_t bytesPerRow = (minBytesPerRow + 15) & ~int64_t{15};
    const int64_t allocSize = bytesPerRow * static_cast<int64_t>(height);
    constexpr int64_t kPixelFormatBGRA = 0x42475241;  // 'BGRA'

    CFMutableDictionaryRef properties =
        CFDictionaryCreateMutable(kCFAllocatorDefault, 0, &kCFTypeDictionaryKeyCallBacks,
                                  &kCFTypeDictionaryValueCallBacks);
    if (!properties) {
        return nullptr;
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
    return surface;
}

// ---------------------------------------------------------------------------
// FrameChannel implementation (Apple only).
// ---------------------------------------------------------------------------

namespace {

constexpr const char* kShmNamePrefix = "macmu.frame.";
constexpr const char* kFrameDoorbellFdEnv = "MACMU_FRAME_DOORBELL_FD";
constexpr const char* kLegacyFrameDoorbellFdEnv = "AEMU_FRAME_DOORBELL_FD";
constexpr uint64_t kShmMagic = 0x4d41434d5546524dull;  // 'MACMUFRM'
constexpr uint32_t kPixelFormatBgra = 0x42475241;      // 'BGRA'

// On-shm header. The slot table follows at |payloadOffset|.
// DUPLICATED in shell/core/frame_consumer.cpp (MIT side); keep both copies in
// sync and bump kFrameShmVersion on any incompatible change. See
// docs/FRAME_CHANNEL_V2_CONTROL_PLANE.md.
struct ShmHeader {
    uint64_t magic;
    uint32_t version;
    uint32_t payloadOffset;  // byte offset from start of mapping to slot[0]
    uint32_t slotCount;      // kFrameSlotCount
    uint32_t slotStride;     // sizeof(ShmDisplaySlot)
    uint64_t reserved[2];
};

// One per display, exactly one cache line so adjacent slots do not false-share
// between the producer post thread and the consumer.
// DUPLICATED in shell/core/frame_consumer.cpp (MIT side).
struct ShmDisplaySlot {
    uint64_t seq;          // seqlock: odd while the producer is writing
    uint64_t frame;        // monotonic per slot; 0 = never published
    uint32_t iosurfaceId;
    uint32_t width;
    uint32_t height;
    uint32_t dpi;          // 0 = unknown; authoritative dpi comes from the
                           // control plane
    uint32_t pixelFormat;  // fourcc, currently always 'BGRA'
    uint32_t flags;        // kFrameSlotFlag*
    uint64_t timestampNs;  // producer steady clock
    uint8_t pad[16];
};
static_assert(sizeof(ShmDisplaySlot) == 64, "slot must stay one cache line");

// Advisory doorbell payload. The consumer must not depend on the contents:
// datagrams can be coalesced or dropped, so the consumer re-scans the slot
// table on any wake.
struct FrameDoorbellMsg {
    uint32_t displayId;
    uint32_t reserved;
    uint64_t frame;
};

std::string channelName(uint32_t wrapperPid) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%s%u", kShmNamePrefix,
                  static_cast<unsigned>(wrapperPid));
    return std::string(buf);
}

std::string shmPath(const std::string& name) {
    // shm_open names begin with a single slash and contain no further slashes.
    return name.find('/') == std::string::npos ? std::string("/") + name : name;
}

bool inheritedDoorbellFdFromEnv(int* outFd) {
    const char* envName = kFrameDoorbellFdEnv;
    std::string value = gfxstream::base::getEnvironmentVariable(kFrameDoorbellFdEnv);
    if (value.empty()) {
        envName = kLegacyFrameDoorbellFdEnv;
        value = gfxstream::base::getEnvironmentVariable(kLegacyFrameDoorbellFdEnv);
    }
    if (value.empty()) {
        return false;
    }

    char* end = nullptr;
    errno = 0;
    const long fd = std::strtol(value.c_str(), &end, 10);
    if (errno != 0 || end == value.c_str() || *end != '\0' || fd < 0 || fd > INT_MAX) {
        GFXSTREAM_ERROR("FrameChannel producer: invalid %s=%s", envName, value.c_str());
        *outFd = -1;
        return true;
    }
    *outFd = static_cast<int>(fd);
    return true;
}

void setCloseOnExecBestEffort(int fd) {
    const int flags = fcntl(fd, F_GETFD);
    if (flags >= 0) {
        fcntl(fd, F_SETFD, flags | FD_CLOEXEC);
    }
}

uint64_t steadyNowNs() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

constexpr size_t shmTotalSize() {
    return sizeof(ShmHeader) + kFrameSlotCount * sizeof(ShmDisplaySlot);
}

uint32_t wrapperPidFromEnv() {
    const std::string v =
        gfxstream::base::getEnvironmentVariable("ANDROID_EMULATOR_WRAPPER_PID");
    if (v.empty()) return 0;
    return static_cast<uint32_t>(std::strtoul(v.c_str(), nullptr, 10));
}

}  // namespace

struct FrameChannel::Impl {
    bool isConsumer = false;
    bool isValid = false;
    std::string name;
    int shmFd = -1;
    void* mapped = nullptr;  // points to ShmHeader
    size_t mappedSize = 0;
    ShmDisplaySlot* slots = nullptr;
    int doorbellFd = -1;
    std::array<std::mutex, kFrameSlotCount> slotMutexes;
    std::array<std::atomic<uint64_t>, kFrameSlotCount> slotGenerations{};

    ~Impl() {
        if (mapped && mapped != MAP_FAILED) {
            munmap(mapped, mappedSize);
        }
        if (shmFd >= 0) {
            close(shmFd);
        }
        if (doorbellFd >= 0) {
            close(doorbellFd);
        }
        // Consumers unlink the shm object on teardown; producers never do.
        if (isConsumer && !name.empty()) {
            shm_unlink(shmPath(name).c_str());
        }
    }

    ShmDisplaySlot* slotTable() {
        if (!mapped) return nullptr;
        auto* header = static_cast<ShmHeader*>(mapped);
        return reinterpret_cast<ShmDisplaySlot*>(static_cast<char*>(mapped) +
                                                 header->payloadOffset);
    }
};

// static
std::unique_ptr<FrameChannel> FrameChannel::createConsumer(uint32_t wrapperPid) {
    auto channel = std::unique_ptr<FrameChannel>(new FrameChannel());
    channel->mImpl = std::unique_ptr<Impl>(new Impl());
    Impl& impl = *channel->mImpl;
    impl.isConsumer = true;
    impl.name = channelName(wrapperPid);

    const std::string path = shmPath(impl.name);
    impl.shmFd = shm_open(path.c_str(), O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
    if (impl.shmFd < 0) {
        GFXSTREAM_WARNING("FrameChannel consumer: shm_open(%s) failed: %s", path.c_str(),
                          std::strerror(errno));
        return channel;  // valid() == false
    }
    const size_t totalSize = shmTotalSize();
    if (ftruncate(impl.shmFd, static_cast<off_t>(totalSize)) != 0) {
        GFXSTREAM_WARNING("FrameChannel consumer: ftruncate failed: %s", std::strerror(errno));
        return channel;
    }
    void* addr = mmap(nullptr, totalSize, PROT_READ | PROT_WRITE, MAP_SHARED, impl.shmFd, 0);
    if (addr == MAP_FAILED) {
        GFXSTREAM_WARNING("FrameChannel consumer: mmap failed: %s", std::strerror(errno));
        return channel;
    }
    impl.mapped = addr;
    impl.mappedSize = totalSize;
    std::memset(addr, 0, totalSize);
    ShmHeader* header = static_cast<ShmHeader*>(addr);
    header->magic = kShmMagic;
    header->version = kFrameShmVersion;
    header->payloadOffset = sizeof(ShmHeader);
    header->slotCount = kFrameSlotCount;
    header->slotStride = sizeof(ShmDisplaySlot);
    impl.slots = impl.slotTable();

    impl.isValid = true;
    GFXSTREAM_INFO("FrameChannel consumer ready: %s (v%u, %u slots)", impl.name.c_str(),
                   kFrameShmVersion, kFrameSlotCount);
    return channel;
}

// static
std::unique_ptr<FrameChannel> FrameChannel::createProducer(uint32_t wrapperPid) {
    auto channel = std::unique_ptr<FrameChannel>(new FrameChannel());
    channel->mImpl = std::unique_ptr<Impl>(new Impl());
    Impl& impl = *channel->mImpl;
    impl.isConsumer = false;
    impl.name = channelName(wrapperPid);

    const std::string path = shmPath(impl.name);
    impl.shmFd = shm_open(path.c_str(), O_RDWR, 0);
    if (impl.shmFd < 0) {
        // Most likely the shell is older and did not create the channel.
        return channel;  // valid() == false, silent fallback
    }
    const size_t totalSize = shmTotalSize();
    struct stat st;
    if (fstat(impl.shmFd, &st) != 0 || st.st_size < static_cast<off_t>(totalSize)) {
        GFXSTREAM_ERROR(
            "FrameChannel producer: shm %s too small (%lld bytes); shell/qemu version skew?",
            path.c_str(), static_cast<long long>(st.st_size));
        return channel;
    }
    void* addr = mmap(nullptr, totalSize, PROT_READ | PROT_WRITE, MAP_SHARED, impl.shmFd, 0);
    if (addr == MAP_FAILED) {
        return channel;
    }
    impl.mapped = addr;
    impl.mappedSize = totalSize;
    ShmHeader* header = static_cast<ShmHeader*>(addr);
    if (header->magic != kShmMagic || header->version != kFrameShmVersion ||
        header->slotCount != kFrameSlotCount ||
        header->slotStride != sizeof(ShmDisplaySlot)) {
        GFXSTREAM_ERROR(
            "FrameChannel producer: shm version mismatch (got v%u, want v%u); "
            "shell and qemu must ship in lockstep.",
            header->version, kFrameShmVersion);
        munmap(addr, totalSize);
        impl.mapped = nullptr;
        return channel;
    }
    impl.slots = impl.slotTable();

    int inheritedDoorbellFd = -1;
    if (inheritedDoorbellFdFromEnv(&inheritedDoorbellFd)) {
        if (inheritedDoorbellFd < 0) {
            return channel;
        }
        impl.doorbellFd = dup(inheritedDoorbellFd);
        if (impl.doorbellFd < 0) {
            GFXSTREAM_ERROR("FrameChannel producer: dup(%d) failed: %s", inheritedDoorbellFd,
                            std::strerror(errno));
            return channel;
        }
        setCloseOnExecBestEffort(inheritedDoorbellFd);
        setCloseOnExecBestEffort(impl.doorbellFd);
        impl.isValid = true;
        GFXSTREAM_INFO("FrameChannel producer ready: %s (v%u, inherited doorbell fd %d)",
                       impl.name.c_str(), kFrameShmVersion, inheritedDoorbellFd);
        return channel;
    }

    GFXSTREAM_ERROR(
        "FrameChannel producer: missing inherited doorbell fd (%s/%s); guest frames will "
        "not be exported to the shell.",
        kFrameDoorbellFdEnv, kLegacyFrameDoorbellFdEnv);
    return channel;  // valid() == false
}

// static
FrameChannel* FrameChannel::sharedProducer() {
    static std::once_flag once;
    static std::unique_ptr<FrameChannel> channel;
    std::call_once(once, [] {
        const uint32_t pid = wrapperPidFromEnv();
        if (pid == 0) {
            return;
        }
        channel = createProducer(pid);
        if (channel && channel->valid()) {
            GFXSTREAM_INFO("MACMU_IOSURFACE_EXPORT shared FrameChannel producer ready (pid=%u).",
                           pid);
        }
    });
    return channel && channel->valid() ? channel.get() : nullptr;
}

FrameChannel::FrameChannel() = default;
FrameChannel::~FrameChannel() = default;

bool FrameChannel::valid() const {
    return mImpl && mImpl->isValid && mImpl->slots != nullptr;
}

void FrameChannel::publish(uint32_t displayId, uint32_t iosurfaceId, uint32_t width,
                           uint32_t height, uint32_t flags, uint64_t frame) {
    uint64_t generation = 0;
    if (!captureGeneration(displayId, &generation)) {
        return;
    }
    publish(displayId, iosurfaceId, width, height, flags, frame, generation);
}

void FrameChannel::publish(uint32_t displayId, uint32_t iosurfaceId, uint32_t width,
                           uint32_t height, uint32_t flags, uint64_t frame,
                           uint64_t expectedGeneration) {
    if (!valid() || displayId >= kFrameSlotCount) {
        return;
    }
    std::lock_guard<std::mutex> lock(mImpl->slotMutexes[displayId]);
    if (!isIosurfaceDisplayExportEnabled(displayId) ||
        mImpl->slotGenerations[displayId].load(std::memory_order_acquire) !=
            expectedGeneration) {
        return;
    }
    ShmDisplaySlot* slot = &mImpl->slots[displayId];
    // Odd/even seqlock: mark the slot as being written (odd), write the
    // payload including |frame|, then mark it stable (even). The consumer
    // rejects odd or changed sequence numbers, so it can never observe a torn
    // snapshot (e.g. a new surface id paired with the previous frame number).
    // The seq words are shared with the shell process; access them with
    // atomic ops so the compiler cannot elide or reorder the two updates.
    const uint64_t seq0 = __atomic_load_n(&slot->seq, __ATOMIC_RELAXED);
    __atomic_store_n(&slot->seq, seq0 + 1, __ATOMIC_RELAXED);
    std::atomic_thread_fence(std::memory_order_release);
    slot->iosurfaceId = iosurfaceId;
    slot->width = width;
    slot->height = height;
    slot->dpi = 0;
    slot->pixelFormat = kPixelFormatBgra;
    slot->flags = flags | (displayId == 0 ? kFrameSlotFlagPrimary : 0);
    slot->timestampNs = steadyNowNs();
    slot->frame = frame;
    __atomic_store_n(&slot->seq, seq0 + 2, __ATOMIC_RELEASE);

    // Ring the doorbell. If the datagram queue is briefly full, a later frame
    // will wake the consumer; there is intentionally no side-channel fallback.
    if (mImpl->doorbellFd >= 0) {
        FrameDoorbellMsg msg = {displayId, 0, frame};
        send(mImpl->doorbellFd, &msg, sizeof(msg), MSG_DONTWAIT);
    }
}

bool FrameChannel::captureGeneration(uint32_t displayId, uint64_t* outGeneration) {
    if (!valid() || displayId >= kFrameSlotCount || outGeneration == nullptr) {
        return false;
    }
    std::lock_guard<std::mutex> lock(mImpl->slotMutexes[displayId]);
    if (!isIosurfaceDisplayExportEnabled(displayId)) {
        return false;
    }
    *outGeneration =
        mImpl->slotGenerations[displayId].load(std::memory_order_acquire);
    return true;
}

void FrameChannel::clear(uint32_t displayId) {
    if (!valid() || displayId >= kFrameSlotCount) {
        return;
    }
    std::lock_guard<std::mutex> lock(mImpl->slotMutexes[displayId]);
    mImpl->slotGenerations[displayId].fetch_add(1, std::memory_order_acq_rel);

    ShmDisplaySlot* slot = &mImpl->slots[displayId];
    const uint64_t seq0 = __atomic_load_n(&slot->seq, __ATOMIC_RELAXED);
    __atomic_store_n(&slot->seq, seq0 + 1, __ATOMIC_RELAXED);
    std::atomic_thread_fence(std::memory_order_release);
    slot->frame = 0;
    slot->iosurfaceId = 0;
    slot->width = 0;
    slot->height = 0;
    slot->dpi = 0;
    slot->pixelFormat = 0;
    slot->flags = 0;
    slot->timestampNs = 0;
    std::memset(slot->pad, 0, sizeof(slot->pad));
    __atomic_store_n(&slot->seq, seq0 + 2, __ATOMIC_RELEASE);
}

bool FrameChannel::read(uint32_t displayId, IosurfaceFrameMetadata* out) {
    if (!valid() || out == nullptr || displayId >= kFrameSlotCount) return false;
    ShmDisplaySlot* slot = &mImpl->slots[displayId];
    // Seqlock read: sample the sequence, reject odd (writer active), read the
    // payload, re-read the sequence, accept only if unchanged. Bounded retries
    // avoid spinning forever under contention.
    for (int attempt = 0; attempt < 8; ++attempt) {
        const uint64_t s0 = __atomic_load_n(&slot->seq, __ATOMIC_ACQUIRE);
        if (s0 & 1) {
            continue;
        }
        const uint32_t iosurfaceId = slot->iosurfaceId;
        const uint32_t width = slot->width;
        const uint32_t height = slot->height;
        const uint32_t flags = slot->flags;
        const uint64_t frame = slot->frame;
        const uint64_t timestampNs = slot->timestampNs;
        std::atomic_thread_fence(std::memory_order_acquire);
        const uint64_t s1 = __atomic_load_n(&slot->seq, __ATOMIC_RELAXED);
        if (s0 == s1) {
            out->displayId = displayId;
            out->iosurfaceId = iosurfaceId;
            out->width = width;
            out->height = height;
            out->flags = flags;
            out->frame = frame;
            out->timestampNs = timestampNs;
            return frame != 0;
        }
    }
    return false;
}

bool FrameChannel::waitForFrame(uint32_t displayId, uint64_t lastFrame, uint64_t timeoutMs,
                                IosurfaceFrameMetadata* out) {
    if (!valid()) return false;
    // First check without blocking.
    if (read(displayId, out) && out->frame > lastFrame) {
        return true;
    }
    if (mImpl->doorbellFd < 0 || !mImpl->isConsumer) {
        return false;
    }
    // Doorbell fd: block on the Unix datagram socket with a timeout. The
    // doorbell is shared across displays, so a wake may be for another slot;
    // keep re-checking until the deadline.
    const uint64_t deadlineMs = steadyNowNs() / 1000000ull + timeoutMs;
    do {
        pollfd pfd = {};
        pfd.fd = mImpl->doorbellFd;
        pfd.events = POLLIN;
        const int remaining =
            static_cast<int>(std::max<uint64_t>(1ull, deadlineMs - steadyNowNs() / 1000000ull));
        const int pollResult = poll(&pfd, 1, remaining);
        if (pollResult > 0 && (pfd.revents & POLLIN)) {
            // Drain any queued notifications, then check shm.
            while (true) {
                FrameDoorbellMsg msg = {};
                const ssize_t bytes =
                    recv(mImpl->doorbellFd, &msg, sizeof(msg), MSG_DONTWAIT);
                if (bytes > 0) {
                    continue;
                }
                if (bytes == 0) {
                    break;
                }
                if (errno == EINTR) {
                    continue;
                }
                break;
            }
            if (read(displayId, out) && out->frame > lastFrame) {
                return true;
            }
        } else if (pollResult == 0) {
            // Fall through to the final shm check below.
        } else if (pollResult < 0 && errno == EINTR) {
            continue;
        } else {
            // Socket died (consumer teardown) or other error: stop waiting.
            break;
        }
        // Final non-blocking check in case the notification was coalesced.
        if (read(displayId, out) && out->frame > lastFrame) {
            return true;
        }
    } while (steadyNowNs() / 1000000ull < deadlineMs);
    return false;
}

#else  // !__APPLE__

struct FrameChannel::Impl {};

std::unique_ptr<FrameChannel> FrameChannel::createConsumer(uint32_t) {
    return std::unique_ptr<FrameChannel>(new FrameChannel());
}
std::unique_ptr<FrameChannel> FrameChannel::createProducer(uint32_t) {
    return std::unique_ptr<FrameChannel>(new FrameChannel());
}
FrameChannel* FrameChannel::sharedProducer() { return nullptr; }
FrameChannel::FrameChannel() : mImpl(nullptr) {}
FrameChannel::~FrameChannel() = default;
bool FrameChannel::valid() const { return false; }
void FrameChannel::publish(uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint64_t) {}
void FrameChannel::publish(uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint64_t,
                           uint64_t) {}
bool FrameChannel::captureGeneration(uint32_t, uint64_t*) { return false; }
void FrameChannel::clear(uint32_t) {}
bool FrameChannel::read(uint32_t, IosurfaceFrameMetadata*) { return false; }
bool FrameChannel::waitForFrame(uint32_t, uint64_t, uint64_t, IosurfaceFrameMetadata*) {
    return false;
}

#endif  // __APPLE__

}  // namespace host
}  // namespace gfxstream
