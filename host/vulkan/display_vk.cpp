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

#include "display_vk.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <sstream>
#include <string>

#include "borrowed_image_vk.h"
#include "gfxstream/common/logging.h"
#include "gfxstream/host/display_operations.h"
#include "gfxstream/system/System.h"
#include "vulkan/vk_enum_string_helper.h"
#include "vulkan/vk_format_utils.h"

#ifdef __APPLE__
#include <CoreFoundation/CoreFoundation.h>
#include <IOSurface/IOSurface.h>
#endif

namespace gfxstream {
namespace host {
namespace vk {

#define ERR_ONCE(fmt, ...)                           \
    do {                                             \
        static bool displayVkInternalLogged = false; \
        if (!displayVkInternalLogged) {              \
            GFXSTREAM_ERROR(fmt, ##__VA_ARGS__);     \
            displayVkInternalLogged = true;          \
        }                                            \
    } while (0)

static std::shared_future<void> completedFuture() {
    std::shared_future<void> future = std::async(std::launch::deferred, [] {}).share();
    future.wait();
    return future;
}

static bool shouldRecreateSwapchain(VkResult result) {
    switch (result) {
        case VK_SUBOPTIMAL_KHR:
        case VK_ERROR_OUT_OF_DATE_KHR:
        // b/217229121: drivers may return VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT in
        // vkQueuePresentKHR even if VK_EXT_full_screen_exclusive is not enabled.
        case VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT:
            return true;

        default:
            return false;
    }
}

static bool isIosurfaceExportEnabled() {
    const std::string value = gfxstream::base::getEnvironmentVariable("MACMU_IOSURFACE_EXPORT");
    const std::string enabled =
        value.empty() ? gfxstream::base::getEnvironmentVariable("AEMU_IOSURFACE_EXPORT") : value;
    return enabled == "1" || enabled == "true" || enabled == "TRUE" || enabled == "yes" ||
           enabled == "YES";
}

#ifdef __APPLE__

class IosurfaceDisplaySink {
   public:
    IosurfaceDisplaySink(const VulkanDispatch& vk, VkPhysicalDevice physicalDevice, VkDevice device,
                         uint32_t queueFamilyIndex, VkQueue queue,
                         std::shared_ptr<gfxstream::base::Lock> queueLock,
                         VkCommandPool commandPool)
        : m_vk(vk),
          m_physicalDevice(physicalDevice),
          m_device(device),
          m_queueFamilyIndex(queueFamilyIndex),
          m_queue(queue),
          m_queueLock(queueLock),
          m_commandPool(commandPool) {
        const std::string path =
            gfxstream::base::getEnvironmentVariable("MACMU_IOSURFACE_EXPORT_PATH");
        const std::string metadataPath =
            path.empty() ? gfxstream::base::getEnvironmentVariable("AEMU_IOSURFACE_EXPORT_PATH")
                         : path;
        m_metadataPath = metadataPath.empty() ? "/tmp/macmu-iosurface.json" : metadataPath;
    }

    ~IosurfaceDisplaySink() { destroyTarget(); }

    Display::PostResult post(const DisplayVk::Post& postCmd) {
        if (postCmd.layers.empty()) {
            return {.success = true, .postCompletedWaitable = completedFuture()};
        }
        if (postCmd.layers.size() > 1) {
            GFXSTREAM_WARNING(
                "MACMU_IOSURFACE_EXPORT currently exports only the first display layer.");
        }
        if (postCmd.layers[0].rotationDegrees != 0.0f ||
            postCmd.layers[0].colorTransform.has_value() || postCmd.colorTransform.has_value()) {
            GFXSTREAM_WARNING("MACMU_IOSURFACE_EXPORT prototype ignores rotation/color transform.");
        }

        const auto* source = static_cast<const BorrowedImageInfoVk*>(postCmd.layers[0].info);
        if (!source || source->image == VK_NULL_HANDLE) {
            GFXSTREAM_ERROR("MACMU_IOSURFACE_EXPORT source image is missing.");
            return {.success = false, .postCompletedWaitable = completedFuture()};
        }

        const uint32_t width = source->imageCreateInfo.extent.width;
        const uint32_t height = source->imageCreateInfo.extent.height;
        if (width == 0 || height == 0) {
            GFXSTREAM_ERROR("MACMU_IOSURFACE_EXPORT source image has invalid size %ux%u.", width,
                            height);
            return {.success = false, .postCompletedWaitable = completedFuture()};
        }
        if (!ensureTarget(width, height)) {
            return {.success = false, .postCompletedWaitable = completedFuture()};
        }

        if (m_commandBuffer == VK_NULL_HANDLE) {
            VkCommandBufferAllocateInfo commandBufferAi = {
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                .commandPool = m_commandPool,
                .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                .commandBufferCount = 1,
            };
            VkResult result =
                m_vk.vkAllocateCommandBuffers(m_device, &commandBufferAi, &m_commandBuffer);
            if (result != VK_SUCCESS) {
                GFXSTREAM_ERROR("MACMU_IOSURFACE_EXPORT failed to allocate command buffer: %s",
                                string_VkResult(result));
                return {.success = false, .postCompletedWaitable = completedFuture()};
            }
        }
        if (m_fence == VK_NULL_HANDLE) {
            VkFenceCreateInfo fenceCi = {
                .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                .flags = VK_FENCE_CREATE_SIGNALED_BIT,
            };
            VkResult result = m_vk.vkCreateFence(m_device, &fenceCi, nullptr, &m_fence);
            if (result != VK_SUCCESS) {
                GFXSTREAM_ERROR("MACMU_IOSURFACE_EXPORT failed to create fence: %s",
                                string_VkResult(result));
                return {.success = false, .postCompletedWaitable = completedFuture()};
            }
        }

        VkResult waitResult =
            m_vk.vkWaitForFences(m_device, 1, &m_fence, VK_TRUE, kVkWaitForFencesTimeoutNsecs);
        if (waitResult != VK_SUCCESS) {
            GFXSTREAM_ERROR("MACMU_IOSURFACE_EXPORT timed out waiting for previous frame: %s",
                            string_VkResult(waitResult));
            return {.success = false, .postCompletedWaitable = completedFuture()};
        }
        m_vk.vkResetFences(m_device, 1, &m_fence);
        m_vk.vkResetCommandBuffer(m_commandBuffer, 0);

        VkCommandBufferBeginInfo beginInfo = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };
        VkResult beginResult = m_vk.vkBeginCommandBuffer(m_commandBuffer, &beginInfo);
        if (beginResult != VK_SUCCESS) {
            GFXSTREAM_ERROR("MACMU_IOSURFACE_EXPORT failed to begin command buffer: %s",
                            string_VkResult(beginResult));
            return {.success = false, .postCompletedWaitable = completedFuture()};
        }

        std::vector<VkImageMemoryBarrier> acquireQueueTransferBarriers;
        std::vector<VkImageMemoryBarrier> acquireLayoutTransitionBarriers;
        std::vector<VkImageMemoryBarrier> releaseLayoutTransitionBarriers;
        std::vector<VkImageMemoryBarrier> releaseQueueTransferBarriers;
        addNeededBarriersToUseBorrowedImage(
            *source, m_queueFamilyIndex, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_TRANSFER_READ_BIT,
            &acquireQueueTransferBarriers, &acquireLayoutTransitionBarriers,
            &releaseLayoutTransitionBarriers, &releaseQueueTransferBarriers);

        if (!acquireQueueTransferBarriers.empty()) {
            m_vk.vkCmdPipelineBarrier(m_commandBuffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                      VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr,
                                      static_cast<uint32_t>(acquireQueueTransferBarriers.size()),
                                      acquireQueueTransferBarriers.data());
        }
        if (!acquireLayoutTransitionBarriers.empty()) {
            m_vk.vkCmdPipelineBarrier(m_commandBuffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                      VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr,
                                      static_cast<uint32_t>(acquireLayoutTransitionBarriers.size()),
                                      acquireLayoutTransitionBarriers.data());
        }

        const VkImageSubresourceRange colorRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        };
        VkImageMemoryBarrier targetToTransfer = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = m_targetLayout == VK_IMAGE_LAYOUT_UNDEFINED ? 0 : VK_ACCESS_MEMORY_READ_BIT,
            .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .oldLayout = m_targetLayout,
            .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = m_targetImage,
            .subresourceRange = colorRange,
        };
        m_vk.vkCmdPipelineBarrier(m_commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                  VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                                  &targetToTransfer);

        VkImageBlit blitRegion = {
            .srcSubresource =
                {
                    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                    .mipLevel = 0,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                },
            .srcOffsets =
                {
                    {0, 0, 0},
                    {static_cast<int32_t>(width), static_cast<int32_t>(height), 1},
                },
            .dstSubresource =
                {
                    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                    .mipLevel = 0,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                },
            .dstOffsets =
                {
                    {0, 0, 0},
                    {static_cast<int32_t>(m_width), static_cast<int32_t>(m_height), 1},
                },
        };
        m_vk.vkCmdBlitImage(m_commandBuffer, source->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                            m_targetImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blitRegion,
                            VK_FILTER_NEAREST);

        VkImageMemoryBarrier targetToGeneral = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = m_targetImage,
            .subresourceRange = colorRange,
        };
        m_vk.vkCmdPipelineBarrier(m_commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                  VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1,
                                  &targetToGeneral);
        m_targetLayout = VK_IMAGE_LAYOUT_GENERAL;

        if (!releaseLayoutTransitionBarriers.empty()) {
            m_vk.vkCmdPipelineBarrier(m_commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                      VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr,
                                      static_cast<uint32_t>(releaseLayoutTransitionBarriers.size()),
                                      releaseLayoutTransitionBarriers.data());
        }
        if (!releaseQueueTransferBarriers.empty()) {
            m_vk.vkCmdPipelineBarrier(m_commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                      VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr,
                                      static_cast<uint32_t>(releaseQueueTransferBarriers.size()),
                                      releaseQueueTransferBarriers.data());
        }

        VkResult endResult = m_vk.vkEndCommandBuffer(m_commandBuffer);
        if (endResult != VK_SUCCESS) {
            GFXSTREAM_ERROR("MACMU_IOSURFACE_EXPORT failed to end command buffer: %s",
                            string_VkResult(endResult));
            return {.success = false, .postCompletedWaitable = completedFuture()};
        }

        VkSubmitInfo submitInfo = {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1,
            .pCommandBuffers = &m_commandBuffer,
        };
        {
            gfxstream::base::AutoLock lock(*m_queueLock);
            VkResult submitResult = m_vk.vkQueueSubmit(m_queue, 1, &submitInfo, m_fence);
            if (submitResult != VK_SUCCESS) {
                GFXSTREAM_ERROR("MACMU_IOSURFACE_EXPORT failed to submit blit: %s",
                                string_VkResult(submitResult));
                return {.success = false, .postCompletedWaitable = completedFuture()};
            }
        }

        VkResult frameResult =
            m_vk.vkWaitForFences(m_device, 1, &m_fence, VK_TRUE, kVkWaitForFencesTimeoutNsecs);
        if (frameResult != VK_SUCCESS) {
            GFXSTREAM_ERROR("MACMU_IOSURFACE_EXPORT frame wait failed: %s",
                            string_VkResult(frameResult));
            return {.success = false, .postCompletedWaitable = completedFuture()};
        }

        ++m_frameNumber;
        publishMetadata();
        return {.success = true, .postCompletedWaitable = completedFuture()};
    }

   private:
    bool ensureTarget(uint32_t width, uint32_t height) {
        if (m_width == width && m_height == height && m_targetImage != VK_NULL_HANDLE) {
            return true;
        }
        destroyTarget();

        if (!m_vk.vkExportMetalObjectsEXT) {
            GFXSTREAM_ERROR("MACMU_IOSURFACE_EXPORT requires VK_EXT_metal_objects.");
            return false;
        }

        VkFormatProperties formatProperties = {};
        m_vk.vkGetPhysicalDeviceFormatProperties(m_physicalDevice, kTargetFormat,
                                                 &formatProperties);
        if (!(formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT)) {
            GFXSTREAM_ERROR("MACMU_IOSURFACE_EXPORT target format cannot be a blit destination.");
            return false;
        }

        VkExportMetalObjectCreateInfoEXT exportObjectCi = {
            .sType = VK_STRUCTURE_TYPE_EXPORT_METAL_OBJECT_CREATE_INFO_EXT,
            .pNext = nullptr,
            .exportObjectType = VK_EXPORT_METAL_OBJECT_TYPE_METAL_IOSURFACE_BIT_EXT,
        };
        VkImageCreateInfo imageCi = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .pNext = &exportObjectCi,
            .flags = 0,
            .imageType = VK_IMAGE_TYPE_2D,
            .format = kTargetFormat,
            .extent = {.width = width, .height = height, .depth = 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        };
        VkResult imageResult = m_vk.vkCreateImage(m_device, &imageCi, nullptr, &m_targetImage);
        if (imageResult != VK_SUCCESS) {
            GFXSTREAM_ERROR("MACMU_IOSURFACE_EXPORT failed to create target image: %s",
                            string_VkResult(imageResult));
            destroyTarget();
            return false;
        }

        VkMemoryRequirements memoryRequirements = {};
        m_vk.vkGetImageMemoryRequirements(m_device, m_targetImage, &memoryRequirements);
        uint32_t memoryTypeIndex = 0;
        if (!findMemoryType(memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                            &memoryTypeIndex)) {
            GFXSTREAM_ERROR("MACMU_IOSURFACE_EXPORT failed to find target memory type.");
            destroyTarget();
            return false;
        }

        VkMemoryAllocateInfo memoryAi = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = memoryRequirements.size,
            .memoryTypeIndex = memoryTypeIndex,
        };
        VkResult memoryResult = m_vk.vkAllocateMemory(m_device, &memoryAi, nullptr, &m_targetMemory);
        if (memoryResult != VK_SUCCESS) {
            GFXSTREAM_ERROR("MACMU_IOSURFACE_EXPORT failed to allocate target memory: %s",
                            string_VkResult(memoryResult));
            destroyTarget();
            return false;
        }

        VkResult bindResult = m_vk.vkBindImageMemory(m_device, m_targetImage, m_targetMemory, 0);
        if (bindResult != VK_SUCCESS) {
            GFXSTREAM_ERROR("MACMU_IOSURFACE_EXPORT failed to bind target image memory: %s",
                            string_VkResult(bindResult));
            destroyTarget();
            return false;
        }

        VkExportMetalIOSurfaceInfoEXT iosurfaceInfo = {
            .sType = VK_STRUCTURE_TYPE_EXPORT_METAL_IO_SURFACE_INFO_EXT,
            .pNext = nullptr,
            .image = m_targetImage,
            .ioSurface = nullptr,
        };
        VkExportMetalObjectsInfoEXT exportInfo = {
            .sType = VK_STRUCTURE_TYPE_EXPORT_METAL_OBJECTS_INFO_EXT,
            .pNext = &iosurfaceInfo,
        };
        m_vk.vkExportMetalObjectsEXT(m_device, &exportInfo);
        if (!iosurfaceInfo.ioSurface) {
            GFXSTREAM_ERROR("MACMU_IOSURFACE_EXPORT failed to export IOSurface.");
            destroyTarget();
            return false;
        }

        m_surface = iosurfaceInfo.ioSurface;
        CFRetain(m_surface);
        m_surfaceId = IOSurfaceGetID(m_surface);
        m_width = width;
        m_height = height;
        m_targetLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        GFXSTREAM_INFO("MACMU_IOSURFACE_EXPORT publishing IOSurface id=%u size=%ux%u to %s",
                       static_cast<uint32_t>(m_surfaceId), m_width, m_height,
                       m_metadataPath.c_str());
        publishMetadata();
        return true;
    }

    bool findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags properties, uint32_t* outIndex) {
        VkPhysicalDeviceMemoryProperties memoryProperties = {};
        m_vk.vkGetPhysicalDeviceMemoryProperties(m_physicalDevice, &memoryProperties);
        for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
            if ((typeBits & (1u << i)) == 0) {
                continue;
            }
            if ((memoryProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                *outIndex = i;
                return true;
            }
        }
        return false;
    }

    void publishMetadata() {
        if (!m_surface) {
            return;
        }
        const auto now = std::chrono::steady_clock::now().time_since_epoch();
        const auto nowNs = std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
        const std::string tmpPath = m_metadataPath + ".tmp";
        std::ofstream out(tmpPath, std::ios::out | std::ios::trunc);
        if (!out) {
            GFXSTREAM_WARNING("MACMU_IOSURFACE_EXPORT could not write metadata file %s",
                              m_metadataPath.c_str());
            return;
        }
        out << "{\n"
            << "  \"iosurface_id\": " << static_cast<uint32_t>(m_surfaceId) << ",\n"
            << "  \"width\": " << m_width << ",\n"
            << "  \"height\": " << m_height << ",\n"
            << "  \"pixel_format\": \"BGRA8Unorm\",\n"
            << "  \"frame\": " << m_frameNumber << ",\n"
            << "  \"timestamp_ns\": " << nowNs << "\n"
            << "}\n";
        out.close();
        std::rename(tmpPath.c_str(), m_metadataPath.c_str());
    }

    void destroyTarget() {
        if (m_fence != VK_NULL_HANDLE) {
            m_vk.vkWaitForFences(m_device, 1, &m_fence, VK_TRUE, kVkWaitForFencesTimeoutNsecs);
        }
        if (m_surface) {
            CFRelease(m_surface);
            m_surface = nullptr;
        }
        if (m_targetImage != VK_NULL_HANDLE) {
            m_vk.vkDestroyImage(m_device, m_targetImage, nullptr);
            m_targetImage = VK_NULL_HANDLE;
        }
        if (m_targetMemory != VK_NULL_HANDLE) {
            m_vk.vkFreeMemory(m_device, m_targetMemory, nullptr);
            m_targetMemory = VK_NULL_HANDLE;
        }
        if (m_commandBuffer != VK_NULL_HANDLE) {
            m_vk.vkFreeCommandBuffers(m_device, m_commandPool, 1, &m_commandBuffer);
            m_commandBuffer = VK_NULL_HANDLE;
        }
        if (m_fence != VK_NULL_HANDLE) {
            m_vk.vkDestroyFence(m_device, m_fence, nullptr);
            m_fence = VK_NULL_HANDLE;
        }
        m_width = 0;
        m_height = 0;
        m_surfaceId = 0;
        m_targetLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    }

    static constexpr VkFormat kTargetFormat = VK_FORMAT_B8G8R8A8_UNORM;

    const VulkanDispatch& m_vk;
    VkPhysicalDevice m_physicalDevice;
    VkDevice m_device;
    uint32_t m_queueFamilyIndex;
    VkQueue m_queue;
    std::shared_ptr<gfxstream::base::Lock> m_queueLock;
    VkCommandPool m_commandPool;
    std::string m_metadataPath;

    VkCommandBuffer m_commandBuffer = VK_NULL_HANDLE;
    VkFence m_fence = VK_NULL_HANDLE;
    VkImage m_targetImage = VK_NULL_HANDLE;
    VkDeviceMemory m_targetMemory = VK_NULL_HANDLE;
    VkImageLayout m_targetLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    IOSurfaceRef m_surface = nullptr;
    IOSurfaceID m_surfaceId = 0;
    uint32_t m_width = 0;
    uint32_t m_height = 0;
    uint64_t m_frameNumber = 0;
};

#else

class IosurfaceDisplaySink {};

#endif  // __APPLE__

DisplayVk::DisplayVk(const VulkanDispatch& vk, VkPhysicalDevice vkPhysicalDevice, VkDevice vkDevice,
                     CompositorVk* compositorVk, uint32_t compositorQueueFamilyIndex,
                     VkQueue compositorVkQueue,
                     std::shared_ptr<gfxstream::base::Lock> compositorVkQueueLock,
                     uint32_t swapChainQueueFamilyIndex, VkQueue swapChainVkqueue,
                     std::shared_ptr<gfxstream::base::Lock> swapChainVkQueueLock,
                     DebugUtilsHelper debugUtils)
    : m_vk(vk),
      m_vkPhysicalDevice(vkPhysicalDevice),
      m_vkDevice(vkDevice),
      m_debugUtilsHelper(debugUtils),
      m_compositorVk(compositorVk),
      m_compositorQueueFamilyIndex(compositorQueueFamilyIndex),
      m_compositorVkQueue(compositorVkQueue),
      m_compositorVkQueueLock(compositorVkQueueLock),
      m_swapChainQueueFamilyIndex(swapChainQueueFamilyIndex),
      m_swapChainVkQueue(swapChainVkqueue),
      m_swapChainVkQueueLock(swapChainVkQueueLock),
      m_vkCommandPool(VK_NULL_HANDLE),
      m_swapChainStateVk(nullptr) {
    // TODO(kaiyili): validate the capabilites of the passed in Vulkan
    // components.
    VkCommandPoolCreateInfo commandPoolCi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = m_compositorQueueFamilyIndex,
    };
    VK_CHECK(m_vk.vkCreateCommandPool(m_vkDevice, &commandPoolCi, nullptr, &m_vkCommandPool));
    m_debugUtilsHelper.addDebugLabel(m_vkCommandPool, "DisplayVk::commandPool");

    constexpr size_t imageBorrowResourcePoolSize = 10;
    for (size_t i = 0; i < imageBorrowResourcePoolSize; i++) {
        m_imageBorrowResources.emplace_back(
            ImageBorrowResource::create(m_vk, m_vkDevice, m_vkCommandPool));

        auto& res = m_imageBorrowResources.back();
        m_debugUtilsHelper.addDebugLabel(res->m_vkCommandBuffer,
                                         "DisplayVk::imageBorrowResources:CB%d", i);
        m_debugUtilsHelper.addDebugLabel(res->m_completeFence,
                                         "DisplayVk:imageBorrowResources:Fence%d", i);
    }
}

DisplayVk::~DisplayVk() {
    m_iosurfaceDisplaySink.reset();
    destroySwapchain();
    m_imageBorrowResources.clear();
    m_vk.vkDestroyCommandPool(m_vkDevice, m_vkCommandPool, nullptr);
}

void DisplayVk::drainQueues() {
    {
        gfxstream::base::AutoLock lock(*m_swapChainVkQueueLock);
        VK_CHECK(m_vk.vkQueueWaitIdle(m_swapChainVkQueue));
    }
    // We don't assume all VkCommandBuffer submitted to m_compositorVkQueueLock is always followed
    // by another operation on the m_swapChainVkQueue. Therefore, only waiting for the
    // m_swapChainVkQueue is not enough to guarantee all resources used are free to be destroyed.
    if (m_swapChainVkQueue != m_compositorVkQueue) {
        gfxstream::base::AutoLock lock(*m_compositorVkQueueLock);
        VK_CHECK(m_vk.vkQueueWaitIdle(m_compositorVkQueue));
    }
}

void DisplayVk::clear() {
    // Report once, as it's generally not a fatal issue but may cause graphical issues.
    ERR_ONCE("DisplayVk::%s: Unimplemented", __func__);
}

void DisplayVk::bindToSurfaceImpl(DisplaySurface* surface) { m_needToRecreateSwapChain = true; }

void DisplayVk::surfaceUpdated(DisplaySurface* surface) { m_needToRecreateSwapChain = true; }

void DisplayVk::unbindFromSurfaceImpl() { destroySwapchain(); }

void DisplayVk::destroySwapchain() {
    drainQueues();
    m_freePostResources.clear();
    m_postResourceFutures.clear();
    m_swapChainStateVk.reset();
    m_needToRecreateSwapChain = true;
}

bool DisplayVk::recreateSwapchain() {
    destroySwapchain();

    const auto* surface = getBoundSurface();
    if (!surface) {
        GFXSTREAM_FATAL("DisplayVk can't create VkSwapchainKHR without a VkSurfaceKHR");
    }
    const auto* surfaceVk = static_cast<const DisplaySurfaceVk*>(surface->getImpl());

    if (!SwapChainStateVk::validateQueueFamilyProperties(
            m_vk, m_vkPhysicalDevice, surfaceVk->getSurface(), m_swapChainQueueFamilyIndex)) {
        GFXSTREAM_FATAL(
            "DisplayVk can't create VkSwapchainKHR with given VkDevice and VkSurfaceKHR.");
    }
    GFXSTREAM_INFO("Creating swapchain with size %" PRIu32 "x%" PRIu32 ".", surface->getWidth(),
                   surface->getHeight());
    auto swapChainCi = SwapChainStateVk::createSwapChainCi(
        m_vk, surfaceVk->getSurface(), m_vkPhysicalDevice, surface->getWidth(),
        surface->getHeight(), {m_swapChainQueueFamilyIndex, m_compositorQueueFamilyIndex});
    if (!swapChainCi) {
        return false;
    }
    VkFormatProperties formatProps;
    m_vk.vkGetPhysicalDeviceFormatProperties(m_vkPhysicalDevice,
                                             swapChainCi->mCreateInfo.imageFormat, &formatProps);
    if (!(formatProps.optimalTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT)) {
        GFXSTREAM_FATAL(
            "DisplayVk: The image format chosen for present VkImage can't be used as the color "
            "attachment, and therefore can't be used as the render target of CompositorVk.");
    }
    m_swapChainStateVk = SwapChainStateVk::createSwapChainVk(
        m_vk, m_vkDevice, swapChainCi->mCreateInfo, m_debugUtilsHelper);
    if (m_swapChainStateVk == nullptr) return false;
    int numSwapChainImages = m_swapChainStateVk->getVkImages().size();

    m_postResourceFutures.resize(numSwapChainImages, std::nullopt);
    for (int i = 0; i < numSwapChainImages + 1; ++i) {
        m_freePostResources.emplace_back(PostResource::create(m_vk, m_vkDevice, m_vkCommandPool));

        auto& res = m_freePostResources.back();
        m_debugUtilsHelper.addDebugLabel(res->m_swapchainImageReleaseFence,
                                         "DisplayVk::postResources:Fence%d", i);
        m_debugUtilsHelper.addDebugLabel(res->m_swapchainImageAcquireSemaphore,
                                         "DisplayVk::postResources:AcquireSemaphore%d", i);
        m_debugUtilsHelper.addDebugLabel(res->m_swapchainImageReleaseFence,
                                         "DisplayVk::postResources:ReleaseSemaphore%d", i);
        m_debugUtilsHelper.addDebugLabel(res->m_vkCommandBuffer, "DisplayVk::postResources:CB%d",
                                         i);
    }

    m_needToRecreateSwapChain = false;
    return true;
}

DisplayVk::PostResult DisplayVk::post(const Post& postCmd) {
    if (isIosurfaceExportEnabled()) {
        return postToIosurface(postCmd);
    }

    const auto* surface = getBoundSurface();
    if (!surface) {
        GFXSTREAM_ERROR("Trying to present to non-existing surface!");
        return PostResult{
            .success = true,
            .postCompletedWaitable = completedFuture(),
        };
    }

    if (m_needToRecreateSwapChain) {
        GFXSTREAM_INFO("Recreating swapchain...");

        constexpr const int kMaxRecreateSwapchainRetries = 8;
        int retriesRemaining = kMaxRecreateSwapchainRetries;
        while (retriesRemaining >= 0 && !recreateSwapchain()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            --retriesRemaining;
            GFXSTREAM_INFO("Swapchain recreation failed, retrying...");
        }

        if (retriesRemaining < 0) {
            GFXSTREAM_FATAL("Failed to create Swapchain. w:%d h:%d", surface->getWidth(),
                            surface->getHeight());
        }

        GFXSTREAM_INFO("Recreating swapchain completed.");
    }

    auto result = postImpl(postCmd);
    if (!result.success) {
        m_needToRecreateSwapChain = true;
    }
    return result;
}

DisplayVk::PostResult DisplayVk::postToIosurface(const Post& postCmd) {
#ifdef __APPLE__
    if (!m_iosurfaceDisplaySink) {
        m_iosurfaceDisplaySink = std::make_unique<IosurfaceDisplaySink>(
            m_vk, m_vkPhysicalDevice, m_vkDevice, m_compositorQueueFamilyIndex, m_compositorVkQueue,
            m_compositorVkQueueLock, m_vkCommandPool);
    }
    return m_iosurfaceDisplaySink->post(postCmd);
#else
    GFXSTREAM_ERROR("MACMU_IOSURFACE_EXPORT is only supported on macOS.");
    return PostResult{.success = false, .postCompletedWaitable = completedFuture()};
#endif
}

DisplayVk::PostResult DisplayVk::postImpl(const Post& postCmd) {
    auto completedFuture = std::async(std::launch::deferred, [] {}).share();
    completedFuture.wait();

    // One for acquire, one for release.
    size_t requiredResources = postCmd.layers.size() * 2;
    std::vector<const ImageBorrowResource*> imageBorrowResources;
    imageBorrowResources.reserve(requiredResources);

    for (size_t i = 0; i < requiredResources; i++) {
        auto freeImageBorrowResource =
            std::find_if(m_imageBorrowResources.begin(), m_imageBorrowResources.end(),
                         [this](const std::unique_ptr<ImageBorrowResource>& imageBorrowResource) {
                             VkResult fenceStatus = m_vk.vkGetFenceStatus(
                                 m_vkDevice, imageBorrowResource->m_completeFence);
                             if (fenceStatus == VK_SUCCESS) {
                                 return true;
                             }
                             if (fenceStatus == VK_NOT_READY) {
                                 return false;
                             }
                             VK_CHECK(fenceStatus);
                             return false;
                         });
        if (freeImageBorrowResource == m_imageBorrowResources.end()) {
            freeImageBorrowResource = m_imageBorrowResources.begin();
            VK_CHECK(m_vk.vkWaitForFences(
                m_vkDevice, 1, &(*freeImageBorrowResource)->m_completeFence, VK_TRUE, UINT64_MAX));
        }
        VK_CHECK(m_vk.vkResetFences(m_vkDevice, 1, &(*freeImageBorrowResource)->m_completeFence));
        imageBorrowResources.push_back(freeImageBorrowResource->get());
    }
    // We need to unconditionally acquire and release the image to satisfy the requiremment for the
    // borrowed image.
    struct ImageBorrower {
        ImageBorrower(const VulkanDispatch& vk, VkQueue queue,
                      std::shared_ptr<gfxstream::base::Lock> queueLock,
                      uint32_t usedQueueFamilyIndex, const BorrowedImageInfoVk& image,
                      const ImageBorrowResource& acquireResource,
                      const ImageBorrowResource& releaseResource, VkImageLayout layout)
            : m_vk(vk),
              m_vkQueue(queue),
              m_queueLock(queueLock),
              m_releaseResource(releaseResource) {
            std::vector<VkImageMemoryBarrier> acquireQueueTransferBarriers;
            std::vector<VkImageMemoryBarrier> acquireLayoutTransitionBarriers;
            std::vector<VkImageMemoryBarrier> releaseLayoutTransitionBarriers;
            std::vector<VkImageMemoryBarrier> releaseQueueTransferBarriers;
            VkAccessFlags accessMask = VK_ACCESS_TRANSFER_READ_BIT;
            if (layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
                accessMask = VK_ACCESS_SHADER_READ_BIT;
            }

            addNeededBarriersToUseBorrowedImage(
                image, usedQueueFamilyIndex,
                /*usedInitialImageLayout=*/layout,
                /*usedFinalImageLayout=*/layout, accessMask, &acquireQueueTransferBarriers,
                &acquireLayoutTransitionBarriers, &releaseLayoutTransitionBarriers,
                &releaseQueueTransferBarriers);

            // Record the acquire commands.
            const VkCommandBufferBeginInfo acquireBeginInfo = {
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            };
            VK_CHECK(
                m_vk.vkBeginCommandBuffer(acquireResource.m_vkCommandBuffer, &acquireBeginInfo));
            if (!acquireQueueTransferBarriers.empty()) {
                m_vk.vkCmdPipelineBarrier(
                    acquireResource.m_vkCommandBuffer,
                    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                    0, 0, nullptr, 0, nullptr,
                    static_cast<uint32_t>(acquireQueueTransferBarriers.size()),
                    acquireQueueTransferBarriers.data());
            }
            if (!acquireLayoutTransitionBarriers.empty()) {
                VkPipelineStageFlags dstStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
                if (layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
                    dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
                }
                m_vk.vkCmdPipelineBarrier(
                    acquireResource.m_vkCommandBuffer,
                    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                    dstStageMask, 0, 0, nullptr, 0, nullptr,
                    static_cast<uint32_t>(acquireLayoutTransitionBarriers.size()),
                    acquireLayoutTransitionBarriers.data());
            }
            VK_CHECK(m_vk.vkEndCommandBuffer(acquireResource.m_vkCommandBuffer));

            // Record the release commands.
            const VkCommandBufferBeginInfo releaseBeginInfo = {
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            };
            VK_CHECK(
                m_vk.vkBeginCommandBuffer(releaseResource.m_vkCommandBuffer, &releaseBeginInfo));
            if (!releaseLayoutTransitionBarriers.empty()) {
                VkPipelineStageFlags srcStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
                if (layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
                    srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
                }
                m_vk.vkCmdPipelineBarrier(
                    releaseResource.m_vkCommandBuffer, srcStageMask,
                    VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr,
                    static_cast<uint32_t>(releaseLayoutTransitionBarriers.size()),
                    releaseLayoutTransitionBarriers.data());
            }
            if (!releaseQueueTransferBarriers.empty()) {
                VkPipelineStageFlags srcStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
                if (layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
                    srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
                }
                m_vk.vkCmdPipelineBarrier(
                    releaseResource.m_vkCommandBuffer, srcStageMask,
                    VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr,
                    static_cast<uint32_t>(releaseQueueTransferBarriers.size()),
                    releaseQueueTransferBarriers.data());
            }
            VK_CHECK(m_vk.vkEndCommandBuffer(releaseResource.m_vkCommandBuffer));

            VkSubmitInfo submitInfo = {
                .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                .waitSemaphoreCount = 0,
                .pWaitSemaphores = nullptr,
                .pWaitDstStageMask = nullptr,
                .commandBufferCount = 1,
                .pCommandBuffers = &acquireResource.m_vkCommandBuffer,
                .signalSemaphoreCount = 0,
                .pSignalSemaphores = nullptr,
            };
            // Submit the acquire commands.
            {
                gfxstream::base::AutoLock lock(*m_queueLock);
                VK_CHECK(
                    m_vk.vkQueueSubmit(m_vkQueue, 1, &submitInfo, acquireResource.m_completeFence));
            }
        }

        const VulkanDispatch& m_vk;
        const VkQueue m_vkQueue;
        std::shared_ptr<gfxstream::base::Lock> m_queueLock;
        const ImageBorrowResource& m_releaseResource;
        ~ImageBorrower() {
            VkSubmitInfo submitInfo = {
                .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                .waitSemaphoreCount = 0,
                .pWaitSemaphores = nullptr,
                .pWaitDstStageMask = nullptr,
                .commandBufferCount = 1,
                .pCommandBuffers = &m_releaseResource.m_vkCommandBuffer,
                .signalSemaphoreCount = 0,
                .pSignalSemaphores = nullptr,
            };
            // Submit the release commands.
            {
                gfxstream::base::AutoLock lock(*m_queueLock);
                VK_CHECK(m_vk.vkQueueSubmit(m_vkQueue, 1, &submitInfo,
                                            m_releaseResource.m_completeFence));
            }
        }
    };

    const auto* surface = getBoundSurface();
    if (!m_swapChainStateVk || !surface) {
        GFXSTREAM_ERROR("Cannot post ColorBuffer: No surface bound.");
        return PostResult{true, std::move(completedFuture)};
    }

    bool useBlit = false;
    // We can only use blit if there is exactly one image, no rotation/color transform,
    // and no custom display frame (full screen).
    if (postCmd.layers.size() == 1) {
        const auto& layer = postCmd.layers[0];
        if (layer.rotationDegrees == 0 && !layer.colorTransform.has_value() &&
            hwc_rect_get_width(&layer.displayFrame) == 0 && !postCmd.colorTransform.has_value()) {
            const auto* sourceImageInfoVk = static_cast<const BorrowedImageInfoVk*>(layer.info);
            if (canPost(sourceImageInfoVk->imageCreateInfo)) {
                useBlit = true;
            }
        }
    }

    std::vector<std::unique_ptr<ImageBorrower>> borrowers;
    borrowers.reserve(postCmd.layers.size());

    for (size_t i = 0; i < postCmd.layers.size(); ++i) {
        const auto& layer = postCmd.layers[i];
        const auto* sourceImageInfoVk = static_cast<const BorrowedImageInfoVk*>(layer.info);
        VkImageLayout layout = useBlit ? VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
                                       : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        borrowers.emplace_back(std::make_unique<ImageBorrower>(
            m_vk, m_compositorVkQueue, m_compositorVkQueueLock, m_compositorQueueFamilyIndex,
            *sourceImageInfoVk, *imageBorrowResources[2 * i], *imageBorrowResources[2 * i + 1],
            layout));
    }

    for (auto& postResourceFutureOpt : m_postResourceFutures) {
        if (!postResourceFutureOpt.has_value()) {
            continue;
        }
        auto postResourceFuture = postResourceFutureOpt.value();
        if (!postResourceFuture.valid()) {
            GFXSTREAM_FATAL("Invalid postResourceFuture in m_postResourceFutures.");
        }
        std::future_status status = postResourceFuture.wait_for(std::chrono::seconds(0));
        if (status == std::future_status::ready) {
            m_freePostResources.emplace_back(postResourceFuture.get());
            postResourceFutureOpt = std::nullopt;
        }
    }
    if (m_freePostResources.empty()) {
        for (auto& postResourceFutureOpt : m_postResourceFutures) {
            if (!postResourceFutureOpt.has_value()) {
                continue;
            }
            m_freePostResources.emplace_back(postResourceFutureOpt.value().get());
            postResourceFutureOpt = std::nullopt;
            break;
        }
    }
    std::shared_ptr<PostResource> postResource = m_freePostResources.front();
    m_freePostResources.pop_front();

    VkSemaphore imageReadySem = postResource->m_swapchainImageAcquireSemaphore;

    uint32_t imageIndex;
    VkResult acquireRes =
        m_vk.vkAcquireNextImageKHR(m_vkDevice, m_swapChainStateVk->getSwapChain(), UINT64_MAX,
                                   imageReadySem, VK_NULL_HANDLE, &imageIndex);
    if (shouldRecreateSwapchain(acquireRes)) {
        return PostResult{false, std::shared_future<void>()};
    }
    if (acquireRes == VK_ERROR_SURFACE_LOST_KHR) {
        GFXSTREAM_ERROR("Cannot post ColorBuffer: Swapchain surface is lost.");
        return PostResult{false, std::move(completedFuture)};
    }
    VK_CHECK(acquireRes);

    if (m_postResourceFutures[imageIndex].has_value()) {
        m_freePostResources.emplace_back(m_postResourceFutures[imageIndex].value().get());
        m_postResourceFutures[imageIndex] = std::nullopt;
    }

    VkCommandBuffer cmdBuff = postResource->m_vkCommandBuffer;
    VK_CHECK(m_vk.vkResetCommandBuffer(cmdBuff, 0));

    const VkCommandBufferBeginInfo beginInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    VK_CHECK(m_vk.vkBeginCommandBuffer(cmdBuff, &beginInfo));

    VkImageLayout currentSwapchainLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkAccessFlags curSrcAccessMask = VK_ACCESS_NONE;
    VkImage currentSwapchainImage = m_swapChainStateVk->getVkImages()[imageIndex];
    VkRenderPass currentSwapchainRenderpass = m_swapChainStateVk->getVkRenderPasses()[imageIndex];
    VkFramebuffer currentSwapchainFramebuffer = m_swapChainStateVk->getVkFramebuffers()[imageIndex];
    const VkImageSubresourceRange subresourceRange = {
        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        .baseMipLevel = 0,
        .levelCount = 1,
        .baseArrayLayer = 0,
        .layerCount = 1,
    };

    // Note: The extent used during swapchain creation must be used here and not the
    // current surface's extent as the swapchain may not have been updated after the
    // surface resized. The blit must not try to write outside of the extent of the
    // existing swapchain images.
    const VkExtent2D swapchainImageExtent = m_swapChainStateVk->getImageExtent();

    CompositorVkBase::ImmediateModeResources* imResources =
        m_compositorVk ? m_compositorVk->acquireImmediateModeResources() : nullptr;

    CompositorVk::ImageDrawParams drawParams = {
        .commandBuffer = cmdBuff,
        .targetFormat = m_swapChainStateVk->getFormat(),
        .targetWidth = swapchainImageExtent.width,
        .targetHeight = swapchainImageExtent.height,
        .targetRenderPass = currentSwapchainRenderpass,
        .targetFramebuffer = currentSwapchainFramebuffer,
        .frameResources = imResources,
        .rotationDegrees = 0.0f,
        .useScreenBlend = false,
        .colorTransform = std::nullopt,
    };

    // For multi-display, we process background/mask per-image in the loop below.
    bool renderBackground = imResources && m_compositorVk->hasScreenBackground();
    // Disable blit if:
    // 1. We have multiple images (composition needed)
    // 2. We have a background to render (blit overwrites it)
    if (postCmd.layers.size() > 1 || renderBackground) {
        useBlit = false;
    }
    // Explicitly clear swapchain if not using blit (ensures clean canvas for composition)
    // Only clear if we are NOT rendering the background skin (e.g. multi-display mode or no skin)
    // If we are rendering the skin, we rely on it to cover the frame (and avoid black corners in
    // gaps).
    if (!useBlit && (!renderBackground || (postCmd.layers.size() > 1))) {
        VkImageMemoryBarrier barrier = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = curSrcAccessMask,
            .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .oldLayout = currentSwapchainLayout,
            .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = currentSwapchainImage,
            .subresourceRange = subresourceRange,
        };
        m_vk.vkCmdPipelineBarrier(cmdBuff, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                  VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                                  &barrier);

        const VkClearColorValue clearColor = {{0.0f, 0.0f, 0.0f, 1.0f}};
        m_vk.vkCmdClearColorImage(cmdBuff, currentSwapchainImage,
                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clearColor, 1,
                                  &subresourceRange);

        currentSwapchainLayout = barrier.newLayout;
        curSrcAccessMask = barrier.dstAccessMask;
    }

    if (useBlit) {
        // Use vkCmdBlitImage to post the image (single image optimized path)
        const auto& layer = postCmd.layers[0];
        const auto* sourceImageInfoVk = static_cast<const BorrowedImageInfoVk*>(layer.info);
        VkImageMemoryBarrier acquireSwapchainImageBarrier = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = curSrcAccessMask,
            .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .oldLayout = currentSwapchainLayout,
            .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = currentSwapchainImage,
            .subresourceRange = subresourceRange,
        };
        m_vk.vkCmdPipelineBarrier(cmdBuff, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                  VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                                  &acquireSwapchainImageBarrier);
        currentSwapchainLayout = acquireSwapchainImageBarrier.newLayout;
        curSrcAccessMask = acquireSwapchainImageBarrier.dstAccessMask;

        const VkImageBlit region = {
            .srcSubresource = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                               .mipLevel = 0,
                               .baseArrayLayer = 0,
                               .layerCount = 1},
            .srcOffsets = {{0, 0, 0},
                           {static_cast<int32_t>(sourceImageInfoVk->imageCreateInfo.extent.width),
                            static_cast<int32_t>(sourceImageInfoVk->imageCreateInfo.extent.height),
                            1}},
            .dstSubresource = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                               .mipLevel = 0,
                               .baseArrayLayer = 0,
                               .layerCount = 1},
            .dstOffsets = {{0, 0, 0},
                           {static_cast<int32_t>(swapchainImageExtent.width),
                            static_cast<int32_t>(swapchainImageExtent.height), 1}},
        };
        VkFormat displayBufferFormat = sourceImageInfoVk->imageCreateInfo.format;
        VkImageTiling displayBufferTiling = sourceImageInfoVk->imageCreateInfo.tiling;

        VkFilter filter = VK_FILTER_NEAREST;
        VkFormatFeatureFlags displayBufferFormatFeatures =
            getFormatFeatures(displayBufferFormat, displayBufferTiling);
        if (formatIsDepthOrStencil(displayBufferFormat)) {
            ERR_ONCE(
                "The format of the display buffer, %s, is a depth/stencil format, we can only use "
                "the VK_FILTER_NEAREST filter according to VUID-vkCmdBlitImage-srcImage-00232.",
                string_VkFormat(displayBufferFormat));
            filter = VK_FILTER_NEAREST;
        } else if (!(displayBufferFormatFeatures &
                     VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)) {
            ERR_ONCE(
                "The format of the display buffer, %s, with the tiling, %s, doesn't support "
                "VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT, so we can only use the "
                "VK_FILTER_NEAREST filter according VUID-vkCmdBlitImage-filter-02001. The "
                "supported features are %s.",
                string_VkFormat(displayBufferFormat), string_VkImageTiling(displayBufferTiling),
                string_VkFormatFeatureFlags(displayBufferFormatFeatures).c_str());
            filter = VK_FILTER_NEAREST;
        } else {
            filter = VK_FILTER_LINEAR;
        }
        m_vk.vkCmdBlitImage(cmdBuff, sourceImageInfoVk->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                            currentSwapchainImage, currentSwapchainLayout, 1, &region, filter);
    } else {
        // Use immediate drawImage call to render the images
        if (currentSwapchainLayout != VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) {
            VkImageMemoryBarrier transitionSwapchainToAttachmentBarrier = {
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .srcAccessMask = curSrcAccessMask,
                .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                .oldLayout = currentSwapchainLayout,
                .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = currentSwapchainImage,
                .subresourceRange = subresourceRange,
            };
            m_vk.vkCmdPipelineBarrier(cmdBuff, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                                      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, nullptr,
                                      0, nullptr, 1, &transitionSwapchainToAttachmentBarrier);
            currentSwapchainLayout = transitionSwapchainToAttachmentBarrier.newLayout;
            curSrcAccessMask = transitionSwapchainToAttachmentBarrier.dstAccessMask;
        }

        // Compute logical width/height from the union of all image display frames
        int32_t logicalWidth = postCmd.frameWidth;
        int32_t logicalHeight = postCmd.frameHeight;
        if (logicalWidth == 0 || logicalHeight == 0) {
            for (const auto& layer : postCmd.layers) {
                logicalWidth = std::max(logicalWidth, layer.displayFrame.right);
                logicalHeight = std::max(logicalHeight, layer.displayFrame.bottom);
            }
        }

        // If logical size is 0 (e.g. single image with 0-init frame), assume it matches swapchain
        if (logicalWidth == 0) logicalWidth = swapchainImageExtent.width;
        if (logicalHeight == 0) logicalHeight = swapchainImageExtent.height;

        // Revert to Stretch scaling to match Input Mapper expectations (Fit Window)
        float scaleX = static_cast<float>(swapchainImageExtent.width) / logicalWidth;
        float scaleY = static_cast<float>(swapchainImageExtent.height) / logicalHeight;

        for (size_t i = 0; i < postCmd.layers.size(); ++i) {
            const auto& layer = postCmd.layers[i];
            const auto* sourceImageInfoVk = static_cast<const BorrowedImageInfoVk*>(layer.info);
            // Strictly disable skin/mask if multi-display mode is active, regardless of image count
            bool isMultiDisplay = postCmd.layers.size() > 1;
            bool disableMask = isMultiDisplay;

            // Prefer per-layer color transform, then global, then none
            if (layer.colorTransform.has_value()) {
                drawParams.colorTransform = layer.colorTransform;
            } else {
                drawParams.colorTransform = postCmd.colorTransform;
            }
            drawParams.rotationDegrees = layer.rotationDegrees;

            if (hwc_rect_get_width(&layer.displayFrame) == 0 ||
                hwc_rect_get_height(&layer.displayFrame) == 0) {
                // Fallback for 0-sized frames
                drawParams.displayFrame.left = 0;
                drawParams.displayFrame.top = 0;
                drawParams.displayFrame.right = swapchainImageExtent.width;
                drawParams.displayFrame.bottom = swapchainImageExtent.height;
            } else {
                // Apply Y-Flip layout to match GL/Input expectations (Bottom-Left origin logic)
                int32_t flippedTop = logicalHeight - layer.displayFrame.bottom;
                drawParams.displayFrame.left =
                    static_cast<int32_t>(layer.displayFrame.left * scaleX);
                drawParams.displayFrame.top = static_cast<int32_t>(flippedTop * scaleY);
                drawParams.displayFrame.right =
                    static_cast<int32_t>(layer.displayFrame.right * scaleX);
                // bottom is determined by adding height (scaled) to top
                drawParams.displayFrame.bottom = static_cast<int32_t>(
                    (flippedTop + hwc_rect_get_height(&layer.displayFrame)) * scaleY);
            }

            // Ensure transition to COLOR_ATTACHMENT_OPTIMAL if we're about to draw (after clear)
            if (currentSwapchainLayout != VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) {
                VkImageMemoryBarrier transitionToAttachment = {
                    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    .srcAccessMask = curSrcAccessMask,
                    .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                    .oldLayout = currentSwapchainLayout,
                    .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .image = currentSwapchainImage,
                    .subresourceRange = subresourceRange,
                };
                m_vk.vkCmdPipelineBarrier(cmdBuff, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                                          VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0,
                                          nullptr, 0, nullptr, 1, &transitionToAttachment);
                currentSwapchainLayout = transitionToAttachment.newLayout;
                curSrcAccessMask = transitionToAttachment.dstAccessMask;
            }

            // Draw Background only if mask is allowed (single display mode)
            if (i == 0 && renderBackground && !disableMask) {
                m_compositorVk->drawScreenBackground(drawParams);

                // Draw subsequent passes in screen blend mode
                drawParams.useScreenBlend = true;

                // Adjust display rendering if a layout has been given
                const int targetWidth = drawParams.targetWidth;
                const int targetHeight = drawParams.targetHeight;
                Rect scaledDisplayRect = {};
                if (m_compositorVk->getScaledDisplayRect(scaledDisplayRect, targetWidth,
                                                         targetHeight)) {
                    drawParams.displayFrame.left = scaledDisplayRect.pos.x;
                    drawParams.displayFrame.top = scaledDisplayRect.pos.y;
                    drawParams.displayFrame.right =
                        drawParams.displayFrame.left + scaledDisplayRect.size.w;
                    drawParams.displayFrame.bottom =
                        drawParams.displayFrame.top + scaledDisplayRect.size.h;
                }
            }

            m_compositorVk->drawImage(drawParams, sourceImageInfoVk->imageView);

            // Draw Mask only if mask is allowed (single display mode)
            if (i == 0 && m_compositorVk->hasScreenMask() && !disableMask) {
                m_compositorVk->drawScreenMask(drawParams);
            }
        }
    }

    VkImageMemoryBarrier releaseSwapchainImageBarrier = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = curSrcAccessMask,
        .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT,
        .oldLayout = currentSwapchainLayout,
        .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = currentSwapchainImage,
        .subresourceRange = subresourceRange,
    };

    VkPipelineStageFlags srcStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkPipelineStageFlags dstStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
    if (currentSwapchainLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) {
        srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
    }

    m_vk.vkCmdPipelineBarrier(cmdBuff, srcStageMask, dstStageMask, 0, 0, nullptr, 0, nullptr, 1,
                              &releaseSwapchainImageBarrier);
    currentSwapchainLayout = releaseSwapchainImageBarrier.newLayout;
    curSrcAccessMask = releaseSwapchainImageBarrier.dstAccessMask;

    VK_CHECK(m_vk.vkEndCommandBuffer(cmdBuff));

    VkFence postCompleteFence = postResource->m_swapchainImageReleaseFence;
    VK_CHECK(m_vk.vkResetFences(m_vkDevice, 1, &postCompleteFence));
    VkSemaphore postCompleteSemaphore = postResource->m_swapchainImageReleaseSemaphore;
    VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_TRANSFER_BIT};
    VkSubmitInfo submitInfo = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                               .waitSemaphoreCount = 1,
                               .pWaitSemaphores = &imageReadySem,
                               .pWaitDstStageMask = waitStages,
                               .commandBufferCount = 1,
                               .pCommandBuffers = &cmdBuff,
                               .signalSemaphoreCount = 1,
                               .pSignalSemaphores = &postCompleteSemaphore};
    {
        gfxstream::base::AutoLock lock(*m_compositorVkQueueLock);
        VK_CHECK(m_vk.vkQueueSubmit(m_compositorVkQueue, 1, &submitInfo, postCompleteFence));
    }
    std::shared_future<std::shared_ptr<PostResource>> postResourceFuture =
        std::async(std::launch::deferred, [postCompleteFence, postResource, this,
                                           imResources]() mutable {
            VkResult res = m_vk.vkWaitForFences(m_vkDevice, 1, &postCompleteFence, VK_TRUE,
                                                kVkWaitForFencesTimeoutNsecs);
            if (res == VK_TIMEOUT) {
                // Retry. If device lost, hopefully this returns immediately.
                res = m_vk.vkWaitForFences(m_vkDevice, 1, &postCompleteFence, VK_TRUE,
                                           kVkWaitForFencesTimeoutNsecs);
            }
            VK_CHECK(res);

            // This should always be waited even in failure
            if (imResources) {
                m_compositorVk->releaseImmediateModeResources(imResources);
            }
            return postResource;
        }).share();
    m_postResourceFutures[imageIndex] = postResourceFuture;

    auto swapChain = m_swapChainStateVk->getSwapChain();
    VkPresentInfoKHR presentInfo = {.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                                    .waitSemaphoreCount = 1,
                                    .pWaitSemaphores = &postCompleteSemaphore,
                                    .swapchainCount = 1,
                                    .pSwapchains = &swapChain,
                                    .pImageIndices = &imageIndex};
    VkResult presentRes;
    {
        gfxstream::base::AutoLock lock(*m_swapChainVkQueueLock);
        presentRes = m_vk.vkQueuePresentKHR(m_swapChainVkQueue, &presentInfo);
    }
    if (shouldRecreateSwapchain(presentRes)) {
        postResourceFuture.wait();
        return PostResult{false, std::shared_future<void>()};
    }
    VK_CHECK(presentRes);
    return PostResult{true, std::async(std::launch::deferred, [postResourceFuture] {
                                // We can't directly wait for the VkFence here, because we
                                // share the VkFences on different frames, but we don't share
                                // the future on different frames. If we directly wait for the
                                // VkFence here, we may wait for a different frame if a new
                                // frame starts to be drawn before this future is waited.
                                postResourceFuture.wait();
                            }).share()};
}

VkFormatFeatureFlags DisplayVk::getFormatFeatures(VkFormat format, VkImageTiling tiling) {
    auto i = m_vkFormatProperties.find(format);
    if (i == m_vkFormatProperties.end()) {
        VkFormatProperties formatProperties;
        m_vk.vkGetPhysicalDeviceFormatProperties(m_vkPhysicalDevice, format, &formatProperties);
        i = m_vkFormatProperties.emplace(format, formatProperties).first;
    }
    const VkFormatProperties& formatProperties = i->second;
    VkFormatFeatureFlags formatFeatures = 0;
    if (tiling == VK_IMAGE_TILING_LINEAR) {
        formatFeatures = formatProperties.linearTilingFeatures;
    } else if (tiling == VK_IMAGE_TILING_OPTIMAL) {
        formatFeatures = formatProperties.optimalTilingFeatures;
    } else {
        GFXSTREAM_ERROR("Unknown tiling %#" PRIx64 ".", static_cast<uint64_t>(tiling));
    }
    return formatFeatures;
}

bool DisplayVk::canPost(const VkImageCreateInfo& postImageCi) {
    // According to VUID-vkCmdBlitImage-srcImage-01999, the format features of srcImage must contain
    // VK_FORMAT_FEATURE_BLIT_SRC_BIT.
    VkFormatFeatureFlags formatFeatures = getFormatFeatures(postImageCi.format, postImageCi.tiling);
    if (!(formatFeatures & VK_FORMAT_FEATURE_BLIT_SRC_BIT)) {
        GFXSTREAM_ERROR(
            "VK_FORMAT_FEATURE_BLIT_SRC_BLIT is not supported for VkImage with format %s, tilling "
            "%s. Supported features are %s.",
            string_VkFormat(postImageCi.format), string_VkImageTiling(postImageCi.tiling),
            string_VkFormatFeatureFlags(formatFeatures).c_str());
        return false;
    }

    // According to VUID-vkCmdBlitImage-srcImage-06421, srcImage must not use a format that requires
    // a sampler Y’CBCR conversion.
    if (formatRequiresSamplerYcbcrConversion(postImageCi.format)) {
        GFXSTREAM_ERROR("Format %s requires a sampler Y'CbCr conversion. Can't be used to post.",
                        string_VkFormat(postImageCi.format));
        return false;
    }

    if (!(postImageCi.usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) &&
        !(postImageCi.usage & VK_IMAGE_USAGE_SAMPLED_BIT)) {
        // According to VUID-vkCmdBlitImage-srcImage-00219, srcImage must have been created with
        // VK_IMAGE_USAGE_TRANSFER_SRC_BIT usage flag.
        // If we use drawImage, we need VK_IMAGE_USAGE_SAMPLED_BIT.
        GFXSTREAM_ERROR(
            "The VkImage is not created with the VK_IMAGE_USAGE_TRANSFER_SRC_BIT or "
            "VK_IMAGE_USAGE_SAMPLED_BIT usage flag. The usage flags are %s.",
            string_VkImageUsageFlags(postImageCi.usage).c_str());
        return false;
    }

    VkFormat swapChainFormat = m_swapChainStateVk->getFormat();
    if (formatIsSInt(postImageCi.format) || formatIsSInt(swapChainFormat)) {
        // According to VUID-vkCmdBlitImage-srcImage-00229, if either of srcImage or dstImage was
        // created with a signed integer VkFormat, the other must also have been created with a
        // signed integer VkFormat.
        if (!(formatIsSInt(postImageCi.format) && formatIsSInt(m_swapChainStateVk->getFormat()))) {
            GFXSTREAM_ERROR(
                "The format(%s) doesn't match with the format of the presentable image(%s): either "
                "of the formats is a signed integer VkFormat, but the other is not.",
                string_VkFormat(postImageCi.format), string_VkFormat(swapChainFormat));
            return false;
        }
    }

    if (formatIsUInt(postImageCi.format) || formatIsUInt(swapChainFormat)) {
        // According to VUID-vkCmdBlitImage-srcImage-00230, if either of srcImage or dstImage was
        // created with an unsigned integer VkFormat, the other must also have been created with an
        // unsigned integer VkFormat.
        if (!(formatIsUInt(postImageCi.format) && formatIsUInt(swapChainFormat))) {
            GFXSTREAM_ERROR(
                "The format(%s) doesn't match with the format of the presentable image(%s): either "
                "of the formats is an unsigned integer VkFormat, but the other is not.",
                string_VkFormat(postImageCi.format), string_VkFormat(swapChainFormat));
            return false;
        }
    }

    if (formatIsDepthOrStencil(postImageCi.format) || formatIsDepthOrStencil(swapChainFormat)) {
        // According to VUID-vkCmdBlitImage-srcImage-00231, if either of srcImage or dstImage was
        // created with a depth/stencil format, the other must have exactly the same format.
        if (postImageCi.format != swapChainFormat) {
            GFXSTREAM_ERROR(
                "The format(%s) doesn't match with the format of the presentable image(%s): either "
                "of the formats is a depth/stencil VkFormat, but the other is not the same format.",
                string_VkFormat(postImageCi.format), string_VkFormat(swapChainFormat));
            return false;
        }
    }

    if (postImageCi.samples != VK_SAMPLE_COUNT_1_BIT) {
        // According to VUID-vkCmdBlitImage-srcImage-00233, srcImage must have been created with a
        // samples value of VK_SAMPLE_COUNT_1_BIT.
        GFXSTREAM_ERROR(
            "The VkImage is not created with the VK_SAMPLE_COUNT_1_BIT samples value. The samples "
            "value is %s.",
            string_VkSampleCountFlagBits(postImageCi.samples));
        return false;
    }
    if (postImageCi.flags & VK_IMAGE_CREATE_SUBSAMPLED_BIT_EXT) {
        // According to VUID-vkCmdBlitImage-dstImage-02545, dstImage and srcImage must not have been
        // created with flags containing VK_IMAGE_CREATE_SUBSAMPLED_BIT_EXT.
        GFXSTREAM_ERROR(
            "The VkImage can't be created with flags containing "
            "VK_IMAGE_CREATE_SUBSAMPLED_BIT_EXT. The flags are %s.",
            string_VkImageCreateFlags(postImageCi.flags).c_str());
        return false;
    }
    return true;
}

std::shared_ptr<DisplayVk::PostResource> DisplayVk::PostResource::create(
    const VulkanDispatch& vk, VkDevice vkDevice, VkCommandPool vkCommandPool) {
    VkFenceCreateInfo fenceCi = {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
    };
    VkFence fence;
    VK_CHECK(vk.vkCreateFence(vkDevice, &fenceCi, nullptr, &fence));

    VkSemaphore semaphores[2];
    for (uint32_t i = 0; i < std::size(semaphores); i++) {
        VkSemaphoreCreateInfo semaphoreCi = {
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        };
        VK_CHECK(vk.vkCreateSemaphore(vkDevice, &semaphoreCi, nullptr, &semaphores[i]));
    }
    VkCommandBuffer commandBuffer;
    VkCommandBufferAllocateInfo commandBufferAllocInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = vkCommandPool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    VK_CHECK(vk.vkAllocateCommandBuffers(vkDevice, &commandBufferAllocInfo, &commandBuffer));

    return std::shared_ptr<PostResource>(new PostResource(
        vk, vkDevice, vkCommandPool, fence, semaphores[0], semaphores[1], commandBuffer));
}

DisplayVk::PostResource::~PostResource() {
    m_vk.vkFreeCommandBuffers(m_vkDevice, m_vkCommandPool, 1, &m_vkCommandBuffer);
    m_vk.vkDestroyFence(m_vkDevice, m_swapchainImageReleaseFence, nullptr);
    m_vk.vkDestroySemaphore(m_vkDevice, m_swapchainImageAcquireSemaphore, nullptr);
    m_vk.vkDestroySemaphore(m_vkDevice, m_swapchainImageReleaseSemaphore, nullptr);
}

DisplayVk::PostResource::PostResource(const VulkanDispatch& vk, VkDevice vkDevice,
                                      VkCommandPool vkCommandPool,
                                      VkFence swapchainImageReleaseFence,
                                      VkSemaphore swapchainImageAcquireSemaphore,
                                      VkSemaphore swapchainImageReleaseSemaphore,
                                      VkCommandBuffer vkCommandBuffer)
    : m_swapchainImageReleaseFence(swapchainImageReleaseFence),
      m_swapchainImageAcquireSemaphore(swapchainImageAcquireSemaphore),
      m_swapchainImageReleaseSemaphore(swapchainImageReleaseSemaphore),
      m_vkCommandBuffer(vkCommandBuffer),
      m_vk(vk),
      m_vkDevice(vkDevice),
      m_vkCommandPool(vkCommandPool) {}

std::unique_ptr<DisplayVk::ImageBorrowResource> DisplayVk::ImageBorrowResource::create(
    const VulkanDispatch& vk, VkDevice device, VkCommandPool commandPool) {
    const VkCommandBufferAllocateInfo allocInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext = nullptr,
        .commandPool = commandPool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    VK_CHECK(vk.vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer));
    const VkFenceCreateInfo fenceCi = {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_FENCE_CREATE_SIGNALED_BIT,
    };
    VkFence fence = VK_NULL_HANDLE;
    VK_CHECK(vk.vkCreateFence(device, &fenceCi, nullptr, &fence));
    return std::unique_ptr<ImageBorrowResource>(
        new ImageBorrowResource(vk, device, commandPool, fence, commandBuffer));
}

DisplayVk::ImageBorrowResource::~ImageBorrowResource() {
    m_vk.vkDestroyFence(m_vkDevice, m_completeFence, nullptr);
    m_vk.vkFreeCommandBuffers(m_vkDevice, m_vkCommandPool, 1, &m_vkCommandBuffer);
}

DisplayVk::ImageBorrowResource::ImageBorrowResource(const VulkanDispatch& vk, VkDevice device,
                                                    VkCommandPool commandPool, VkFence fence,
                                                    VkCommandBuffer commandBuffer)
    : m_completeFence(fence),
      m_vkCommandBuffer(commandBuffer),
      m_vk(vk),
      m_vkDevice(device),
      m_vkCommandPool(commandPool) {}

}  // namespace vk
}  // namespace host
}  // namespace gfxstream
