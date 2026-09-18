// Stub Level Zero header — Mac-side SYNTAX CHECK ONLY for mojo_level_zero_rt.cpp.
// Never linked or shipped; the real build is g5_build_runtime.sh on Aurora.
#pragma once
#include <cstdint>
#include <cstddef>
typedef int ze_result_t;
#define ZE_RESULT_SUCCESS 0
typedef struct _zh { int x; } *ze_driver_handle_t;
typedef struct _zd { int x; } *ze_device_handle_t;
typedef struct _zc { int x; } *ze_context_handle_t;
typedef struct _zl { int x; } *ze_command_list_handle_t;
typedef struct _zm { int x; } *ze_module_handle_t;
typedef struct _zk { int x; } *ze_kernel_handle_t;
typedef struct _zb { int x; } *ze_module_build_log_handle_t;
enum { ZE_INIT_FLAG_GPU_ONLY = 1, ZE_DEVICE_TYPE_GPU = 1,
       ZE_MODULE_FORMAT_IL_SPIRV = 0,
       ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS = 1,
       ZE_COMMAND_QUEUE_PRIORITY_NORMAL = 0,
       ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE = 1,
       ZE_COMMAND_QUEUE_FLAG_IN_ORDER = 2,
       ZE_KERNEL_INDIRECT_ACCESS_FLAG_HOST = 1,
       ZE_KERNEL_INDIRECT_ACCESS_FLAG_DEVICE = 2,
       ZE_KERNEL_INDIRECT_ACCESS_FLAG_SHARED = 4,
       ZE_STRUCTURE_TYPE_CONTEXT_DESC = 1, ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
       ZE_STRUCTURE_TYPE_MODULE_DESC, ZE_STRUCTURE_TYPE_KERNEL_DESC,
       ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC,
       ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES, ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES };
struct ze_context_desc_t { int stype; const void *pNext; uint32_t flags; };
struct ze_command_queue_desc_t { int stype; const void *pNext; uint32_t ordinal, index, flags; int mode, priority; };
struct ze_command_queue_group_properties_t { int stype; void *pNext; uint32_t flags, maxMemoryFillPatternSize, numQueues; };
struct ze_module_desc_t { int stype; const void *pNext; int format; size_t inputSize; const uint8_t *pInputModule; const char *pBuildFlags; const void *pConstants; };
struct ze_kernel_desc_t { int stype; const void *pNext; uint32_t flags; const char *pKernelName; };
struct ze_device_mem_alloc_desc_t { int stype; const void *pNext; uint32_t flags, ordinal; };
struct ze_host_mem_alloc_desc_t { int stype; const void *pNext; uint32_t flags; };
struct ze_device_properties_t { int stype; void *pNext; int type; uint32_t vendorId, deviceId; char name[256]; };
struct ze_group_count_t { uint32_t groupCountX, groupCountY, groupCountZ; };
extern "C" {
ze_result_t zeInit(uint32_t);
ze_result_t zeDriverGet(uint32_t *, ze_driver_handle_t *);
ze_result_t zeDeviceGet(ze_driver_handle_t, uint32_t *, ze_device_handle_t *);
ze_result_t zeDeviceGetProperties(ze_device_handle_t, ze_device_properties_t *);
ze_result_t zeDeviceGetCommandQueueGroupProperties(ze_device_handle_t, uint32_t *, ze_command_queue_group_properties_t *);
ze_result_t zeContextCreate(ze_driver_handle_t, const ze_context_desc_t *, ze_context_handle_t *);
ze_result_t zeContextDestroy(ze_context_handle_t);
ze_result_t zeCommandListCreateImmediate(ze_context_handle_t, ze_device_handle_t, const ze_command_queue_desc_t *, ze_command_list_handle_t *);
ze_result_t zeCommandListDestroy(ze_command_list_handle_t);
ze_result_t zeCommandListHostSynchronize(ze_command_list_handle_t, uint64_t);
ze_result_t zeCommandListAppendBarrier(ze_command_list_handle_t, void *, uint32_t, void *);
ze_result_t zeCommandListAppendLaunchKernel(ze_command_list_handle_t, ze_kernel_handle_t, const ze_group_count_t *, void *, uint32_t, void *);
ze_result_t zeCommandListAppendMemoryCopy(ze_command_list_handle_t, void *, const void *, size_t, void *, uint32_t, void *);
ze_result_t zeCommandListAppendMemoryFill(ze_command_list_handle_t, void *, const void *, size_t, size_t, void *, uint32_t, void *);
ze_result_t zeModuleCreate(ze_context_handle_t, ze_device_handle_t, const ze_module_desc_t *, ze_module_handle_t *, ze_module_build_log_handle_t *);
ze_result_t zeModuleDestroy(ze_module_handle_t);
ze_result_t zeModuleBuildLogGetString(ze_module_build_log_handle_t, size_t *, char *);
ze_result_t zeModuleBuildLogDestroy(ze_module_build_log_handle_t);
ze_result_t zeKernelCreate(ze_module_handle_t, const ze_kernel_desc_t *, ze_kernel_handle_t *);
ze_result_t zeKernelDestroy(ze_kernel_handle_t);
ze_result_t zeKernelSetArgumentValue(ze_kernel_handle_t, uint32_t, size_t, const void *);
ze_result_t zeKernelSetGroupSize(ze_kernel_handle_t, uint32_t, uint32_t, uint32_t);
ze_result_t zeKernelSetIndirectAccess(ze_kernel_handle_t, uint32_t);
ze_result_t zeMemAllocDevice(ze_context_handle_t, const ze_device_mem_alloc_desc_t *, size_t, size_t, ze_device_handle_t, void **);
ze_result_t zeMemAllocHost(ze_context_handle_t, const ze_host_mem_alloc_desc_t *, size_t, size_t, void **);
ze_result_t zeMemFree(ze_context_handle_t, void *);
}
