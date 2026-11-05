#!/usr/bin/env python3
"""Query the Vulkan feature bits that decide ggml-vulkan's fp16 path, without headers.

Old vulkaninfo (Ubuntu 18.04) prints only core VkPhysicalDeviceFeatures, never the
pNext-chained extension structs, so `vulkaninfo | grep shaderFloat16` is silent whether
the bit is true or false. This asks the driver directly via vkGetPhysicalDeviceFeatures2.
Python 3.6 compatible; needs only libvulkan.so.1.

Run from the control node; nothing is installed on the board:
    ssh jetson 'python3 -' < testbed/scripts/vk_fp16_probe.py
First run 2026-09-25 on the Nano (L4T R32.7.6): shaderFloat16 True -- see
testbed/node-jetson.md.
"""
import ctypes as C

vk = C.CDLL("libvulkan.so.1")

# --- structure type constants (from vulkan_core.h) ---
ST_APPLICATION_INFO = 0
ST_INSTANCE_CREATE_INFO = 1
ST_PHYSICAL_DEVICE_FEATURES_2 = 1000059000
ST_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES = 1000082000
ST_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES = 1000083000


def VK_MAKE_VERSION(major, minor, patch):
    return (major << 22) | (minor << 12) | patch


class VkApplicationInfo(C.Structure):
    _fields_ = [("sType", C.c_int32), ("pNext", C.c_void_p),
                ("pApplicationName", C.c_char_p), ("applicationVersion", C.c_uint32),
                ("pEngineName", C.c_char_p), ("engineVersion", C.c_uint32),
                ("apiVersion", C.c_uint32)]


class VkInstanceCreateInfo(C.Structure):
    _fields_ = [("sType", C.c_int32), ("pNext", C.c_void_p), ("flags", C.c_uint32),
                ("pApplicationInfo", C.POINTER(VkApplicationInfo)),
                ("enabledLayerCount", C.c_uint32), ("ppEnabledLayerNames", C.c_void_p),
                ("enabledExtensionCount", C.c_uint32), ("ppEnabledExtensionNames", C.c_void_p)]


class VkPhysicalDevice16BitStorageFeatures(C.Structure):
    _fields_ = [("sType", C.c_int32), ("pNext", C.c_void_p),
                ("storageBuffer16BitAccess", C.c_uint32),
                ("uniformAndStorageBuffer16BitAccess", C.c_uint32),
                ("storagePushConstant16", C.c_uint32),
                ("storageInputOutput16", C.c_uint32)]


class VkPhysicalDeviceShaderFloat16Int8Features(C.Structure):
    _fields_ = [("sType", C.c_int32), ("pNext", C.c_void_p),
                ("shaderFloat16", C.c_uint32), ("shaderInt8", C.c_uint32)]


class VkPhysicalDeviceFeatures2(C.Structure):
    # VkPhysicalDeviceFeatures is 55 VkBool32s; the probe does not need them by name.
    _fields_ = [("sType", C.c_int32), ("pNext", C.c_void_p),
                ("features", C.c_uint32 * 55)]


vk.vkCreateInstance.restype = C.c_int32
vk.vkCreateInstance.argtypes = [C.POINTER(VkInstanceCreateInfo), C.c_void_p, C.POINTER(C.c_void_p)]
vk.vkEnumeratePhysicalDevices.restype = C.c_int32
vk.vkEnumeratePhysicalDevices.argtypes = [C.c_void_p, C.POINTER(C.c_uint32), C.POINTER(C.c_void_p)]
vk.vkGetInstanceProcAddr.restype = C.c_void_p
vk.vkGetInstanceProcAddr.argtypes = [C.c_void_p, C.c_char_p]
vk.vkGetPhysicalDeviceProperties.restype = None
vk.vkGetPhysicalDeviceProperties.argtypes = [C.c_void_p, C.c_void_p]
vk.vkDestroyInstance.restype = None
vk.vkDestroyInstance.argtypes = [C.c_void_p, C.c_void_p]

app = VkApplicationInfo(ST_APPLICATION_INFO, None, b"hs-fp16-probe", 1, b"none", 0,
                        VK_MAKE_VERSION(1, 2, 0))
ici = VkInstanceCreateInfo(ST_INSTANCE_CREATE_INFO, None, 0, C.pointer(app), 0, None, 0, None)
inst = C.c_void_p()
res = vk.vkCreateInstance(C.byref(ici), None, C.byref(inst))
if res != 0:
    raise SystemExit("vkCreateInstance failed: VkResult=%d" % res)

n = C.c_uint32(0)
vk.vkEnumeratePhysicalDevices(inst, C.byref(n), None)
devs = (C.c_void_p * n.value)()
vk.vkEnumeratePhysicalDevices(inst, C.byref(n), devs)

PFN = C.CFUNCTYPE(None, C.c_void_p, C.POINTER(VkPhysicalDeviceFeatures2))
addr = vk.vkGetInstanceProcAddr(inst, b"vkGetPhysicalDeviceFeatures2")
if not addr:
    raise SystemExit("vkGetPhysicalDeviceFeatures2 not exposed by this loader/driver")
getf2 = PFN(addr)

for i in range(n.value):
    props = C.create_string_buffer(4096)          # VkPhysicalDeviceProperties is < 1 KB
    vk.vkGetPhysicalDeviceProperties(devs[i], props)
    raw = props.raw
    api = int.from_bytes(raw[0:4], "little")
    drv = int.from_bytes(raw[4:8], "little")
    name = raw[20:276].split(b"\0", 1)[0].decode()

    s16 = VkPhysicalDevice16BitStorageFeatures(ST_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES, None)
    f16 = VkPhysicalDeviceShaderFloat16Int8Features(
        ST_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES, C.cast(C.pointer(s16), C.c_void_p))
    f2 = VkPhysicalDeviceFeatures2(ST_PHYSICAL_DEVICE_FEATURES_2, C.cast(C.pointer(f16), C.c_void_p))
    getf2(devs[i], C.byref(f2))

    print("device                              : %s" % name)
    print("apiVersion                          : %d.%d.%d" % (api >> 22, (api >> 12) & 0x3FF, api & 0xFFF))
    print("driverVersion (raw)                 : 0x%08x" % drv)
    print("shaderFloat16                       : %s" % bool(f16.shaderFloat16))
    print("shaderInt8                          : %s" % bool(f16.shaderInt8))
    print("storageBuffer16BitAccess            : %s" % bool(s16.storageBuffer16BitAccess))
    print("uniformAndStorageBuffer16BitAccess  : %s" % bool(s16.uniformAndStorageBuffer16BitAccess))
    print("storagePushConstant16               : %s" % bool(s16.storagePushConstant16))
    print("storageInputOutput16                : %s" % bool(s16.storageInputOutput16))

vk.vkDestroyInstance(inst, None)
