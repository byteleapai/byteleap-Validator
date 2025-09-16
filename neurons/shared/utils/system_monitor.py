"""
System Monitor
Cross-platform hardware and runtime information collection used by miner/worker
"""

import platform
import subprocess
import time
from typing import Any, Dict, List, Optional

import bittensor as bt
import GPUtil
import psutil
import requests

try:
    import py3nvml.py3nvml as nvml

    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

try:
    import cpuinfo

    CPUINFO_AVAILABLE = True
except ImportError:
    CPUINFO_AVAILABLE = False


class EnhancedSystemMonitor:
    """Enhanced system monitoring tool with improved cross-platform hardware detection"""

    def __init__(self):
        if NVML_AVAILABLE:
            try:
                nvml.nvmlInit()
                self._nvml_initialized = True
            except Exception:
                self._nvml_initialized = False
        else:
            self._nvml_initialized = False

    def get_system_info(self) -> Dict[str, Any]:
        return {
            "cpu_count": self.get_cpu_count(),
            "cpu_usage": self.get_cpu_usage(),
            "memory_total": self.get_memory_total(),
            "memory_available": self.get_memory_available(),
            "memory_usage": self.get_memory_usage(),
            "disk_total": self.get_disk_total(),
            "disk_free": self.get_disk_free(),
            "gpu_info": self.get_gpu_info(),
            "cpu_info": self.get_cpu_info(),
            "memory_info": self.get_memory_info(),
            "system_info": self.get_system_platform_info(),
            "motherboard_info": self.get_motherboard_info(),
            "uptime_seconds": self.get_system_uptime(),
            "public_ip": self.get_public_ip(),
        }

    def get_cpu_count(self) -> int:
        return psutil.cpu_count(logical=True)

    def get_cpu_usage(self) -> float:
        return psutil.cpu_percent(interval=None)

    def get_memory_total(self) -> int:
        return psutil.virtual_memory().total // (1024 * 1024)

    def get_memory_available(self) -> int:
        return psutil.virtual_memory().available // (1024 * 1024)

    def get_memory_usage(self) -> float:
        return psutil.virtual_memory().percent

    def get_disk_total(self) -> int:
        return psutil.disk_usage("/").total // (1024 * 1024 * 1024)

    def get_disk_free(self) -> int:
        return psutil.disk_usage("/").free // (1024 * 1024 * 1024)

    def get_gpu_info(self) -> List[Dict[str, Any]]:
        gpu_list = []
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                gpu_list.append(
                    {
                        "id": gpu.id,
                        "name": gpu.name,
                        "memory_total": gpu.memoryTotal,
                        "memory_used": gpu.memoryUsed,
                        "memory_free": gpu.memoryFree,
                        "memory_util": gpu.memoryUtil * 100,
                        "gpu_util": gpu.load * 100,
                        "temperature": gpu.temperature,
                        "vendor": "NVIDIA",
                        "type": "discrete",
                    }
                )
        except Exception:
            pass

        if not gpu_list and self._nvml_initialized:
            try:
                device_count = nvml.nvmlDeviceGetCount()
                for i in range(device_count):
                    handle = nvml.nvmlDeviceGetHandleByIndex(i)
                    name = nvml.nvmlDeviceGetName(handle).decode("utf-8")
                    memory_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                    util_rates = nvml.nvmlDeviceGetUtilizationRates(handle)
                    temp = nvml.nvmlDeviceGetTemperature(
                        handle, nvml.NVML_TEMPERATURE_GPU
                    )
                    gpu_list.append(
                        {
                            "id": i,
                            "name": name,
                            "memory_total": memory_info.total // (1024 * 1024),
                            "memory_used": memory_info.used // (1024 * 1024),
                            "memory_free": memory_info.free // (1024 * 1024),
                            "memory_util": (memory_info.used / memory_info.total) * 100,
                            "gpu_util": util_rates.gpu,
                            "temperature": temp,
                            "vendor": "NVIDIA",
                            "type": "discrete",
                        }
                    )
            except Exception:
                pass

        if not gpu_list and platform.system() == "Darwin":
            try:
                result = subprocess.run(
                    ["system_profiler", "SPDisplaysDataType", "-json"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    import json

                    data = json.loads(result.stdout)
                    displays = data.get("SPDisplaysDataType", [])
                    for i, display in enumerate(displays):
                        chip_name = display.get("_name", "Unknown GPU")
                        model = display.get("sppci_model", chip_name)
                        cores = display.get("sppci_cores", 0)
                        memory_total = 0
                        vram = display.get("spdisplays_vram", "0 MB")
                        if isinstance(vram, str):
                            parts = vram.split()
                            if len(parts) >= 2:
                                try:
                                    value = int(parts[0])
                                    unit = parts[1].upper()
                                    if unit.startswith("GB"):
                                        memory_total = value * 1024
                                    elif unit.startswith("MB"):
                                        memory_total = value
                                except ValueError:
                                    pass
                        if memory_total == 0 and (
                            "Apple" in chip_name
                            or "M1" in chip_name
                            or "M2" in chip_name
                            or "M3" in chip_name
                            or "M4" in chip_name
                        ):
                            try:
                                total_memory = psutil.virtual_memory().total // (
                                    1024 * 1024
                                )
                                memory_total = min(total_memory // 2, 32768)
                            except Exception:
                                memory_total = 8192
                        gpu_list.append(
                            {
                                "id": i,
                                "name": model or chip_name,
                                "memory_total": memory_total,
                                "memory_free": memory_total,
                                "vendor": "Apple",
                                "cores": cores,
                                "type": "integrated",
                                "platform": "apple_silicon",
                            }
                        )
            except Exception as e:
                bt.logging.debug(f"Failed to get macOS GPU info: {e}")

        return gpu_list

    def get_cpu_info(self) -> Dict[str, Any]:
        try:
            cpu_info: Dict[str, Any] = {}
            cpu_info["logical_cores"] = psutil.cpu_count(logical=True)
            cpu_info["physical_cores"] = psutil.cpu_count(logical=False)
            cpu_info["architecture"] = platform.machine()
            cpu_info["processor"] = platform.processor()
            try:
                freq = psutil.cpu_freq()
                if freq:
                    cpu_info["frequency_mhz"] = {
                        "current": freq.current,
                        "min": freq.min,
                        "max": freq.max,
                    }
            except Exception:
                pass
            if platform.system() == "Darwin":
                try:
                    result = subprocess.run(
                        ["system_profiler", "SPHardwareDataType", "-json"],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    if result.returncode == 0:
                        import json

                        data = json.loads(result.stdout)
                        hardware = data.get("SPHardwareDataType", [{}])[0]
                        chip_type = hardware.get("chip_type")
                        machine_model = hardware.get("machine_model")
                        machine_name = hardware.get("machine_name")
                        if chip_type:
                            cpu_info["brand"] = chip_type
                            cpu_info["model"] = chip_type
                            cpu_info["vendor_id"] = "Apple"
                        if machine_model:
                            cpu_info["machine_model"] = machine_model
                        if machine_name:
                            cpu_info["machine_name"] = machine_name
                        if chip_type:
                            if "M1" in chip_type:
                                cpu_info["family"] = "Apple Silicon M1"
                            elif "M2" in chip_type:
                                cpu_info["family"] = "Apple Silicon M2"
                            elif "M3" in chip_type:
                                cpu_info["family"] = "Apple Silicon M3"
                            elif "M4" in chip_type:
                                cpu_info["family"] = "Apple Silicon M4"
                            else:
                                cpu_info["family"] = "Apple Silicon"
                        processor_info = hardware.get("number_processors", "")
                        if processor_info and "proc" in processor_info:
                            parts = processor_info.replace("proc ", "").split(":")
                            if len(parts) >= 3:
                                cpu_info["total_cores"] = int(parts[0])
                                cpu_info["performance_cores"] = int(parts[1])
                                cpu_info["efficiency_cores"] = int(parts[2])
                        bt.logging.debug(f"macOS CPU detected: {chip_type}")
                except Exception as e:
                    bt.logging.debug(f"Could not get macOS CPU info: {e}")
            elif platform.system() == "Linux":
                try:
                    with open("/proc/cpuinfo", "r") as f:
                        cpuinfo_content = f.read()
                        for line in cpuinfo_content.split("\n"):
                            if "model name" in line:
                                cpu_info["model_name"] = line.split(":")[1].strip()
                                if (
                                    not cpu_info.get("brand")
                                    or cpu_info.get("brand") == "Unknown"
                                ):
                                    cpu_info["brand"] = cpu_info["model_name"]
                                break
                except Exception:
                    pass
            if CPUINFO_AVAILABLE:
                try:
                    detailed_info = cpuinfo.get_cpu_info()
                    if not cpu_info.get("brand") or cpu_info.get("brand") == "Unknown":
                        cpu_info["brand"] = detailed_info.get(
                            "brand_raw", detailed_info.get("brand", "Unknown")
                        )
                    if not cpu_info.get("model") or cpu_info.get("model") == "Unknown":
                        cpu_info["model"] = detailed_info.get("model", "Unknown")
                    if (
                        not cpu_info.get("family")
                        or cpu_info.get("family") == "Unknown"
                    ):
                        cpu_info["family"] = detailed_info.get("family", "Unknown")
                except Exception:
                    pass
            return cpu_info
        except Exception as e:
            bt.logging.warning(f"Error getting CPU info: {e}")
            return {"error": str(e)}

    def get_memory_info(self) -> Dict[str, Any]:
        try:
            vm = psutil.virtual_memory()
            sm = psutil.swap_memory()
            return {
                "total": vm.total // (1024 * 1024),
                "available": vm.available // (1024 * 1024),
                "used": vm.used // (1024 * 1024),
                "percent": vm.percent,
                "swap_total": sm.total // (1024 * 1024),
                "swap_used": sm.used // (1024 * 1024),
                "swap_percent": sm.percent,
            }
        except Exception as e:
            bt.logging.warning(f"Error getting memory info: {e}")
            return {"error": str(e)}

    def get_system_platform_info(self) -> Dict[str, Any]:
        try:
            return {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
            }
        except Exception as e:
            bt.logging.warning(f"Error getting system info: {e}")
            return {"error": str(e)}

    def get_motherboard_info(self) -> Dict[str, Any]:
        try:
            mb_info: Dict[str, Any] = {}
            if platform.system() == "Linux":
                try:
                    result = subprocess.run(
                        ["dmidecode", "-s", "baseboard-manufacturer"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    if result.returncode == 0:
                        mb_info["manufacturer"] = result.stdout.strip()
                    result = subprocess.run(
                        ["dmidecode", "-s", "baseboard-product-name"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    if result.returncode == 0:
                        mb_info["product_name"] = result.stdout.strip()
                    result = subprocess.run(
                        ["dmidecode", "-s", "system-serial-number"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    if result.returncode == 0:
                        serial = result.stdout.strip()
                        if serial and serial != "To Be Filled By O.E.M.":
                            mb_info["serial_number"] = serial
                except Exception as e:
                    bt.logging.debug(f"Could not get DMI info: {e}")
            elif platform.system() == "Darwin":
                try:
                    result = subprocess.run(
                        ["system_profiler", "SPHardwareDataType", "-json"],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    if result.returncode == 0:
                        import json

                        data = json.loads(result.stdout)
                        hardware = data.get("SPHardwareDataType", [{}])[0]
                        mb_info["brand"] = "Apple"
                        mb_info["model_identifier"] = hardware.get(
                            "machine_model", "Unknown"
                        )
                        mb_info["machine_name"] = hardware.get(
                            "machine_name", "Unknown"
                        )
                        mb_info["serial_number"] = hardware.get(
                            "serial_number", "Unknown"
                        )
                        mb_info["chip_type"] = hardware.get("chip_type", "Unknown")
                        physical_memory = hardware.get("physical_memory", "Unknown")
                        if physical_memory and physical_memory != "Unknown":
                            mb_info["physical_memory"] = physical_memory
                except Exception as e:
                    bt.logging.debug(f"Could not get macOS hardware info: {e}")
            elif platform.system() == "Windows":
                try:
                    result = subprocess.run(
                        [
                            "wmic",
                            "baseboard",
                            "get",
                            "manufacturer,product,serialnumber",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    if result.returncode == 0:
                        lines = result.stdout.strip().split("\n")
                        if len(lines) > 1:
                            values = lines[1].split()
                            if len(values) >= 3:
                                mb_info["manufacturer"] = values[0]
                                mb_info["product_name"] = values[1]
                                mb_info["serial_number"] = values[2]
                except Exception as e:
                    bt.logging.debug(f"Could not get Windows motherboard info: {e}")
            return mb_info
        except Exception as e:
            bt.logging.warning(f"Error getting motherboard info: {e}")
            return {"error": str(e)}

    def get_system_uptime(self) -> Optional[float]:
        try:
            return time.time() - psutil.boot_time()
        except Exception as e:
            bt.logging.warning(f"Error getting system uptime: {e}")
            return None

    def get_public_ip(self) -> Optional[str]:
        try:
            response = requests.get("https://httpbin.org/ip", timeout=5)
            if response.status_code == 200:
                return response.json().get("origin")
        except Exception:
            pass
        try:
            response = requests.get("https://api.ipify.org?format=json", timeout=5)
            if response.status_code == 200:
                return response.json().get("ip")
        except Exception:
            pass
        return None
