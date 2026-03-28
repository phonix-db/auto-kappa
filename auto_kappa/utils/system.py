#
# cpu.py
#
# Copyright (c) 2026 Masato Ohnishi
#
# This file is distributed under the terms of the MIT license.
# Please see the file 'LICENCE.txt' in the root directory
# or http://opensource.org/licenses/mit-license.php for information.
#
import platform
import subprocess
import psutil
from pathlib import Path

def run(cmd):
    try:
        return subprocess.check_output(
            cmd, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None

def get_cpu_model():
    system = platform.system()
    
    if system == "Windows":
        return run([
            "powershell",
            "-NoProfile",
            "-Command",
            "(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)"
        ])
    
    elif system == "Darwin":  # macOS
        return run(["sysctl", "-n", "machdep.cpu.brand_string"])
    
    elif system == "Linux":
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.exists():
            for line in cpuinfo.read_text(errors="ignore").splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    
    return None

def get_cpu_info():
    freq = psutil.cpu_freq()
    return {
        "os": platform.system(),
        "cpu_model": get_cpu_model(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "freq_mhz_current": freq.current if freq else None,
        "freq_mhz_max": freq.max if freq else None,
    }

def get_os_info():
    system = platform.system()
    
    # --- Linux ---
    if system == "Linux":
        os_release = Path("/etc/os-release")
        info = {}
        
        if os_release.exists():
            for line in os_release.read_text(errors="ignore").splitlines():
                if "=" in line:
                    k, v = line.split("=", 1)
                    info[k] = v.strip().strip('"')
        
        return {
            "os_family": "Linux",
            "distro": info.get("NAME"),              # CentOS Stream, Ubuntu, Rocky Linux etc
            # "distro_id": info.get("ID"),              # centos, ubuntu, rocky
            "version": info.get("VERSION"),           # 7 (Core), 22.04 LTS
            # "version_id": info.get("VERSION_ID"),     # 7, 8, 22.04
            # "pretty_name": info.get("PRETTY_NAME"),   # readable name
            "kernel": platform.release(),
            "arch": platform.machine(),
        }
    
    # --- Windows ---
    if system == "Windows":
        return {
            "os_family": "Windows",
            "version": platform.version(),
            "release": platform.release(),
            "edition": platform.platform(),
            "arch": platform.machine(),
        }
    
    # --- macOS ---
    if system == "Darwin":
        return {
            "os_family": "macOS",
            "version": platform.mac_ver()[0],
            "arch": platform.machine(),
            "kernel": platform.release(),
        }
    
    return {"os_family": system}
