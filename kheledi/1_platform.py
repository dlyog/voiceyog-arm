#!/usr/bin/env python3
"""
What Arm part is this, and what can it actually execute?

One file that answers the same questions on Linux and macOS, because the
comparison this directory exists for is only meaningful if both machines are
described by the same code. Linux answers through MIDR_EL1, cpu_capacity and
AT_HWCAP2; macOS answers through sysctl. The output schema is identical
either way.

The ISA section is the point. KleidiAI's fp32 and convolution kernels in ONNX
Runtime are installed only when the CPU reports SME or SME2, so `sme` below is
the single field that decides whether KleidiAI is reachable on this machine at
all.

Self-contained: no imports from anywhere else in this repository, and nothing
here needs the model or a virtualenv.

    python3 1_platform.py            # human readable
    python3 1_platform.py --json     # machine readable
"""
import ctypes
import json
import os
import platform
import subprocess
import sys

# aarch64 Linux hwcap bit positions, from the kernel's
# Documentation/arch/arm64/elf_hwcaps.rst
AT_HWCAP, AT_HWCAP2 = 16, 26
HWCAP_SVE = 1 << 22
HWCAP_ASIMDDP = 1 << 20
HWCAP_ASIMDHP = 1 << 10
HWCAP2_SVE2 = 1 << 1
HWCAP2_I8MM = 1 << 13
HWCAP2_BF16 = 1 << 14
HWCAP2_SME = 1 << 23
HWCAP2_SME2 = 1 << 37


def _sysctl(name, kind=int):
    try:
        out = subprocess.run(["sysctl", "-n", name], capture_output=True,
                             text=True, check=True).stdout.strip()
        return kind(out) if kind is int else out
    except Exception:
        return None


def probe_macos():
    brand = _sysctl("machdep.cpu.brand_string", str) or "unknown"
    p_cores = _sysctl("hw.perflevel0.physicalcpu") or 0
    e_cores = _sysctl("hw.perflevel1.physicalcpu") or 0

    def feat(n):
        v = _sysctl(f"hw.optional.arm.{n}")
        return bool(v) if v is not None else False

    return {
        "os": "macOS",
        "os_version": platform.mac_ver()[0],
        "cpu": brand,
        "total_cores": (p_cores + e_cores) or os.cpu_count(),
        "performance_cores": p_cores,
        "efficiency_cores": e_cores,
        "performance_core_ids": None,   # macOS does not expose a stable mapping
        "asymmetric": e_cores > 0,
        "isa": {
            "sve": feat("FEAT_SVE"),
            "sve2": feat("FEAT_SVE2"),
            "i8mm": feat("FEAT_I8MM"),
            "bf16": feat("FEAT_BF16"),
            "dotprod": feat("FEAT_DotProd"),
            "fp16": feat("FEAT_FP16"),
            "sme": feat("FEAT_SME"),
            "sme2": feat("FEAT_SME2"),
        },
        "isa_source": "sysctl hw.optional.arm.*",
    }


def probe_linux():
    libc = ctypes.CDLL("libc.so.6")
    libc.getauxval.restype = ctypes.c_ulong
    libc.getauxval.argtypes = [ctypes.c_ulong]
    h1, h2 = libc.getauxval(AT_HWCAP), libc.getauxval(AT_HWCAP2)

    # cpu_capacity is what the scheduler itself uses to rank asymmetric cores.
    caps = {}
    for c in range(os.cpu_count() or 0):
        p = f"/sys/devices/system/cpu/cpu{c}/cpu_capacity"
        try:
            caps[c] = int(open(p).read().strip())
        except Exception:
            pass

    p_ids, e_ids = [], []
    if caps:
        hi = max(caps.values())
        # A generous split: anything within 15% of the fastest is a big core.
        for c, v in sorted(caps.items()):
            (p_ids if v >= 0.85 * hi else e_ids).append(c)

    midrs = set()
    for c in range(os.cpu_count() or 0):
        p = f"/sys/devices/system/cpu/cpu{c}/regs/identification/midr_el1"
        try:
            midrs.add(open(p).read().strip())
        except Exception:
            pass

    model = None
    try:
        out = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
        names = [l.split(":", 1)[1].strip() for l in out.splitlines()
                 if l.startswith("Model name:")]
        model = " + ".join(dict.fromkeys(names)) if names else None
    except Exception:
        pass

    try:
        osv = [l for l in open("/etc/os-release") if l.startswith("PRETTY_NAME")]
        osv = osv[0].split("=", 1)[1].strip().strip('"') if osv else ""
    except Exception:
        osv = ""

    return {
        "os": "Linux",
        "os_version": f"{osv} · kernel {platform.release()}",
        "cpu": model or "unknown",
        "total_cores": os.cpu_count(),
        "performance_cores": len(p_ids),
        "efficiency_cores": len(e_ids),
        "performance_core_ids": p_ids,
        "asymmetric": len(e_ids) > 0,
        "midr_el1": sorted(midrs),
        "cpu_capacity": caps,
        "isa": {
            "sve": bool(h1 & HWCAP_SVE),
            "sve2": bool(h2 & HWCAP2_SVE2),
            "i8mm": bool(h2 & HWCAP2_I8MM),
            "bf16": bool(h2 & HWCAP2_BF16),
            "dotprod": bool(h1 & HWCAP_ASIMDDP),
            "fp16": bool(h1 & HWCAP_ASIMDHP),
            "sme": bool(h2 & HWCAP2_SME),
            "sme2": bool(h2 & HWCAP2_SME2),
        },
        "isa_source": f"getauxval AT_HWCAP=0x{h1:x} AT_HWCAP2=0x{h2:x}",
    }


def tuned_threads(info):
    """Threads to give ONNX Runtime's intra-op pool.

    ONNX Runtime splits an operator across intra-op threads and joins them,
    and that join is a barrier: the operator finishes when its slowest thread
    does. On an asymmetric part one thread parked on an efficiency core holds
    up every other thread, on every operator. So the count comes from the
    number of PERFORMANCE cores, not the number of cores.

    The measured optimum is one fewer than the performance-core count on GB10
    (9 of 10) and the full count on M1 Max (8 of 8). Both are reproduced by
    reserving a core only when there are more than eight of them, which leaves
    one for the main thread and the OS on the larger part.
    """
    p = info.get("performance_cores") or 0
    if p <= 0:
        return max(1, (info.get("total_cores") or 2) // 2)
    return p - 1 if p > 8 else p


def probe():
    if sys.platform == "darwin":
        info = probe_macos()
    elif sys.platform.startswith("linux"):
        info = probe_linux()
    else:
        info = {"os": sys.platform, "cpu": platform.processor(),
                "total_cores": os.cpu_count(), "isa": {}, "asymmetric": False,
                "performance_cores": 0, "efficiency_cores": 0}
    info["machine"] = platform.machine()
    info["hostname"] = platform.node()
    info["python"] = platform.python_version()
    info["tuned_intra_op_threads"] = tuned_threads(info)
    isa = info.get("isa", {})
    info["kleidiai_reachable"] = bool(isa.get("sme") or isa.get("sme2"))
    info["kleidiai_note"] = (
        "ONNX Runtime installs every KleidiAI kernel behind "
        "HasArm_SME() || HasArm_SME2(). Without one of those this CPU never "
        "dispatches to KleidiAI, and executing its fp32 kernels anyway raises "
        "SIGILL.")
    return info


def main():
    info = probe()
    if "--json" in sys.argv:
        print(json.dumps(info, indent=2))
        return
    print(f"  host        {info['hostname']}  ({info['machine']})")
    print(f"  os          {info['os']} {info.get('os_version','')}")
    print(f"  cpu         {info['cpu']}")
    print(f"  cores       {info['total_cores']} total · "
          f"{info['performance_cores']} performance · {info['efficiency_cores']} efficiency"
          f"{'  (asymmetric)' if info['asymmetric'] else ''}")
    if info.get("performance_core_ids"):
        print(f"  fast core ids  {info['performance_core_ids']}")
    print(f"  threads     {info['tuned_intra_op_threads']}  (derived, not guessed)")
    print()
    isa = info.get("isa", {})
    order = ["sve", "sve2", "i8mm", "bf16", "dotprod", "fp16", "sme", "sme2"]
    print("  ISA")
    for k in order:
        if k in isa:
            mark = "yes" if isa[k] else "no"
            tail = "   <-- KleidiAI gate" if k in ("sme", "sme2") else ""
            print(f"    {k:<9} {mark}{tail}")
    print(f"  source      {info.get('isa_source','')}")
    print()
    print(f"  KleidiAI reachable on this CPU: "
          f"{'YES' if info['kleidiai_reachable'] else 'NO'}")


if __name__ == "__main__":
    main()
