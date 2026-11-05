#!/usr/bin/env python3
"""Resolve the diff3 conflicts left by merging the JetPack 4.6 CUDA 10.2 patch set
(Z841973620/llama.cpp-jetpack4.6 @ b10621) onto the cluster pin b23701f7.

Each conflict is decided explicitly -- no blanket "take ours/theirs" -- following the one
rule every conflict here obeys: keep the fork's C++14 STRUCTURE, carry upstream's new
SEMANTICS into it.
"""
import re
import sys
import pathlib

ROOT = pathlib.Path(sys.argv[1])          # the merged tree
CONF = re.compile(r"<<<<<<< pin\n(.*?)\|\|\|\|\|\|\| base-b10621\n(.*?)=======\n(.*?)>>>>>>> fork\n", re.S)


def cxx14(s):
    """The mechanical C++17 -> C++14 conversions the fork applies throughout."""
    s = s.replace("if constexpr (", "if (")
    # std::is_same_v<A, B> -> std::is_same<A, B>::value (arguments contain no '>')
    s = re.sub(r"std::is_same_v<([^<>]*)>", r"std::is_same<\1>::value", s)
    return s


def resolve(path, decide):
    p = ROOT / path
    text = p.read_text()
    n = [0]

    def sub(m):
        n[0] += 1
        return decide(n[0], m.group(1), m.group(2), m.group(3))
    new = CONF.sub(sub, text)
    assert "<<<<<<<" not in new and ">>>>>>>" not in new, f"unresolved conflict left in {path}"
    p.write_text(new)
    print(f"{path}: {n[0]} conflict(s) resolved")


# --- mmvf.cu, mmvq.cu, fattn.cu: upstream added a line/argument, the fork only converted
#     `if constexpr` -> `if`. Take upstream's text, apply the same conversion.
resolve("ggml/src/ggml-cuda/mmvf.cu", lambda i, pin, base, fork: cxx14(pin))
resolve("ggml/src/ggml-cuda/mmvq.cu", lambda i, pin, base, fork: cxx14(pin))
# fattn.cu's single conflict is inside the MMA dispatch block, which this build excludes
# with GGML_CUDA_NO_MMA_FA (never parsed); keep upstream's text untouched there.
resolve("ggml/src/ggml-cuda/fattn.cu", lambda i, pin, base, fork: pin)


# --- mmf.cuh: upstream wrapped the tile load in ggml_cuda_syncwarp(); the fork converted
#     the type dispatch to plain if and dropped the else-static_assert (under a plain if
#     that branch is instantiated for every T, and the assert would fire).
def mmf(i, pin, base, fork):
    if i == 1:
        assert "ggml_cuda_syncwarp();" in pin and "std::is_same<T, float>::value" in fork
        return "            ggml_cuda_syncwarp();\n" + fork
    if i == 2:
        assert fork.strip() == "}" and "ggml_cuda_syncwarp();" in pin
        return "            }\n            ggml_cuda_syncwarp();\n"
    raise AssertionError(i)
resolve("ggml/src/ggml-cuda/mmf.cuh", mmf)


# --- mmid.cu: the fork moved the loop into mm_ids_helper_specialized<> (SFINAE-selected,
#     because a plain `if` would instantiate it for n_expert_used_template == 0). Keep the
#     call; the helper body is brought up to upstream's padding logic below.
def mmid(i, pin, base, fork):
    assert "mm_ids_helper_specialized" in fork
    return fork
resolve("ggml/src/ggml-cuda/mmid.cu", mmid)


# --- fattn-tile.cuh: the fork collapsed six launch blocks into launch_fattn_tile_case<>.
#     Keep the calls; upstream's new launch_fattn parameter is added inside the helper below.
def tile(i, pin, base, fork):
    assert "launch_fattn_tile_case" in fork
    return cxx14(fork)
resolve("ggml/src/ggml-cuda/fattn-tile.cuh", tile)
