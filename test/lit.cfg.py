# -*- Python -*-

import os
import lit.formats
import lit.util
from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst

# Configuration file for the 'lit' test runner.

config.name = "SPMD"
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)
config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.spmd_obj_root, "test")

config.substitutions.append(("%PATH%", config.environment["PATH"]))
config.substitutions.append(("%shlibext", config.llvm_shlib_ext))

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])

# Propagate LLVM shared-library directory so spmd-opt can find its RPATH libs
# on systems where the default GLIBC version doesn't satisfy the build target.
llvm_libs_dir = os.path.join(os.path.dirname(config.llvm_tools_dir), "lib")
llvm_config.with_environment("LD_LIBRARY_PATH", llvm_libs_dir, append_path=True)
llvm_config.use_default_substitutions()

config.excludes = ["Inputs", "CMakeLists.txt", "README.txt"]

config.spmd_tools_dir = os.path.join(config.spmd_obj_root, "bin")
config.spmd_libs_dir = os.path.join(config.spmd_obj_root, "lib")

llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)
llvm_config.with_environment("PATH", config.spmd_tools_dir, append_path=True)

tools = ["spmd-opt", "FileCheck", "mlir-opt", "mlir-translate", "llc"]
tool_dirs = [config.spmd_tools_dir, config.llvm_tools_dir]
llvm_config.add_tool_substitutions(tools, tool_dirs)

# Detect whether the NVPTX backend is available in llc.
import subprocess
_llc_path = os.path.join(config.llvm_tools_dir, "llc")
try:
    result = subprocess.run(
        [_llc_path, "--version"], capture_output=True, text=True)
    if "nvptx64" in result.stdout.lower():
        config.available_features.add("nvptx-registered-target")
except Exception:
    pass
