import lit.formats
import os

config.name = "SPMD Dialect"
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)
config.suffixes = ['.mlir']
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.spmd_obj_root, 'test')

config.substitutions.append(('%PATH%', config.environment['PATH']))
config.substitutions.append(('%shlibext', config.llvm_shlib_ext))

llvm_config.with_system_environment(['HOME', 'INCLUDE', 'LIB', 'TMP', 'TEMP'])
llvm_config.use_default_substitutions()
llvm_config.add_tool_substitutions(['spmd-opt', 'FileCheck'], config.llvm_tools_dir)
