#!/usr/bin/env python3
"""Apply the bf16 matmul packer_l1_acc fix to TTNNPipelines.cpp"""

import sys

FILEPATH = "/workspace/tt-xla/third_party/tt-mlir/src/tt-mlir/lib/Dialect/TTNN/Pipelines/TTNNPipelines.cpp"

OLD_CODE = """\
    setConfigOptions.fp32DestAccEn = options.computeCfgFp32DestAccEn.getValue();

    if (setConfigOptions.fp32DestAccEn ||"""

NEW_CODE = """\
    setConfigOptions.fp32DestAccEn = options.computeCfgFp32DestAccEn.getValue();

    // When fp32_dest_acc_en is true, also enable packer_l1_acc.
    // Without this, the explicit DeviceComputeKernelConfigAttr built by
    // tt-xla disables the L1 packing precision that TTNN's
    // init_device_compute_kernel_config() enables by default for bf16 ops
    // (default_l1_acc = !is_float_32). TTNNSetComputeKernelConfig.cpp only
    // sets packerL1Acc when the option is explicitly true, so we must mirror
    // the TTNN default here.
    // See: https://github.com/nvukobratTT/tt-xla/issues/1
    if (setConfigOptions.fp32DestAccEn) {
      setConfigOptions.packerL1Acc = true;
    }

    if (setConfigOptions.fp32DestAccEn ||"""

with open(FILEPATH, 'r') as f:
    content = f.read()

if OLD_CODE not in content:
    print("ERROR: Old code not found in file — patch may already be applied or file changed")
    sys.exit(1)

new_content = content.replace(OLD_CODE, NEW_CODE, 1)

with open(FILEPATH, 'w') as f:
    f.write(new_content)

print("bf16 matmul packer_l1_acc fix applied successfully")
