#!/usr/bin/env python3
"""Apply the LinearOp compute_config serialization fix to TTNNToFlatbuffer.cpp"""

import sys

FILEPATH = "/workspace/tt-xla/third_party/tt-mlir/src/tt-mlir/lib/Target/TTNN/TTNNToFlatbuffer.cpp"

OLD_CODE = """\
  auto activation = toFlatbuffer(cache, op.getActivation()).value_or(0);
  return ::tt::target::ttnn::CreateLinearOp(
      *cache.fbb, a, b, bias, output, op.getTransposeA(), op.getTransposeB(),
      matmulProgramConfigType, matmulProgramConfigDesc, activation);
}

// ANCHOR: adding_an_op_matmul_serialize_to_binary"""

NEW_CODE = """\
  auto activation = toFlatbuffer(cache, op.getActivation()).value_or(0);

  // Serialize compute_config — same as MatmulOp.
  // Without this, when MatmulWithBiasFusionPattern fires (matmul+add → LinearOp),
  // fp32_dest_acc_en and packer_l1_acc settings are silently dropped at the
  // flatbuffer boundary, causing precision regression vs plain matmul.
  // See: https://github.com/nvukobratTT/tt-xla/issues/1
  std::optional<
      ::flatbuffers::Offset<::tt::target::ttnn::DeviceComputeKernelConfig>>
      computeConfig = toFlatbuffer(cache, op.getComputeConfig());

  return ::tt::target::ttnn::CreateLinearOp(
      *cache.fbb, a, b, bias, output, op.getTransposeA(), op.getTransposeB(),
      matmulProgramConfigType, matmulProgramConfigDesc, activation,
      computeConfig.value_or(0));
}

// ANCHOR: adding_an_op_matmul_serialize_to_binary"""

with open(FILEPATH, 'r') as f:
    content = f.read()

if OLD_CODE not in content:
    print("ERROR: Old code not found in file — patch may already be applied or file changed")
    print("Looking for activation/CreateLinearOp block...")
    idx = content.find('  auto activation = toFlatbuffer(cache, op.getActivation()).value_or(0);\n  return ::tt::target::ttnn::CreateLinearOp(')
    if idx >= 0:
        print(f"Found at index {idx}, surrounding context:")
        print(content[idx:idx+300])
    sys.exit(1)

new_content = content.replace(OLD_CODE, NEW_CODE, 1)

with open(FILEPATH, 'w') as f:
    f.write(new_content)

print("LinearOp compute_config fix applied successfully")
