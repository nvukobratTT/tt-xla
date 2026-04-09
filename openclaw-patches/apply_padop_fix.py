#!/usr/bin/env python3
"""Apply the PadOp row-major iteration fix to StableHLOToTTIRPatterns.cpp"""

import sys

FILEPATH = "/workspace/tt-xla/third_party/tt-mlir/src/tt-mlir/lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp"

OLD_CODE = """\
      llvm::SmallVector<int64_t> flatIndices;
      flatIndices.append(counters.begin(), counters.end());
      int64_t numIndices = 1;

      size_t current_index = 0;
      while (current_index < counters.size() &&
             counters[counters.size() - 1] <
                 upperbounds[upperbounds.size() - 1]) {
        counters[current_index] += steps[current_index];
        bool reset = false;
        while (current_index < counters.size() &&
               counters[current_index] >= upperbounds[current_index]) {
          counters[current_index] = lowerbounds[current_index];
          current_index++;
          if (current_index < counters.size()) {
            counters[current_index] += steps[current_index];
          }
          reset = true;
        }
        if (current_index >= counters.size()) {
          break;
        }
        if (reset) {
          current_index = 0;
        }
        flatIndices.append(counters.begin(), counters.end());
        numIndices++;
      }"""

NEW_CODE = """\
      llvm::SmallVector<int64_t> flatIndices;
      int64_t numIndices = 0;

      // Enumerate indices in row-major order (innermost dimension varies
      // fastest) so the order matches how flattenTensor linearizes the source
      // tensor. The previous outermost-first iteration produced column-major
      // index order, which mismatched the row-major-flattened update tensor and
      // placed source elements at wrong output positions.
      // See: https://github.com/nvukobratTT/tt-xla/issues/5
      std::function<void(int64_t)> enumerate = [&](int64_t dim) {
        if (static_cast<size_t>(dim) == counters.size()) {
          flatIndices.append(counters.begin(), counters.end());
          numIndices++;
          return;
        }
        for (counters[dim] = lowerbounds[dim];
             counters[dim] < upperbounds[dim]; counters[dim] += steps[dim]) {
          enumerate(dim + 1);
        }
      };
      enumerate(0);"""

with open(FILEPATH, 'r') as f:
    content = f.read()

if OLD_CODE not in content:
    print("ERROR: Old code not found in file — patch may already be applied or file changed")
    sys.exit(1)

new_content = content.replace(OLD_CODE, NEW_CODE, 1)

with open(FILEPATH, 'w') as f:
    f.write(new_content)

print("PadOp fix applied successfully")
