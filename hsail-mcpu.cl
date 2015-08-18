// Check that -mcpu works for all supported GPUs

// RUN: %clang -### -target hsail -x cl -S -emit-llvm < %s
// RUN: %clang -### -target hsail -x cl -S -emit-llvm -mcpu=kaveri %s -o - 2>&1 | FileCheck -check-prefix=KAVERI %s
// RUN: %clang -### -target hsail64 -x cl -S -emit-llvm -mcpu=kaveri %s -o - 2>&1 | FileCheck -check-prefix=KAVERI %s

// KAVERI:  "-target-cpu" "kaveri"
