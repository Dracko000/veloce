Product Requirements Document (PRD)
​Project Name: Veloce (Low-Bit, Zero-Allocation LLM Programming Language & Compiler)
Version: 1.0
Status: Draft / Architectural Blueprint
​1. Executive Summary
​Veloce is a domain-specific programming language (DSL) and ahead-of-time (AOT) optimizing compiler engineered specifically for Large Language Model (LLM) development, quantization, and inference. Unlike general-purpose languages (Python, C++, Rust) that suffer from abstraction overhead, dynamic memory fragmentation, deep nested loops, and heavy reliance on discrete GPUs, Veloce treats sub-byte quantization (e.g., 1.58-bit ternary, 4-bit integer) as first-class primitives, eliminates explicit looping via declarative tensor algebra, and enforces static compile-time memory planning.
​2. Problem Statement
​Modern LLM development faces critical bottlenecks:
​1. Memory Wall: High-precision weights (fp16/fp32) and unoptimized KV-caches require massive VRAM/RAM, pricing out edge and commodity CPU hardware.
​2. GPU Lock-in: Current frameworks rely heavily on proprietary GPU ecosystems (e.g., CUDA) for acceptable throughput.
​3. Compute & Loop Overhead: Imperative nested loops cause severe CPU cache misses, while fragmented frameworks introduce kernel-launch and intermediate memory-read/write bottlenecks.
​4. Language Complexity: Existing systems languages impose steep cognitive loads (borrow checkers, complex template metaprogramming, async function coloring) that slow down AI engineering iteration.
​3. Goals & Success Metrics
​Memory Footprint Reduction: Reduce runtime RAM usage by \ge 80\% compared to fp16 PyTorch baselines via native trit (1.58-bit) packing and static buffer reuse.
​Hardware Independence: Achieve real-time token generation (\ge 15\text{ tokens/sec} on 7B-class ternary models) on standard multi-core CPUs (x86 AVX-512 / ARM NEON) without a discrete GPU.
​Zero Runtime Allocation: 0\text{ bytes} dynamically allocated on the heap during the inference loop.
​Compilation Speed: Compile standard Transformer architectures in < 2\text{ seconds}.
​4. Target Audience
​AI Systems Engineers: Building ultra-fast inference engines for edge devices, mobile hardware, and CPU-only cloud instances.
​LLM Researchers: Experimenting with extreme quantization (BitNet / 1.58-bit), custom attention mechanisms, and MatMul-free architectures without writing raw C++ intrinsics.
​5. Core Functional Requirements
​5.1 First-Class Sub-Byte Type System
​FR-1.1 Native Low-Bit Primitives: Support trit (\{-1, 0, 1\}, 1.58-bit), i2, i4, fp8, and fp16 as built-in scalar and tensor element types.
​FR-1.2 Automatic Bit-Packing: The compiler must automatically pack multiple sub-byte values into contiguous machine words (e.g., packing 5 trit values into a single u8 byte) at compile time or model load time without manual bitwise operations from the user.
​FR-1.3 MatMul-Free Lowering: Matrix multiplications involving trit tensors must automatically lower to pure integer addition/subtraction (ADD/SUB) instructions, bypassing floating-point multipliers.
​5.2 Loop-Free Declarative Tensor Algebra & Kernel Fusion
​FR-2.1 Declarative Syntax: Eliminate imperative for/while loops for tensor operations. Provide native Einstein-summation (einsum) and linear algebra operators (@, *, +).
​FR-2.2 Automatic Operator Fusion: The compiler IR must detect contiguous element-wise and reduction chains (e.g., RMSNorm -> Linear -> SiLU -> ResidualAdd) and fuse them into a single hardware-level loop pass inside CPU registers, eliminating intermediate RAM round-trips.
​FR-2.3 Implicit Vectorization: Automatically map fused kernels to target SIMD instruction sets (AVX2, AVX-512, ARM NEON/SVE).
​5.3 Static Memory Arena & Zero-Copy I/O
​FR-3.1 Static Liveness Analysis: Compute exact tensor lifetimes across the computation graph at compile time.
​FR-3.2 In-Place Buffer Reuse: Assign non-overlapping tensor activations to the exact same physical memory offsets within a pre-allocated static Arena, bounding peak activation memory to the size of the single largest layer.
​FR-3.3 Zero-Copy Model Loading: Support direct mmap loading of serialized .veloce weight binaries whose on-disk byte alignment matches the compiler's internal packed memory layout.
​5.4 Hardware-Agnostic Backend & Concurrency
​FR-4.1 Multi-Target Codegen: Lower the optimized Intermediate Representation (IR) to LLVM IR / C23 with SIMD pragmas, supporting x86_64, ARM64, and optional Vulkan/WebGPU compute shaders.
​FR-4.2 Implicit Cache-Aware Parallelism: Automatically partition tensor workloads across physical CPU cores based on L1/L2/L3 cache topology without user-managed threads or async/await.
​6. Non-Functional Requirements
​Determinism: Execution time and memory footprint must be 100% deterministic across runs.
​Minimal Toolchain: Distributed as a single standalone compiler binary (velocec) with zero external runtime dependencies.
​Interoperability: Direct C ABI compatibility (extern "C") so compiled models can be embedded into Rust, Go, C++, or Python hosts effortlessly.
​7. Default Model Capability Profile
​Every model built with Veloce declares a standard capability profile in its .veloce model card, targeting four pillars:
​
​- Deeper knowledge: more accurate answers in specialist topics.
​- Richer world understanding: real-world context reasoning, not just text patterns.
​- Stronger coding and agentic: plan, call tools, complete multi-step tasks.
​- Real-world task focus: complex coding, front-end builds, professional workflows.
​
​The profile is declarative: each pillar carries a status of declared, trained, or verified, and trained/verified claims require attached evidence. The compiler and runtime cannot supply these capabilities on their own; they come from the trained model (data, scale, post-training, evaluation). See docs/model-capabilities.md.
​
​8. Out of Scope (v1.0)
​Dynamic computational graphs (eager execution mode).
​Distributed multi-node training across clusters.
​General-purpose OS abstractions (networking, GUI, file-system management beyond model mmap).
​Training or fine-tuning at 100M+ parameter scale (the v1.0 toolchain exports and serves such models; producing their weights is out of scope).
