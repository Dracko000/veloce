Entity-Relationship Diagram (ERD) — Veloce Compiler & Runtime Architecture
​Since Veloce is a domain-specific programming language and static compiler rather than a traditional transactional web app, this ERD models the Compiler Internal Data Structures, Intermediate Representation (IR) Graph, Static Memory Planner, and Packed Tensor Storage Schema.
​1. Visual Entity-Relationship Diagram (Mermaid)

erDiagram
    MODULE ||--o{ FUNCTION : contains
    MODULE ||--o{ MEMORY_ARENA : declares
    MODULE ||--o{ WEIGHT_ASSET : maps

    FUNCTION ||--|{ IR_NODE : "consists of (DAG)"
    FUNCTION }o--|| MEMORY_ARENA : "allocates within"

    IR_NODE ||--o{ TENSOR_DESCRIPTOR : "produces / consumes"
    IR_NODE }o--o| FUSED_KERNEL : "grouped into"

    TENSOR_DESCRIPTOR }|--|| DATA_TYPE : "typed as"
    TENSOR_DESCRIPTOR ||--|| MEMORY_SLICE : "assigned offset in"

    MEMORY_ARENA ||--o{ MEMORY_SLICE : partitions
    WEIGHT_ASSET ||--|{ PACKED_BLOCK : "stored as"
    DATA_TYPE ||--o| PACKING_RULE : "governed by"

    MODULE {
        uuid module_id PK
        string name
        string target_arch "e.g., x86_64_avx512, arm64_neon"
        int cache_line_bytes "64 or 128"
    }

    FUNCTION {
        uuid function_id PK
        uuid module_id FK
        uuid arena_id FK
        string identifier
        bool is_pure
    }

    MEMORY_ARENA {
        uuid arena_id PK
        uuid module_id FK
        string name
        bigint total_capacity_bytes
        bigint peak_used_bytes
    }

    MEMORY_SLICE {
        uuid slice_id PK
        uuid arena_id FK
        bigint byte_offset
        bigint byte_length
        int birth_step "IR topological start"
        int death_step "IR topological end"
    }

    IR_NODE {
        uuid node_id PK
        uuid function_id FK
        uuid fused_kernel_id FK
        string op_code "MATMUL_TRIT, RMS_NORM, SILU, ROPE, ADD"
        int topological_order
        bool is_in_place
    }

    FUSED_KERNEL {
        uuid kernel_id PK
        string symbol_name
        string simd_strategy "AVX512_VNNI, NEON_DOTPROD"
        int register_tile_m
        int register_tile_n
    }

    TENSOR_DESCRIPTOR {
        uuid tensor_id PK
        uuid node_id FK
        uuid dtype_id FK
        uuid slice_id FK
        json shape "e.g., [1, 4096]"
        json strides
        bool is_constant_weight
    }

    DATA_TYPE {
        uuid dtype_id PK
        string name "trit, i2, i4, fp8, fp16"
        float bit_width "1.58, 2.0, 4.0, 8.0, 16.0"
        bool requires_bit_packing
    }

    PACKING_RULE {
        uuid rule_id PK
        uuid dtype_id FK
        int elements_per_word "e.g., 5 trits per 8-bit byte"
        int container_bits "8, 32, or 64"
        string decode_instruction "ADD_SUB_MASK, LUT_UNPACK"
    }

    WEIGHT_ASSET {
        uuid asset_id PK
        uuid module_id FK
        string file_path
        bigint file_size_bytes
        bool zero_copy_mmap
    }

    PACKED_BLOCK {
        uuid block_id PK
        uuid asset_id FK
        bigint file_offset
        bigint aligned_size_bytes
        string tensor_symbol
    }

2. Core Entity Descriptions

Entity Role in Compiler / Runtime
MODULE The top-level compilation unit representing an LLM architecture and its hardware compilation target (CPU SIMD capabilities and cache line size).
DATA_TYPE & PACKING_RULE Defines sub-byte primitives (trit, i4, etc.) and the exact mathematical bit-packing layout (e.g., base-3 encoding of 5 trit values into 1 uint8) so the compiler generates branchless unpacking instructions.
IR_NODE Represents a declarative mathematical operation in the Directed Acyclic Graph (DAG), completely free of explicit user-written loops.
FUSED_KERNEL Represents a merged subgraph of consecutive IR_NODEs (e.g., Normalization + Linear Projection + Activation) compiled into a single register-resident loop.
MEMORY_ARENA & MEMORY_SLICE Models the static memory planner. birth_step and death_step track tensor liveness so non-overlapping TENSOR_DESCRIPTORs share identical byte_offset addresses inside the MEMORY_ARENA.
WEIGHT_ASSET & PACKED_BLOCK Maps quantized model weights directly from disk (mmap) into aligned memory blocks matching the compiler's expected SIMD register layout without runtime deserialization.
