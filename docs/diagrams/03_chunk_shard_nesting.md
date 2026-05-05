# Chunk, Shard & Superchunk Nesting

## Overview

The Zarr v3 output uses a **three-level nesting** for efficient parallel writes and reads. Understanding this hierarchy is critical for tuning performance and reasoning about I/O patterns.

```
Tile Volume (full image)
  └── Shard (file granularity — each shard = 1 S3 object)
        └── Chunk (compression unit — smallest addressable read)
```

On the **read side**, a fourth concept — the **Superchunk** — amortizes HDF5 I/O by reading a large contiguous region and slicing it into shard-sized blocks without additional disk access.

---

## Spatial Hierarchy Diagram

```mermaid
flowchart TD
    subgraph Volume["Full Tile Volume"]
        V["tile_000_ch_488.ims<br/>Shape: 768 × 10,752 × 14,336 (Z × Y × X)<br/>dtype: uint16<br/>Raw size: ~450 GB"]
    end

    subgraph ShardGrid["Shard Grid Overlay"]
        SG["Shard shape: 512 × 512 × 512<br/>Grid: ⌈768/512⌉ × ⌈10752/512⌉ × ⌈14336/512⌉<br/>= 2 × 21 × 28 = 1,176 shards<br/><br/>Each shard = 1 S3 object<br/>Max uncompressed: 512³ × 2B = 256 MB"]
    end

    subgraph ChunkGrid["Chunk Tiling Within One Shard"]
        CG["Chunk shape: 128 × 256 × 256<br/>Chunks per shard:<br/>⌈512/128⌉ × ⌈512/256⌉ × ⌈512/256⌉<br/>= 4 × 2 × 2 = 16 chunks per shard<br/><br/>Each chunk compressed independently (zstd)<br/>Max uncompressed: 128×256×256 × 2B = 16 MB"]
    end

    Volume --> ShardGrid --> ChunkGrid

    style Volume fill:#ffebee,stroke:#b71c1c
    style ShardGrid fill:#fff3e0,stroke:#e65100
    style ChunkGrid fill:#e8f5e9,stroke:#2e7d32
```

---

## Physical Layout of a Single Shard File

Each shard on S3 is a self-contained binary file with the following structure:

```mermaid
flowchart LR
    subgraph ShardFile["S3 Object: c/0/3/5 (shard at grid position z=0, y=3, x=5)"]
        direction TB
        
        subgraph Chunks["Compressed Chunk Data (variable length)"]
            C0["Chunk [0,0,0]<br/>zstd compressed<br/>~4-8 MB"]
            C1["Chunk [0,0,1]<br/>zstd compressed"]
            C2["Chunk [0,1,0]<br/>zstd compressed"]
            C3["Chunk [0,1,1]<br/>zstd compressed"]
            C4["Chunk [1,0,0]"]
            C5["Chunk [1,0,1]"]
            C6["Chunk [1,1,0]"]
            C7["Chunk [1,1,1]"]
            C8["Chunk [2,0,0]"]
            C15["… Chunk [3,1,1]"]
        end
        
        subgraph Index["Shard Index (fixed, at end)"]
            IDX["16 entries × (offset, length)<br/>little-endian uint64<br/>+ CRC32C checksum<br/>= 260 bytes"]
        end
    end

    style Chunks fill:#e3f2fd,stroke:#1565c0
    style Index fill:#fff8e1,stroke:#f57f17
```

### Codec Chain (per chunk):

```mermaid
flowchart LR
    Raw["Raw uint16 data<br/>(128 × 256 × 256)<br/>= 16 MB"] 
    --> Transpose["transpose(order='C')<br/>Ensure C-contiguous layout"]
    --> Zstd["zstd(level=3)<br/>Compress<br/>~3-5× ratio"]
    --> Stored["Stored in shard<br/>~3-5 MB typical"]

    style Raw fill:#ffebee,stroke:#b71c1c
    style Transpose fill:#f3e5f5,stroke:#6a1b9a
    style Zstd fill:#e3f2fd,stroke:#1565c0
    style Stored fill:#e8f5e9,stroke:#2e7d32
```

---

## Byte-Size Math

### Per Chunk (compression unit)

| Metric | Value |
|--------|-------|
| Chunk shape | 128 × 256 × 256 |
| Voxels per chunk | 8,388,608 |
| dtype | uint16 (2 bytes) |
| **Uncompressed size** | **16 MB** |
| Typical zstd ratio | 3–5× for microscopy data |
| **Compressed size** | **~3–5 MB** |

### Per Shard (file granularity)

| Metric | Value |
|--------|-------|
| Shard shape | 512 × 512 × 512 |
| Voxels per shard | 134,217,728 |
| Chunks per shard | 4 × 2 × 2 = 16 |
| **Uncompressed size** | **256 MB** |
| **Compressed size** | **~50–85 MB** |
| S3 PutObject | Single request per shard |

### Per Tile (full image)

| Metric | Value |
|--------|-------|
| Tile shape | 768 × 10,752 × 14,336 |
| Total voxels | ~118 billion |
| Shards per tile | 2 × 21 × 28 = 1,176 |
| **Uncompressed size** | **~301 GB** (base level only) |
| **Compressed size** | **~60–100 GB** |

### Per Dataset (all tiles + pyramids)

| Metric | Value |
|--------|-------|
| Tiles | 20 |
| Total base-level shards | 20 × 1,176 = 23,520 |
| Pyramid overhead | ~14% (geometric series 1/8 + 1/64 + …) |
| **Estimated output size** | **~1.3–2.3 TB** compressed |

---

## Superchunk Read Pattern (I/O Optimization)

The **superchunk** is a read-side concept in `imaris_to_zarr_parallel()` that amortizes HDF5 I/O latency by reading a large contiguous region, then slicing it into shard-sized blocks:

```mermaid
flowchart TD
    subgraph HDF5["Imaris HDF5 (native chunks: 32 × 128 × 128)"]
        DataSet["ResolutionLevel 0 / TimePoint 0 / Channel 0 / Data"]
    end

    subgraph ReadPattern["Superchunk Read (1024 × 1024 × 1024)"]
        SC["reader.iter_superchunks(<br/>  superchunk_shape=(1024, 1024, 1024),<br/>  yield_shape=(512, 512, 512)<br/>)<br/><br/>One HDF5 read: 1024³ × 2B = 2 GB<br/>Then yields 8 shard-sized blocks<br/>without additional I/O"]
    end

    subgraph Yield["Yielded Shard-Sized Blocks"]
        B0["Block (0:512, 0:512, 0:512)"]
        B1["Block (0:512, 0:512, 512:1024)"]
        B2["Block (0:512, 512:1024, 0:512)"]
        B3["Block (0:512, 512:1024, 512:1024)"]
        B4["Block (512:1024, 0:512, 0:512)"]
        B5["Block (512:1024, 0:512, 512:1024)"]
        B6["Block (512:1024, 512:1024, 0:512)"]
        B7["Block (512:1024, 512:1024, 512:1024)"]
    end

    HDF5 -->|"Single contiguous read<br/>(amortizes HDF5 chunk decompression)"| ReadPattern
    ReadPattern -->|"numpy slicing<br/>(zero-copy view)"| Yield

    style HDF5 fill:#fff3e0,stroke:#e65100
    style ReadPattern fill:#e8eaf6,stroke:#283593
    style Yield fill:#e8f5e9,stroke:#2e7d32
```

### Superchunk vs Direct Shard Read

| Approach | I/O Operations | Memory | Use Case |
|----------|---------------|--------|----------|
| **Direct shard read** (`process_single_shard`) | 1 HDF5 read per shard | 256 MB | Distributed mode (default) |
| **Superchunk batch** (`iter_superchunks`) | 1 HDF5 read per 8 shards | 2 GB | Single-process parallel mode |

In **distributed mode** (shard-per-worker), direct shard reads are used because each worker only processes its own shards — the superchunk optimization only helps the single-process `imaris_to_zarr_parallel()` writer.

---

## Pyramid Level Scaling

Each pyramid level halves each spatial dimension, reducing the shard grid by 8× per level:

```mermaid
flowchart LR
    subgraph P0["Level 0 (Base)"]
        L0["Shape: 768 × 10752 × 14336<br/>Shards: 2 × 21 × 28 = 1,176<br/>Size: ~301 GB raw"]
    end
    
    subgraph P1["Level 1"]
        L1["Shape: 384 × 5376 × 7168<br/>Shards: 1 × 11 × 14 = 154<br/>Size: ~37.6 GB raw"]
    end
    
    subgraph P2["Level 2"]
        L2["Shape: 192 × 2688 × 3584<br/>Shards: 1 × 6 × 7 = 42<br/>Size: ~4.7 GB raw"]
    end
    
    subgraph P3["Level 3"]
        L3["Shape: 96 × 1344 × 1792<br/>Shards: 1 × 3 × 4 = 12<br/>Size: ~587 MB raw"]
    end
    
    subgraph P4["Level 4"]
        L4["Shape: 48 × 672 × 896<br/>Shards: 1 × 2 × 2 = 4<br/>Size: ~73 MB raw"]
    end

    P0 -->|"÷2 per axis"| P1 -->|"÷2 per axis"| P2 -->|"÷2 per axis"| P3 -->|"÷2 per axis"| P4

    style P0 fill:#ffcdd2,stroke:#b71c1c
    style P1 fill:#f8bbd0,stroke:#880e4f
    style P2 fill:#e1bee7,stroke:#4a148c
    style P3 fill:#d1c4e9,stroke:#311b92
    style P4 fill:#c5cae9,stroke:#1a237e
```

### Shard Clamping at Small Levels

When a pyramid level's dimension is smaller than the shard shape, `create_scale_spec()` clamps shards:

```python
# From create_scale_spec() line ~200
clamped_shard = []
for s, c, d in zip(shard_shape, clamped_chunk, data_shape):
    clamped = min(s, d)                    # Don't exceed data dimension
    clamped = (clamped // c) * c           # Round to chunk multiple
    if clamped < c: clamped = c            # At least one chunk
    clamped_shard.append(clamped)
```

This ensures **shard shape is always a multiple of chunk shape** (Zarr v3 sharding requirement), even for small pyramid levels.

---

## TensorStore Zarr v3 Spec Structure

The spec built by `create_scale_spec()` defines the complete storage schema:

```mermaid
flowchart TD
    subgraph Spec["TensorStore Spec (create_scale_spec output)"]
        Driver["driver: 'zarr3'"]
        
        subgraph KVStore["kvstore"]
            KV_Driver["driver: 's3'"]
            Bucket["bucket: 'aind-open-data'"]
            Path["path: 'dataset/SPIM/tile_000.ome.zarr/0'"]
            Region["aws_region: 'us-west-2'"]
            Context["context:<br/>  cache_pool: 1 GB<br/>  data_copy_concurrency: N cpus<br/>  s3_request_concurrency: N cpus"]
        end
        
        subgraph Metadata["metadata"]
            Shape["shape: [1, 1, 768, 10752, 14336]"]
            ChunkGrid["chunk_grid:<br/>  name: 'regular'<br/>  chunk_shape: [1, 1, 512, 512, 512]<br/>  (= shard shape, file boundary)"]
            DType["data_type: 'uint16'"]
            
            subgraph Codecs["codecs"]
                subgraph Sharding["sharding_indexed"]
                    InnerChunk["chunk_shape: [1, 1, 128, 256, 256]"]
                    subgraph InnerCodecs["codecs (per chunk)"]
                        Transpose["transpose: order 'C'"]
                        Zstd["zstd: level 3"]
                    end
                    subgraph IndexCodecs["index_codecs"]
                        Bytes["bytes: little-endian"]
                        CRC["crc32c"]
                    end
                    IndexLoc["index_location: 'end'"]
                end
            end
        end
        
        Flags["create: true<br/>open: true<br/>delete_existing: false"]
    end

    style Spec fill:#fafafa,stroke:#424242
    style KVStore fill:#e3f2fd,stroke:#1565c0
    style Metadata fill:#f3e5f5,stroke:#6a1b9a
    style Codecs fill:#fff8e1,stroke:#f57f17
    style Sharding fill:#fff3e0,stroke:#e65100
    style InnerCodecs fill:#e8f5e9,stroke:#2e7d32
    style IndexCodecs fill:#fce4ec,stroke:#b71c1c
```

### Important Zarr v3 Semantics

| Zarr v3 Concept | Mapped To | Meaning |
|-----------------|-----------|---------|
| `chunk_grid.chunk_shape` | Shard shape `[1,1,512,512,512]` | Defines the **file boundary** — each grid cell = 1 stored file/object |
| `sharding_indexed.chunk_shape` | Inner chunk `[1,1,128,256,256]` | The **compression unit** within each shard |
| `codecs` (outer) | `sharding_indexed` only | The shard codec wraps everything |
| `codecs` (inner) | `transpose` + `zstd` | Applied per inner chunk |

> **Critical distinction:** In Zarr v3 with sharding, the "chunk" in `chunk_grid` is actually the **shard** (storage boundary), while the "chunk" inside `sharding_indexed` is the true compression chunk. TensorStore's API reflects this.

---

## Read Path (Client Perspective)

When a viewer (e.g., Neuroglancer) reads a region:

```mermaid
sequenceDiagram
    participant Client as Neuroglancer
    participant S3 as S3 Bucket
    participant Shard as Shard File

    Client->>Client: Determine which shard(s) contain the region<br/>shard_idx = floor(coord / shard_shape)
    Client->>S3: GET c/{z}/{y}/{x} (Range: last 260 bytes)
    S3-->>Client: Shard index trailer
    Client->>Client: Parse index → find chunk offset/length
    Client->>S3: GET c/{z}/{y}/{x} (Range: offset..offset+length)
    S3-->>Client: Compressed chunk bytes
    Client->>Client: zstd decompress → 128×256×256 uint16
    Client->>Client: Render voxels
```

The **shard index at the end** enables random-access reads of individual chunks within a shard using HTTP Range requests — only the needed chunk is fetched and decompressed.

---

## Key Code References

| Component | File | Line |
|-----------|------|------|
| `create_scale_spec()` | `compress/imaris_to_zarr.py` | L134 |
| `_build_kvstore_spec()` | `compress/imaris_to_zarr.py` | L92 |
| `enumerate_shard_indices()` | `compress/imaris_to_zarr.py` | L366 |
| `compute_shard_grid()` | `compress/imaris_to_zarr.py` | L283 |
| `shard_index_to_slices()` | `compress/imaris_to_zarr.py` | L310 |
| `iter_superchunks()` | `utils/io_utils.py` | ImarisReader method |
| `iter_block_aligned_slices()` | `compress/imaris_to_zarr.py` | L1130 |
| Shard clamping logic | `compress/imaris_to_zarr.py` | L195–210 |
| Chunk/shard defaults | `models.py` | L63–72 |
