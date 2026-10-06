# Worker Partitioning & Shard Distribution

## Overview

The ExaSPIM data transformation pipeline uses **shard-level partitioning** (default) to distribute work across up to 64 SLURM array workers. Each worker independently reads from shared Imaris `.ims` files and writes to disjoint shard files in the output Zarr v3 store — **no inter-worker coordination** is required during data processing.

---

## Orchestration: Airflow → SLURM Array → Workers

```mermaid
flowchart TB
    subgraph Airflow["aind-data-transfer-service (Airflow)"]
        DAG["compress_data DAG"]
    end

    DAG -->|"POST /submit_job<br/>image_resources.array = 0-63"| SLURM

    subgraph SLURM["SLURM Array Job (64 pods)"]
        W0["Worker 0<br/>partition_to_process=0"]
        W1["Worker 1<br/>partition_to_process=1"]
        W2["Worker 2<br/>partition_to_process=2"]
        WN["Worker 63<br/>partition_to_process=63"]
    end

    subgraph Container["Each Container (ghcr.io/…/aind-exaspim-data-transformation)"]
        Entry["job_entrypoint()<br/>multiprocessing.set_start_method('spawn')"]
        Parse["Parse --job-settings JSON<br/>→ ImarisJobSettings"]
        Job["ImarisCompressionJob.run_job()"]
    end

    W0 & W1 & W2 & WN -->|"python -m …imaris_job<br/>--job-settings '{…}'"| Entry
    Entry --> Parse --> Job

    style Airflow fill:#e3f2fd,stroke:#1976d2
    style SLURM fill:#fff3e0,stroke:#f57c00
    style Container fill:#f1f8e9,stroke:#558b2f
```

---

## Shard-Level Partitioning Strategy

The **default** `partition_mode="shard"` distributes work at sub-file granularity. This is critical because a single tile can be 6 TB — with only 2–20 tiles, file-level distribution would leave most workers idle.

```mermaid
flowchart TD
    A["_get_sorted_stack_paths()<br/>Find & sort all .ims files<br/>e.g. tile_000.ims, tile_001.ims, …, tile_019.ims"]
    
    A --> B["_build_global_shard_task_list()"]
    
    subgraph BuildTasks["Build Global Task List"]
        B --> C["For each .ims file:<br/>ImarisReader.get_metadata_shape() → (Z, Y, X)"]
        C --> D["enumerate_shard_indices(shape, shard_shape)<br/>→ list of (z_idx, y_idx, x_idx) tuples"]
        D --> E["Append (file_path, shard_index) to global list"]
    end
    
    E --> F["Global flat list of ALL shard tasks<br/>e.g. 20 tiles × ~462 shards/tile = 9,240 tasks"]
    
    F --> G["partition_list(tasks, num_of_partitions=64)<br/>Round-robin: task[i] → worker[i % 64]"]
    
    G --> H["Worker 0: tasks 0, 64, 128, …<br/>Worker 1: tasks 1, 65, 129, …<br/>…<br/>Worker 63: tasks 63, 127, 191, …"]
    
    H --> I["Group by file for efficient processing:<br/>tasks_by_file[Path] = [shard_idx, …]"]
    
    I --> J["_process_file_shards() per file group"]

    style BuildTasks fill:#fce4ec,stroke:#c62828
```

### Distribution Math (Typical Dataset)

| Parameter | Value |
|-----------|-------|
| Tiles | 20 × `.ims` files |
| Tile shape | `~768 × 10752 × 14336` voxels |
| Shard shape | `512 × 512 × 512` |
| Shards per tile | `⌈768/512⌉ × ⌈10752/512⌉ × ⌈14336/512⌉ = 2 × 21 × 28 = 1,176` |
| **Total shards (base level)** | **20 × 1,176 = 23,520** |
| Workers | 64 |
| **Shards per worker** | **~368** (round-robin balanced) |
| Dask workers per container | 4 |
| **Effective parallelism** | **64 × 4 = 256 concurrent shard writes** |

---

## Worker Internal Architecture: Dask LocalCluster

Each SLURM worker creates a **local Dask cluster** to further parallelize shard processing within its container:

```mermaid
flowchart TB
    subgraph Worker["SLURM Worker N (partition_to_process=N)"]
        direction TB
        
        Main["_run_shard_partitioned()<br/>my_tasks = partitioned[N]<br/>(~368 shard tasks)"]
        
        Main --> Group["Group tasks by file:<br/>tasks_by_file = {<br/>  tile_003.ims: [(0,0,0), (0,0,1), …],<br/>  tile_007.ims: [(1,2,3), (1,2,4), …]<br/>}"]
        
        Group --> Cluster["LocalCluster(<br/>  n_workers=4,<br/>  threads_per_worker=1<br/>)"]
        
        Cluster --> Process["Per-file: _process_file_shards()"]
        
        subgraph DaskCluster["Dask LocalCluster (4 workers)"]
            DW0["Dask Worker 0"]
            DW1["Dask Worker 1"]
            DW2["Dask Worker 2"]
            DW3["Dask Worker 3"]
        end
        
        Process -->|"client.submit(<br/>process_single_shard,<br/>**task)"| DW0 & DW1 & DW2 & DW3
    end

    subgraph SharedRead["Shared Read (NFS/Lustre)"]
        IMS[("tile_003.ims<br/>(HDF5/LZ4, ~6 TB)")]
    end

    subgraph SharedWrite["Independent Writes (S3)"]
        S3_0["shard c/0/0/0"]
        S3_1["shard c/0/0/1"]
        S3_2["shard c/0/1/0"]
        S3_3["shard c/0/1/1"]
    end

    DW0 -->|"read_block(slices)"| IMS
    DW1 -->|"read_block(slices)"| IMS
    DW2 -->|"read_block(slices)"| IMS
    DW3 -->|"read_block(slices)"| IMS

    DW0 -->|"ts.open().write()"| S3_0
    DW1 -->|"ts.open().write()"| S3_1
    DW2 -->|"ts.open().write()"| S3_2
    DW3 -->|"ts.open().write()"| S3_3

    style Worker fill:#f3e5f5,stroke:#7b1fa2
    style DaskCluster fill:#e8f5e9,stroke:#2e7d32
    style SharedRead fill:#fff8e1,stroke:#f9a825
    style SharedWrite fill:#e3f2fd,stroke:#1565c0
```

---

## `process_single_shard()` — The Atomic Work Unit

Each Dask worker executes this function for ONE shard at a time. It is fully self-contained:

```mermaid
sequenceDiagram
    participant DW as Dask Worker
    participant HDF5 as ImarisReader (HDF5)
    participant TS as TensorStore (S3)

    DW->>DW: shard_index_to_slices((1,3,5), (512,512,512), data_shape)<br/>→ (slice(512,768), slice(1536,2048), slice(2560,3072))
    DW->>HDF5: read_block(slices_3d, data_path)<br/>HDF5 hyperslab selection (LZ4 decompress)
    HDF5-->>DW: numpy array (256×512×512) uint16
    DW->>DW: block_data_5d = data[np.newaxis, np.newaxis, ...]<br/>→ shape (1, 1, 256, 512, 512)
    DW->>TS: ts.open(write_spec, open=True, create=False)
    TS-->>DW: TensorStore handle
    DW->>TS: store[slices_5d].write(block_data_5d).result()
    TS-->>DW: Write complete (shard file → S3)
    DW-->>DW: return {shard_index, bytes_read, elapsed_seconds, …}
```

**Key properties:**
- No lock contention — each shard maps to a unique S3 object
- Memory per worker = 1 shard: `512³ × 2 bytes = 256 MB`
- HDF5 concurrent read is safe (read-only access)
- Each shard is written as one atomic S3 PutObject

---

## Pyramid Level Distribution

After base-level (level 0) shards are written, **pyramid levels 1–4** are independently re-partitioned across the same 64 workers. Each level has fewer shards (halved per axis → 1/8th):

```mermaid
flowchart LR
    subgraph L0["Level 0 (Full Resolution)"]
        L0_shards["23,520 shards<br/>512³ each"]
    end
    
    subgraph L1["Level 1 (2× downsampled)"]
        L1_shards["2,940 shards<br/>(grid halved per axis)"]
    end
    
    subgraph L2["Level 2 (4× downsampled)"]
        L2_shards["~368 shards"]
    end
    
    subgraph L3["Level 3 (8× downsampled)"]
        L3_shards["~46 shards"]
    end
    
    subgraph L4["Level 4 (16× downsampled)"]
        L4_shards["~6 shards"]
    end

    L0 --> L1 --> L2 --> L3 --> L4

    style L0 fill:#ffcdd2,stroke:#b71c1c
    style L1 fill:#f8bbd0,stroke:#880e4f
    style L2 fill:#e1bee7,stroke:#4a148c
    style L3 fill:#d1c4e9,stroke:#311b92
    style L4 fill:#c5cae9,stroke:#1a237e
```

```mermaid
flowchart TD
    subgraph PerLevel["For each pyramid level 1..4"]
        A["enumerate_shard_indices(lvl_shape, shard_shape)<br/>→ list of level-specific shard indices"]
        B["_partition_list(lvl_indices, 64)<br/>Round-robin across workers"]
        C["my_lvl_shards = partitioned[partition_to_process]"]
        D["create_shard_tasks() with data_path=<br/>/DataSet/ResolutionLevel {lvl}/TimePoint 0/Channel 0/Data"]
        E["Submit to Dask OR sequential execution"]
    end
    
    A --> B --> C --> D --> E

    style PerLevel fill:#e8eaf6,stroke:#283593
```

> **Note:** Pyramid shards are partitioned **independently** from base shards. Worker 5's level-2 shards need not correspond spatially to its level-0 shards. This maximizes load balance since higher levels have far fewer shards.

---

## File-Level Mode (Legacy, `partition_mode="file"`)

For comparison, the legacy mode distributes **entire files** round-robin:

```mermaid
flowchart LR
    subgraph Files["20 .ims tiles (sorted)"]
        F0["tile_000.ims"]
        F1["tile_001.ims"]
        F2["tile_002.ims"]
        F19["tile_019.ims"]
    end
    
    subgraph Workers["64 SLURM Workers"]
        W0["Worker 0 → tile_000"]
        W1["Worker 1 → tile_001"]
        W19["Worker 19 → tile_019"]
        W20["Worker 20 → (idle)"]
        W63["Worker 63 → (idle)"]
    end

    F0 --> W0
    F1 --> W1
    F19 --> W19

    style Workers fill:#ffebee,stroke:#c62828
```

**Problem:** With 20 tiles and 64 workers, **44 workers sit idle**. Shard-level mode eliminates this waste completely.

---

## Idempotent Store Creation (Race-Free)

All workers idempotently create/open Zarr stores before submitting shard tasks:

```mermaid
sequenceDiagram
    participant W0 as Worker 0
    participant W1 as Worker 1
    participant WN as Worker 63
    participant S3 as S3 (zarr.json)

    Note over W0, WN: Step 2: All workers create stores (idempotent)
    
    par All workers execute simultaneously
        W0->>S3: ts.open(spec, create=True, open=True, delete_existing=False)
        W1->>S3: ts.open(spec, create=True, open=True, delete_existing=False)
        WN->>S3: ts.open(spec, create=True, open=True, delete_existing=False)
    end
    
    S3-->>W0: Store handle (created or opened existing)
    S3-->>W1: Store handle (opened existing)
    S3-->>WN: Store handle (opened existing)

    Note over W0, WN: Step 3: Workers write DISJOINT shards (no conflicts)
    
    W0->>S3: Write shard c/0/0/0, c/0/0/4, c/0/1/0, …
    W1->>S3: Write shard c/0/0/1, c/0/0/5, c/0/1/1, …
    WN->>S3: Write shard c/0/0/63, c/0/2/7, …

    Note over W0: Step 5: Only Worker 0 (or Dask coordinator) writes metadata
    W0->>S3: Write zarr.json (OME-NGFF 0.5 multiscales)
```

---

## Key Code References

| Component | File | Line |
|-----------|------|------|
| `partition_list()` round-robin | `imaris_job.py` | L44 |
| `_build_global_shard_task_list()` | `imaris_job.py` | L683 |
| `_run_shard_partitioned()` | `imaris_job.py` | L723 |
| `_process_file_shards()` | `imaris_job.py` | L790 |
| `enumerate_shard_indices()` | `compress/imaris_to_zarr.py` | L366 |
| `process_single_shard()` | `compress/imaris_to_zarr.py` | L383 |
| `create_shard_tasks()` | `compress/imaris_to_zarr.py` | L466 |
| Pyramid level re-partition | `compress/imaris_to_zarr.py` | L1900 |
| Worker 0 metadata duties | `imaris_job.py` | L865–L870 |
