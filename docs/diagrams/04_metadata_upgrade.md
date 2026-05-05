# Metadata Upgrade & Service Calls

## Overview

**Worker 0** exclusively handles metadata operations before proceeding to compression. This includes:

1. **Fetching** `subject.json` and `procedures.json` from `aind-metadata-service`
2. **Upgrading** `acquisition.json` and `instrument.json` from schema v1 → v2.5+
3. **Backing up** originals to S3 under `derived/`
4. **Uploading** upgraded files to the S3 dataset root

All metadata operations are **non-blocking** — failures are logged but do not abort compression.

---

## Sequence Diagram: Complete Metadata Flow

```mermaid
sequenceDiagram
    participant Job as ImarisCompressionJob<br/>(Worker 0)
    participant FS as Local Filesystem<br/>(NFS)
    participant MS as aind-metadata-service<br/>(http://aind-metadata-service)
    participant S3 as S3 Bucket<br/>(aind-open-data)
    participant Upgrader as aind-metadata-upgrader<br/>(Upgrade class)

    Note over Job: Only runs when partition_to_process == 0

    %% Phase 1: Fetch additional metadata
    rect rgb(232, 245, 233)
        Note over Job, S3: Phase 1: _get_additional_metadata()
        
        Job->>Job: _derive_subject_id(source_dir)<br/>e.g. "exaSPIM_765830_2026-01-15" → "765830"
        
        Job->>FS: Check if subject.json exists locally
        alt File exists locally
            FS-->>Job: Found — skip
        else Not local
            Job->>S3: head_object(subject.json) — already in S3?
            alt Already in S3 (placed by gather_preliminary_metadata)
                S3-->>Job: 200 OK — skip
            else Not in S3
                Job->>MS: GET /api/v2/subject/765830
                MS-->>Job: 200 OK + JSON body
                Job->>FS: Write subject.json locally
                Job->>S3: PutObject subject.json
            end
        end
        
        Job->>FS: Check if procedures.json exists locally
        alt File exists locally
            FS-->>Job: Found — skip
        else Not local
            Job->>S3: head_object(procedures.json)
            alt Already in S3
                S3-->>Job: 200 OK — skip
            else Not in S3
                Job->>MS: GET /api/v2/procedures/765830
                MS-->>Job: 200/400 + JSON body
                Job->>FS: Write procedures.json locally
                Job->>S3: PutObject procedures.json
            end
        end
    end

    %% Phase 2: Upgrade metadata
    rect rgb(227, 242, 253)
        Note over Job, S3: Phase 2: _upgrade_metadata()
        
        Job->>FS: Load acquisition.json from Path(source_dir).parent
        Job->>Job: Check schema_version < 2.0.0?
        
        alt Already v2+ (no upgrade needed)
            Job->>Job: Log "skipping upgrade" — return
        else Needs upgrade (v1.x)
            Job->>FS: Load instrument.json (optional)
            
            alt instrument.json exists
                Job->>Job: _sanitize_instrument_manufacturers(inst_data)
                Job->>Upgrader: Upgrade().upgrade_instrument(inst_data)
                Upgrader-->>Job: upgraded_instrument (v2.5)
                Job->>Upgrader: Upgrade().upgrade_acquisition(acq_data, inst_data)
                Upgrader-->>Job: upgraded_acquisition (v2.5)
            else No instrument.json
                Job->>Upgrader: AcquisitionV1V2(acq_data, stub_metadata)
                Upgrader-->>Job: upgraded_acquisition (v2.5)
            end
            
            %% Backup originals
            Job->>S3: PutObject derived/v1_acquisition.json
            Job->>S3: PutObject derived/v1_instrument.json
            
            %% Upload upgraded
            Job->>S3: PutObject acquisition.json (v2.5)
            Job->>S3: PutObject instrument.json (v2.5)
        end
    end

    %% Phase 3: Derivatives
    rect rgb(255, 243, 224)
        Note over Job, S3: Phase 3: _upload_derivatives_folder()
        Job->>FS: Check if derivatives/ exists
        alt Exists
            Job->>S3: aws s3 sync derivatives/ → s3://…/SPIM/derivatives/
        else Does not exist
            Job->>Job: Skip
        end
    end

    Note over Job: Metadata complete — proceed to compression
```

---

## `_sanitize_instrument_manufacturers()` — Pre-Processing

Before the upgrader runs, instrument data is sanitized to prevent crashes:

```mermaid
flowchart TD
    Input["instrument.json (v1) loaded as dict"]
    
    subgraph Sanitize["_sanitize_instrument_manufacturers()"]
        direction TB
        
        S1["1. Walk all device lists:<br/>objectives, detectors, light_sources,<br/>fluorescence_filters, lenses, scanning_stages,<br/>motorized_stages, additional_devices, daqs, optical_tables"]
        
        S2["2. For each device.manufacturer.name:<br/>if Organization.from_name(name) is None:<br/>  → Replace with 'Other'<br/>  → Preserve original in device.notes"]
        
        S3["3. Back-fill missing magnification on objectives:<br/>if magnification is None → default 1.0"]
        
        S4["4. Parse center_wavelength from filter model strings:<br/>e.g. 'ZET405/488/561/620m' → [405, 488, 561, 620]<br/>Only for Multiband filters with center_wavelength=None"]
        
        S5["5. Strip deprecated fields from motorized stages:<br/>Remove: stage_axis_direction, stage_axis_name<br/>(only valid on ScanningStage in v2)"]
        
        S6["6. Fix invalid travel_unit on motorized stages:<br/>'degree' → 'millimeter'<br/>(v2 only accepts linear units)"]
    end
    
    Output["Sanitized instrument dict<br/>(safe for aind-metadata-upgrader)"]
    
    Input --> S1 --> S2 --> S3 --> S4 --> S5 --> S6 --> Output

    style Sanitize fill:#fff8e1,stroke:#f57f17
```

---

## Upgrade Decision Flow

```mermaid
flowchart TD
    Start["upgrade_metadata(source_dir, s3_location)"]
    
    Start --> LoadAcq["Load acquisition.json<br/>from Path(source_dir).parent"]
    
    LoadAcq --> CheckAcq{acquisition.json<br/>exists?}
    CheckAcq -->|No| Error["Raise FileNotFoundError<br/>(acquisition.json is required)"]
    CheckAcq -->|Yes| CheckVersion{"schema_version<br/>< 2.0.0?"}
    
    CheckVersion -->|"No (already v2+)"| Skip["Log: 'skipping upgrade'<br/>Return early"]
    CheckVersion -->|"Yes (v1.x)"| LoadInst["Load instrument.json"]
    
    LoadInst --> CheckInst{instrument.json<br/>exists?}
    
    CheckInst -->|Yes| WithInst["_upgrade_with_instrument()"]
    CheckInst -->|No| WithoutInst["_upgrade_acquisition_only()"]
    
    subgraph WithInstrument["Full Upgrade Path"]
        WithInst --> Sanitize["_sanitize_instrument_manufacturers()"]
        Sanitize --> UpgradeInst["Upgrade().upgrade_instrument(inst_data)"]
        UpgradeInst --> UpgradeAcq["Upgrade().upgrade_acquisition(acq_data, inst_data)"]
    end
    
    subgraph WithoutInstrument["Minimal Upgrade Path"]
        WithoutInst --> Stub["Create stub metadata:<br/>{fluorescence_filters: [],<br/> light_sources: []}"]
        Stub --> UpgradeAcqOnly["AcquisitionV1V2(acq_data, stub)"]
    end
    
    UpgradeAcq --> Backup
    UpgradeAcqOnly --> Backup
    
    Backup["Backup originals to S3:<br/>derived/v1_acquisition.json<br/>derived/v1_instrument.json"]
    Backup --> Upload["Upload upgraded files:<br/>acquisition.json (v2.5)<br/>instrument.json (v2.5)"]

    style WithInstrument fill:#e8f5e9,stroke:#2e7d32
    style WithoutInstrument fill:#fff3e0,stroke:#e65100
```

---

## Subject ID Derivation

The metadata service requires a subject ID (LabTracks number). This is parsed from the dataset folder name:

```mermaid
flowchart LR
    Path["source_dir:<br/>/allen/aind/stage/exaSPIM/<br/>exaSPIM_765830_2026-01-15_12-00-00/<br/>exaSPIM/"]
    
    Path --> Parent["Path(source_dir).parent.name<br/>= 'exaSPIM_765830_2026-01-15_12-00-00'"]
    
    Parent --> Split["Split by '_' → ['exaSPIM', '765830', '2026-01-15', '12-00-00']"]
    
    Split --> Extract["subject_id = '765830'<br/>labtracks_id = '765830'.split('-')[0] = '765830'"]
    
    Extract --> URL["Metadata Service URL:<br/>GET /api/v2/subject/765830<br/>GET /api/v2/procedures/765830"]

    style Path fill:#f3e5f5,stroke:#6a1b9a
    style URL fill:#e3f2fd,stroke:#1565c0
```

---

## Error Handling Strategy

All metadata operations use a **fail-soft** pattern — errors are logged but never abort compression:

```mermaid
flowchart TD
    subgraph ErrorHandling["Non-Blocking Error Handling"]
        Op["Metadata operation<br/>(fetch/upgrade/upload)"]
        
        Op --> Try{"try:"}
        
        Try -->|Success| Log["Log success<br/>Continue"]
        Try -->|"Exception<br/>(OSError, ValueError,<br/>RuntimeError, ClientError,<br/>BotoCoreError)"| Catch["logging.error(<br/>'METADATA FETCH/UPGRADE FAILED<br/>— continuing with compression')"]
        
        Catch --> Continue["Continue to compression<br/>(run_job proceeds normally)"]
    end

    style ErrorHandling fill:#fce4ec,stroke:#b71c1c
```

**Rationale:** The primary value of this pipeline is the ~120 TB compression job. Metadata can always be manually corrected later — but re-running a multi-hour compression job due to a metadata service timeout would waste significant compute resources.

---

## S3 Path Construction

The pipeline receives `s3_location` with a **modality subfolder** (e.g., `s3://bucket/dataset/SPIM`). Metadata belongs at the **dataset root** (one level up):

```mermaid
flowchart TD
    Input["s3_location from job_settings:<br/>s3://aind-open-data/exaSPIM_765830_2026-01-15_12-00-00/SPIM"]
    
    Input --> Strip["_get_dataset_root_s3():<br/>urlparse → strip last path segment"]
    
    Strip --> Root["s3_dataset_root:<br/>s3://aind-open-data/exaSPIM_765830_2026-01-15_12-00-00"]
    
    Root --> MetaFiles["Metadata uploaded to root:<br/>s3://…/exaSPIM_765830_…/acquisition.json<br/>s3://…/exaSPIM_765830_…/instrument.json<br/>s3://…/exaSPIM_765830_…/subject.json<br/>s3://…/exaSPIM_765830_…/procedures.json"]
    
    Root --> Backups["Backups under derived/:<br/>s3://…/exaSPIM_765830_…/derived/v1_acquisition.json<br/>s3://…/exaSPIM_765830_…/derived/v1_instrument.json"]

    style Input fill:#fff3e0,stroke:#e65100
    style Root fill:#e8f5e9,stroke:#2e7d32
```

---

## Integration with Airflow Pipeline

```mermaid
flowchart LR
    subgraph AirflowDAG["aind-data-transfer-service DAG"]
        Step1["gather_preliminary_metadata<br/>(may place subject.json,<br/>procedures.json in S3)"]
        Step2["compress_data<br/>(this package)"]
        Step3["quality_control"]
    end
    
    Step1 -->|"Files may already<br/>exist in S3"| Step2
    Step2 -->|"Zarr + metadata<br/>in S3"| Step3

    style AirflowDAG fill:#e3f2fd,stroke:#1565c0
```

The `_get_additional_metadata()` function checks whether files were already placed by the upstream `gather_preliminary_metadata` step using `_s3_object_exists()` — this avoids redundant metadata service calls and prevents overwriting correct data.

---

## Key Code References

| Component | File | Line |
|-----------|------|------|
| `_get_additional_metadata()` dispatch | `imaris_job.py` | L565 |
| `_upgrade_metadata()` dispatch | `imaris_job.py` | L600 |
| `_get_dataset_root_s3()` | `imaris_job.py` | L530 |
| `get_additional_metadata()` | `upgrade_metadata.py` | L509 |
| `upgrade_metadata()` | `upgrade_metadata.py` | L617 |
| `_sanitize_instrument_manufacturers()` | `upgrade_metadata.py` | L152 |
| `_derive_subject_id()` | `upgrade_metadata.py` | (internal) |
| `_needs_upgrade()` | `upgrade_metadata.py` | L60 |
| `_backup_original_to_s3()` | `upgrade_metadata.py` | L115 |
| `_upload_upgraded_to_s3()` | `upgrade_metadata.py` | L122 |
| `_s3_object_exists()` | `upgrade_metadata.py` | L96 |
| Error handling pattern | `imaris_job.py` | L580, L625 |
