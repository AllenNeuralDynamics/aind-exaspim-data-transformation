# aind-exaspim-data-transformation

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
![Code Style](https://img.shields.io/badge/code%20style-black-black)
[![semantic-release: angular](https://img.shields.io/badge/semantic--release-angular-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)
![Interrogate](https://img.shields.io/badge/interrogate-90.8%25-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-83%25-yellow)
![Python](https://img.shields.io/badge/python->=3.12-blue?logo=python)

## Usage
 - To use this template, click the green `Use this template` button and `Create new repository`.
 - After github initially creates the new repository, please wait an extra minute for the initialization scripts to finish organizing the repo.
 - To enable the automatic semantic version increments: in the repository go to `Settings` and `Collaborators and teams`. Click the green `Add people` button. Add `svc-aindscicomp` as an admin. Modify the file in `.github/workflows/tag_and_publish.yml` and remove the if statement in line 65. The semantic version will now be incremented every time a code is committed into the main branch.
 - To publish to PyPI, enable semantic versioning and uncomment the publish block in `.github/workflows/tag_and_publish.yml`. The code will now be published to PyPI every time the code is committed into the main branch.
 - The `.github/workflows/test_and_lint.yml` file will run automated tests and style checks every time a Pull Request is opened. If the checks are undesired, the `test_and_lint.yml` can be deleted. The strictness of the code coverage level, etc., can be modified by altering the configurations in the `pyproject.toml` file and the `.flake8` file.
 - Please make any necessary updates to the README.md and CITATION.cff files

## Level of Support
Please indicate a level of support:
 - [ ] Supported: We are releasing this code to the public as a tool we expect others to use. Issues are welcomed, and we expect to address them promptly; pull requests will be vetted by our staff before inclusion.
 - [ ] Occasional updates: We are planning on occasional updating this tool with no fixed schedule. Community involvement is encouraged through both issues and pull requests.
 - [ ] Unsupported: We are not currently supporting this code, but simply releasing it to the community AS IS but are not able to provide any guarantees of support. The community is welcome to submit issues, but you should not expect an active response.

## Release Status
GitHub's tags and Release features can be used to indicate a Release status.

 - Stable: v1.0.0 and above. Ready for production.
 - Beta:  v0.x.x or indicated in the tag. Ready for beta testers and early adopters.
 - Alpha: v0.x.x or indicated in the tag. Still in early development.

## Installation
To use the software, in the root directory, run
```bash
pip install -e .
```

To develop the code, run
```bash
pip install -e . --group dev
```
Note: --group flag is available only in pip versions >=25.1

Alternatively, if using `uv`, run
```bash
uv sync
```

## Features

### Single Tile Upload Mode

For integration testing and performance validation of very large datasets, the package supports a `single_tile_upload` mode. When enabled, only the first tile/file from the dataset is processed, while still maintaining full horizontal scaling across workers.

**Use Case**: Testing upload performance on representative data (e.g., 6TB tile) without processing the entire dataset (e.g., 120TB).

**Configuration**:
```json
{
    "input_source": "s3://bucket/dataset",
    "output_directory": "/scratch/output",
    "s3_location": "s3://bucket/zarr-output",
    "num_of_partitions": 16,
    "partition_to_process": 0,
    "partition_mode": "shard",
    "single_tile_upload": true
}
```

**Key Points**:
- The first tile is selected after deterministic sorting (all workers select the same file)
- The selected tile is still distributed across workers using the configured partitioning strategy
- Works with both `shard` and `file` partition modes
- Default value is `false` for backward compatibility

**Benefits**:
- ~95% cost reduction for integration tests
- Faster iteration cycles for debugging and optimization
- Realistic performance testing with representative data volume
- Same code paths as production (just with restricted input scope)

For more details, see [Single Tile Upload Design Document](docs/SINGLE_TILE_UPLOAD_DESIGN.md).
