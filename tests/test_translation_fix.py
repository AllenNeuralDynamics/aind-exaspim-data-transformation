"""
Test script to verify that translation fields are now included in zarr.json
"""

import json

from aind_exaspim_data_transformation.compress.omezarr_metadata import (
    write_ome_ngff_metadata,
)

# Test parameters similar to what the conversion functions use
arr_shape = [1, 1, 768, 2688, 3584]
chunk_size = [1, 1, 128, 128, 128]
image_name = "test_image"
n_lvls = 5
scale_factors = (1.0, 2.0, 2.0)  # Z, Y, X
voxel_size = (1.0, 1.0, 1.0)  # Z, Y, X in micrometers
channel_names = ["test_channel"]
origin = [0.0, 0.0, 0.0]  # Z, Y, X origin

print("Testing metadata generation WITH origin parameter...")
metadata_with_origin = write_ome_ngff_metadata(
    arr_shape=arr_shape,
    chunk_size=chunk_size,
    image_name=image_name,
    n_lvls=n_lvls,
    scale_factors=scale_factors,
    voxel_size=voxel_size,
    channel_names=channel_names,
    origin=origin,
)

print("\nGenerated metadata:")
print(json.dumps(metadata_with_origin, indent=2))

# Check if translations are present
multiscales = metadata_with_origin["attributes"]["ome"]["multiscales"][0]
datasets = multiscales["datasets"]

print("\n" + "=" * 80)
print("Checking translation fields in each pyramid level:")
print("=" * 80)

all_have_translations = True
for i, dataset in enumerate(datasets):
    transforms = dataset.get("coordinateTransformations", [])
    has_translation = False
    translation_value = None

    for transform in transforms:
        if transform.get("type") == "translation":
            has_translation = True
            translation_value = transform.get("translation")
            break

    if has_translation:
        print(f"✓ Level {i}: HAS translation field: {translation_value}")
    else:
        print(f"✗ Level {i}: MISSING translation field")
        all_have_translations = False

print("\n" + "=" * 80)
if all_have_translations:
    print("SUCCESS: All pyramid levels have translation fields!")
else:
    print("FAILURE: Some pyramid levels are missing translation fields")
print("=" * 80)

# Verify the translation values follow the expected pattern
print("\nVerifying translation values follow expected pattern:")
print("Expected pattern for 2x downsampling:")
print("  Level 0 (scale 1x):  translation = [0.0, 0.0, 0.0, 0.0, 0.0]")
print("  Level 1 (scale 2x):  translation = [0.0, 0.0, 0.5, 0.5, 0.5]")
print("  Level 2 (scale 4x):  translation = [0.0, 0.0, 1.5, 1.5, 1.5]")
print("  Level 3 (scale 8x):  translation = [0.0, 0.0, 3.5, 3.5, 3.5]")
print("  Level 4 (scale 16x): translation = [0.0, 0.0, 7.5, 7.5, 7.5]")

print("\nActual translations:")
for i, dataset in enumerate(datasets):
    transforms = dataset.get("coordinateTransformations", [])
    for transform in transforms:
        if transform.get("type") == "translation":
            translation = transform.get("translation")
            print(f"  Level {i}: translation = {translation}")

print("\n" + "=" * 80)
print("Test completed!")
print("=" * 80)
