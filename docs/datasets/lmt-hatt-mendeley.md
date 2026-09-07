# LMT, Penn HaTT, and Mendeley tactile textures

This guide describes the Step 07 support boundary verified on 2026-09-06. Importing or opening a record performs no network access, LMT remains discovery-only because its reuse terms are unclear, and code-capable Mendeley pickle is never opened without `trust_pickle=True`.

## Capability summary

| Record | Support | What works | Deliberate boundary |
| --- | --- | --- | --- |
| `lmt-textures` | C2 discoverable | Institutional provenance, terms warning, citations, and separate 69-, 108-, and 184-material child records | No download or parser without substantive reuse terms and a verified native manifest |
| `penn-hatt` | C3 loadable | Bounded recorded-data XML parsing with units, sample rate, acceleration, force, position, and speed checks | User-obtained noncommercial copy only; rendering/model files are not interpreted |
| `mendeley-tactile-textures` | C3 loadable | Confirmed and verified 3.83 GB acquisition plus paired pressure/IMU sequences from the extracted V1 layout | Explicit trusted pickle; units and canonical splits are not published |

`list_datasets()` and `get_dataset_metadata()` expose all four LMT records and both loadable datasets. `open_dataset()` constructs the Penn and Mendeley adapters only from user-provided local paths and never acquires or extracts data implicitly.

## LMT Haptic Texture Database

The Technical University of Munich page describes controlled and freehand exploration data, while its official archive lists distinct bundles for 69, 108, and 184 materials. The toolkit registers `lmt-textures` as an umbrella plus `lmt-textures-69`, `lmt-textures-108`, and `lmt-textures-184`; their listed decimal sizes are 6,934 MB, 3,190 MB, and 83,484 MB, respectively.

The current archive URL is `https://zeus.lkn.ei.tum.de/downloads/texture/`, but it did not produce a reliable automated response during verification. More importantly, the institutional page's `Licence` section supplies no substantive grant, so the runtime records are C2 references and `open_dataset("lmt-textures")` intentionally reports that no adapter exists.

The umbrella cites both the original 2014 texture-database paper and the 2017 multimodal material-classification paper. Choose and cite the child release actually used; the toolkit does not pretend the different material counts, protocols, annotations, or archive sizes are versions of one interchangeable file.

## Penn Haptic Texture Toolkit / HaTT

The official Penn technical report documents 100 approximately homogeneous and isotropic textures across ten material categories. It describes 1024-by-1024 surface images at about 15 pixels/mm and two ten-second, 10 kHz recordings per texture, along with data-driven rendering models and code.

The attached Penn license permits attributed noncommercial research rather than unrestricted open-data reuse. Obtain the files under those terms and point the adapter only at your local recorded-data tree:

```bash
python examples/datasets/penn_hatt_local.py /data/PennHaTT/RecordedData --limit 2
```

The XML reader accepts the documented `SampleRate`, acceleration, force, position, speed, and unit fields in compound or nested-axis form. It bounds each XML file, rejects DTD/entity declarations, validates finite numeric sequences and complete three-axis vectors, preserves each source path as a group, and ignores model XML that contains none of the documented recording channels.

The adapter does not render virtual textures, interpret proprietary model parameters, associate images by guessed filenames, or invent evaluation splits. Those capabilities require an active publisher manifest and representative lawful fixtures rather than deductions from the report.

## Multimodal Tactile Texture Dataset on Mendeley Data

Mendeley Data V1 is DOI `10.17632/n666tk4mw9.1`, published 2023-08-15 under CC BY 4.0. Its versioned public API exposes a 3,831,798,836-byte ZIP with SHA-256 `56468a6bc7191b46e12f09fece605d117bf29e8c678ae458b885e01a41917572` plus a 29,865-byte reader notebook.

Acquisition is explicit, resumable, and integrity checked; it refuses to start until the caller acknowledges the large transfer:

```python
from tactile_toolkit.datasets import download_mendeley_textures

archive = download_mendeley_textures(confirm_large_download=True)
print(archive)
```

Extract the verified archive in an isolated location with an archive tool that rejects absolute and parent-traversal members. Review the contents before granting pickle trust, install a pandas version capable of reading the legacy pandas 0.14.1 serialization when necessary, and inspect a bounded prefix:

```bash
python examples/datasets/mendeley_textures_local.py /data/mendeley-textures \
  --trust-pickle --limit 2
```

The adapter recognizes only `pickles_30`, `pickles_35`, and `pickles_40`, textures `01` through `12`, and paired `full_baro/baro*.pkl` plus `full_imu/imu*.pkl` trials. It validates the `baro` column and nine published IMU columns, converts a native `DatetimeIndex` to relative seconds, keeps pressure and IMU clocks separate, rejects non-finite values and missing pairs, and ignores derived single-axis directories to prevent duplicate samples.

The immutable ZIP README and directory structure say the exploration speeds are 30, 35, and 40 mm/s, although the Mendeley landing-page prose says 30, 40, and 45 mm/s. The toolkit follows the released artifact, records the discrepancy, leaves sensor units unset because the publisher does not define them, and does not fabricate fixed train/validation/test splits.
