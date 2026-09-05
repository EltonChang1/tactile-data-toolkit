# Open-Tactile-Schema

Version: `open-tactile-schema/0.1`

Every converted dataset is a hierarchy of arrays that share a leading time axis `T`. `N` is the number of surface elements (taxels, or pooled gel pixels). Optional fields appear only when the corresponding modality is present.

| Path | Dtype | Shape | Unit | When required |
| --- | --- | --- | --- | --- |
| `/timestamps` | float64 | `(T,)` | s | always |
| `/observation/tactile/contact_mask` | bool | `(T, N)` | binary | always |
| `/observation/tactile/normal_force` | float32 | `(T, N)` | N | always |
| `/observation/tactile/shear_force` | float32 | `(T, N, 2)` | N | always |
| `/observation/tactile/point_cloud` | float32 | `(T, N, 6)` | m, N | always |
| `/observation/tactile/raw_image` | uint8 | `(T, H, W, 3)` | RGB | vision-tactile |
| `/observation/tactile/depth_map` | float32 | `(T, H, W)` | m | optional, vision-tactile |
| `/observation/tactile/pressure_map` | float32 | `(T, GH, GW)` | N | optional, taxel |
| `/observation/wrench` | float32 | `(T, 6)` | N, N·m | wrench present |
| `/observation/pose` | float32 | `(T, 7)` | m, quaternion | optional, pose present |

Conventions:

- Point-cloud columns are `(x, y, z, fx, fy, fz)` in the sensor frame. `x` right, `y` down, `z` out of the surface. Indentation moves `z` negative.
- Shear `(fx, fy)` is in the surface plane. Normal `fz` is positive into the sensor.
- Wrench is `(Fx, Fy, Fz, Tx, Ty, Tz)` at the wrist.
- Pose is `(x, y, z, qx, qy, qz, qw)`.
- Gel images are RGB, not BGR.

## Validation

```python
from tactile_toolkit.export import open_zarr, schema_view
from tactile_toolkit.schema import validate

report = validate(open_zarr("dataset.zarr"))
report.raise_if_invalid()

report = validate(schema_view("dataset.h5", demo=0))
print(report.summary())
```

The validator checks presence, dtype, rank, and that symbolic dimensions (`T`, `N`, `H`, `W`) are consistent across fields. `tactile validate path` is the command-line form for Zarr and HDF5.

Root attributes (Zarr / HDF5 `data` group) include `schema_version`, `modalities`, `sensors`, `fps`, and a compact quality-gate report. Per-feature Welford statistics live under `/stats`.
