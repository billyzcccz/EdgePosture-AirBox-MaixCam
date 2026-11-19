# Mechanical & Enclosure Design

3D-printed mechanical assemblies designed for desktop sitting posture tracking with the **Sipeed MaixCAM** (RISC-V / SG2002) front-end node.

## Files

| File | Description | Recommended Material |
| :--- | :--- | :--- |
| `servo_gimbal_bracket.step` | Dual-axis SG90 pan-tilt gimbal bracket | PETG (heat-resistant near servos) |
| `maixcam_case_top.stl` | MaixCAM protective shell (upper) | PLA / PETG |
| `maixcam_case_bottom.stl` | MaixCAM protective shell (lower) | PLA / PETG |
| `maixcam_pro_extension_shell.step` | Extended enclosure for MaixCAM Pro | PETG |
| `cad_parts/*.SLDPRT`, `*.SLDASM` | SolidWorks native parts for desk clamp, gimbal base, servo arm, and mounting brackets | PETG / PLA |

## Design Highlights

- **Universal Desk Clamp (`底座总装.SLDASM`)**: Embedded M5 nut and screw clamp that attaches to desk edges or sits flat on a tabletop.
- **2-DOF Lightweight Gimbal**: Orthogonal dual SG90 servos (180° each) with reinforced 3 mm PETG ribs to suppress resonance and servo heat deformation.
- **Quick-Release Camera Mount**: Trapezoidal slot interface on the camera backplate for fast assembly and debugging.
