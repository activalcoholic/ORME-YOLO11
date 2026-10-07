# Source attribution

This repository includes a modified copy of the Ultralytics Python package.
The original license text and existing source notices are preserved.
The included source reports version 8.4.24. The original working directory
was an unpacked source tree; an upstream Git commit ID was not available.

Project source: https://github.com/ultralytics/ultralytics

ORME-specific changes include MBFEM and its parser registration, custom
bounding-box and repulsion losses, and underwater degradation augmentation.
The package remains named `ultralytics` so that the original imports and
existing custom checkpoint classes continue to resolve.

The main ORME configuration was recovered directly from the saved comparison
checkpoint rather than inferred from the latest experimental YAML.
The source training entry point and comparison scripts refer to different
experimental configurations; the release documents this distinction.
