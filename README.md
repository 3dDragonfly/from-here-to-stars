# From Here to Stars

Phone astrophotography with a Samsung Galaxy S10e: Windows capture controls,
RAW image stacking, and optional deterministic image cleanup.

From Here to Stars started as AstroCam. It uses the same S10e when moving
between computers, with a roadmap toward live stacking, target finding and
telescope tracking. Existing scripts and output files retain their AstroCam
filenames.

Repository: https://github.com/3dDragonfly/from-here-to-stars

## Current capabilities

- Capture a timed series using Samsung Camera in Pro mode and download new files.
- Select usable DNG frames and align stars across a wide field.
- Compare quality-weighted MAD and Winsorized sigma-clipped integration.
- Save linear float32 TIFF masters, viewing images and measurement reports.
- Optionally reduce background noise for display without changing the master.

Live stacking, Raspberry Pi deployment, mount control, automatic targeting and
tracking through clouds are planned. They are not implemented by this release.
See [TODO.txt](TODO.txt).

## Windows setup

Use 64-bit Windows and Python 3.13 (tested with 3.13.7). Enable Add Python to PATH
when installing Python. An SSD and 16 GB RAM are practical recommendations, not
measured minimum requirements. Leave room for captures and temporary images:
the stacker caches approximately 146 MB per accepted S10e frame, plus results.

1. Download or clone this repository.
2. Install dependencies from PowerShell in the repository directory:

   ```powershell
   python -m pip install -r ".\AstroCam-Stack\requirements.txt"
   ```

3. Download the Windows package from the official
   [scrcpy project](https://github.com/Genymobile/scrcpy). Extract its complete
   contents beside `AstroCam-Capture.cmd`. Keep the bundled ADB files and DLLs
   together. Third-party binaries are not included here.
4. Enable USB debugging on the S10e, connect it with a data cable, and authorize
   this computer on the phone. Each new computer needs its own authorization.

## Capture

Open Samsung Camera, select Pro mode, enable RAW saving, and set exposure,
ISO and focus manually. The exposure field controls scheduling;
it does not set the camera's actual shutter speed. Match those values yourself.
The current Pro-mode check expects the camera's English UI label `Pro, Mode`.

Double-click `AstroCam-Capture.cmd`. With exactly one authorized ADB device,
the script discovers it automatically. If several connections exist, supply
the S10e's ADB serial explicitly:

```powershell
.\AstroCam-Capture.cmd -Device "YOUR_S10E_ADB_SERIAL"
```

For an already configured ADB Wi-Fi connection, use its `IP:port` as the serial.
The script does not establish wireless debugging itself. Captures are downloaded
after the series finishes into `AstroCam-billeder`. Stopping early leaves photos
on the phone. The original files are not deleted.

## Stack existing DNG images

Double-click `AstroCam-Stack.cmd` and select one capture folder, or run:

```powershell
python .\AstroCam-Stack\astrocam_stack.py --input "D:\AstroCam\session"
```

The default v0.4 pipeline compares both integration methods. Use `--legacy`
for the earlier v0.3 pipeline. Results go to a new folder inside the session;
input DNG files are opened read-only. See
[the detailed v0.4 notes](AstroCam-Stack/LAES-MIG-v0.4.txt).

## Optional viewing cleanup

```powershell
python .\AstroCam-Stack\astrocam_cleanup.py --source "D:\AstroCam\session\Stack_v04_RESULT" --output "D:\AstroCam\cleaned"
```

Replace these example paths with the actual stack directory and a new output
directory. Cleanup outputs are stretched, nonlinear viewing images. Background
removal and smoothing can suppress faint diffuse structures. Preserve the linear
master and RAW files for measurements. No generated or replacement stars are used.

## Validation and limitations

Run the synthetic registration, flux preservation and outlier rejection checks:

```powershell
python .\tests\test_advanced.py
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tests\test_capture_connection.ps1
```

The existing pipeline has processed a real 18-frame S10e series. Better alignment
gave sharper stars, but did not improve every noise metric. Neither stacking
method is universally best. Dark/flat calibration is still planned; current
display cleanup is not a substitute for sensor calibration. Capture portability
changes have not yet been verified with the physical phone on a second computer.

The repository contains source and documentation. Personal captures, location
metadata, generated results, local dependencies and experimental field-specific
scripts are not included. A software license has not yet been selected.

## Planned tracking architecture

The laptop or Raspberry Pi would measure multi-star drift before image alignment,
maintain a live stack, and send corrections to a compatible motor controller.
The controller would maintain continuous motor timing between image updates.
During clouds, prediction would continue from reliable tracking history, with
learning paused and uncertainty displayed. Recovery would verify the original
star field before correcting accumulated drift. Actual hardware support depends
on the board, firmware, mount geometry and measured performance.
