# CAD-Map-Region-Extractor
# MAP-EXTRACTOR

MAP-EXTRACTOR traces separate green areas in an image and exports their
outlines to a DXF file for use in CAD software.

It is intended for site plans, maps, and building footprints where the
areas to extract are marked in green. Straight-edged shapes can be
adjusted to follow their own detected orientation. Curved outlines can
keep their contour instead of being forced into right-angle geometry.

## Download the Released Windows Build

Open the [latest GitHub release](../../releases/latest) and download `MAP-EXTRACTOR.exe` from the **Assets** section.

Run the downloaded executable on Windows. Python does not need to be installed to use the packaged build.

Check the release notes for any additional files or instructions.


## Features

-   Detects green regions using HSV colour thresholds.
-   Exports each detected outline as a closed DXF polyline.
-   Places outlines on the `GREEN` layer.
-   Can trace holes inside detected regions on the `HOLES` layer.
-   Smooths and simplifies outlines to reduce pixel noise.
-   Detects shape orientation and aligns right-angle geometry to that
    orientation.
-   Supports grid-based staircase outlines for additional detail.
-   Allows a fixed angle or angle snapping.
-   Can save the detected colour mask for checking which pixels were
    selected.
-   Includes a graphical interface and command-line options.

## Requirements

For running the Python script:

-   Python 3.10 or newer is recommended.
-   OpenCV
-   ezdxf
-   NumPy
-   FreeSimpleGUI

Install the packages with:

``` bash
py -m pip install opencv-python ezdxf numpy FreeSimpleGUI
```

## Run the application

### Graphical interface

Place `MAP-EXTRACTOR.py` in your working folder, then run:

``` bash
py MAP-EXTRACTOR.py
```

You can also open the GUI explicitly:

``` bash
py MAP-EXTRACTOR.py --gui
```

Choose an input image and an output `.dxf` file, adjust the settings,
then run the extraction.

### Command line

Basic example:

``` bash
py MAP-EXTRACTOR.py input.png output.dxf
```

Other supported image formats can be used if OpenCV can read them.

Trace interior holes as well:

``` bash
py MAP-EXTRACTOR.py input.png output.dxf --holes
```

Set the DXF scale to 0.1 drawing units per image pixel:

``` bash
py MAP-EXTRACTOR.py input.png output.dxf --scale 0.1
```

Save the detected colour mask to inspect the selected pixels:

``` bash
py MAP-EXTRACTOR.py input.png output.dxf --save-mask mask.png
```

Use a specific colour instead of the default green thresholds:

``` bash
py MAP-EXTRACTOR.py input.png output.dxf --color "#2E9E3A"
```

In PowerShell, keep the quotes around the hex colour value.

## Shape and angle settings

### Orientation and angle snapping

By default, the program detects each shape's orientation and snaps it to
the nearest multiple of 15 degrees. The right-angle outline is then
constructed relative to that orientation.

Use a 30-degree step when working with plans that follow 30-degree
increments:

``` bash
py MAP-EXTRACTOR.py input.png output.dxf --angle-step 30
```

Disable angle snapping while keeping automatic orientation detection:

``` bash
py MAP-EXTRACTOR.py input.png output.dxf --angle-step 0
```

Force every shape to use a specified angle:

``` bash
py MAP-EXTRACTOR.py input.png output.dxf --angle 30
```

Positive angle values rotate counter-clockwise in the DXF coordinate
system.

### Grid mode

Grid mode creates staircase-shaped outlines using cells of the chosen
size, in each shape's rotated frame:

``` bash
py MAP-EXTRACTOR.py input.png output.dxf --grid 6
```

Larger grid values produce chunkier outlines. Smaller values retain more
detail.

### Curved outlines

Curved-shape detection is enabled by default. When a shape is identified
as curved, its smoothed contour is kept instead of applying the
right-angle rule.

Adjust curve detection and smoothing:

``` bash
py MAP-EXTRACTOR.py input.png output.dxf --curve-min-points 12 --curve-angle-tolerance 8 --curve-smooth 13
```

Apply the right-angle rule to all shapes, including curved ones:

``` bash
py MAP-EXTRACTOR.py input.png output.dxf --no-curve-detection
```

## Common options

  ---------------------------------------------------------------------------
  Option                   Purpose                                    Default
  ------------------------ --------------------- ----------------------------
  `--hue-min`              Minimum hue for green                         `35`
                           detection, on         
                           OpenCV's 0--179 scale 

  `--hue-max`              Maximum hue for green                         `85`
                           detection             

  `--sat-min`              Minimum saturation                            `60`

  `--val-min`              Minimum brightness                            `40`

  `--color`                Match a specified hex                          Off
                           colour instead of the 
                           default HSV limits    

  `--min-area`             Ignore regions                                `50`
                           smaller than this     
                           many pixels           

  `--simplify`             Simplify                                       `2`
                           straight-edged        
                           contours              

  `--smooth`               Smooth the mask                                `7`
                           before tracing; `0`   
                           disables it           

  `--min-step`             Remove short steps                             `5`
                           from right-angle      
                           outlines              

  `--angle-step`           Angle increment used                          `15`
                           for orientation       
                           snapping; `0`         
                           disables snapping     

  `--angle`                Force a fixed angle                        Not set
                           for every shape       

  `--grid`                 Grid cell size; `0`                            `0`
                           disables grid mode    

  `--scale`                DXF units per image                            `1`
                           pixel                 

  `--holes`                Trace holes inside                             Off
                           green regions         

  `--save-mask`            Save the binary                            Not set
                           detection mask to     
                           this path             

  `--no-curve-detection`   Apply right-angle                              Off
                           processing to curved  
                           shapes too            
  ---------------------------------------------------------------------------

Run this command to see all available arguments:

``` bash
py MAP-EXTRACTOR.py --help
```

## Colour detection tips

If the program misses green regions or selects unwanted areas:

1.  Save a mask with `--save-mask mask.png`.
2.  Open the mask image and check which pixels were detected.
3.  Adjust `--hue-min`, `--hue-max`, `--sat-min`, and `--val-min`, or
    use `--color` with the tolerance options.
4.  If the outline is too noisy, increase `--smooth` or `--simplify`. If
    small details disappear, lower those values.

When using `--color`, the `--hue-tol`, `--sat-tol`, and `--val-tol`
options control the allowed colour variation.

## Export notes

-   The exported outlines are closed polylines.
-   Main outlines are stored on the `GREEN` layer.
-   When `--holes` is enabled, interior holes are stored on the `HOLES`
    layer.
-   `--scale` sets the drawing units per pixel. Use a known dimension in
    the source image to calculate an appropriate scale before relying on
    measurements.
-   Image tracing depends on image resolution, colour contrast, and the
    selected threshold settings. Check the resulting DXF in your CAD
    software before using it for measured drawings.

## Build a Windows EXE

You can package the GUI as a Windows executable with PyInstaller.

Install PyInstaller:

``` powershell
py -m pip install --upgrade pyinstaller
```

From the folder containing `MAP-EXTRACTOR.py`, run:

``` powershell
py -m PyInstaller --clean --onefile --windowed --name "MAP-EXTRACTOR" --collect-all cv2 --collect-all ezdxf --collect-all FreeSimpleGUI MAP-EXTRACTOR.py
```

The executable will be created at:

``` text
dist\MAP-EXTRACTOR.exe
```

If the build fails or the application closes unexpectedly, build once
without `--windowed` so that error messages remain visible:

``` powershell
py -m PyInstaller --clean --onefile --name "MAP-EXTRACTOR" --collect-all cv2 --collect-all ezdxf --collect-all FreeSimpleGUI MAP-EXTRACTOR.py
```

Test the executable on your own computer before sharing it with other
users.

## Project files

-   `MAP-EXTRACTOR.py`: application source code.
-   `README.md`: project documentation.


## License

MAP-EXTRACTOR is licensed under the **GNU General Public License v3.0 (GPL-3.0)**.

See the [`LICENSE`](LICENSE) file for the full license terms.
