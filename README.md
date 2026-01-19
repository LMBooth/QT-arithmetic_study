[![DOI](https://zenodo.org/badge/659735771.svg)](https://doi.org/10.5281/zenodo.18303716)

# QT-arithmetic_study
Qt/PyQt5 arithmetic difficulty study that sends LSL markers for each trigger.

Code used to run these experiments and convert the original xdf files to BIDS [can be found here.](https://github.com/LMBooth/QT-arithmetic_study)

## Quickstart
1. Create a virtual environment and install dependencies.
2. Run the experiment from the `Arithmetic_Difficulty` directory so the question set is found.

Create the venv:
```bash
python -m venv .venv
```

Activate it (pick your shell):
```powershell
.\.venv\Scripts\Activate.ps1
```
```cmd
.\.venv\Scripts\activate.bat
```
```bash
source .venv/bin/activate
```

Install requirements and run:
```bash
python -m pip install -r requirements.txt
cd Arithmetic_Difficulty
python mainExperiment.py
```

## Data capture and conversion
- The experiment emits LSL markers on the `arithmetic-Markers` stream; use Lab Recorder (or another LSL recorder) to save XDF files.
- Store raw recordings under `Arithmetic_Data/sourcedata` (this repo includes the folder as the expected input location).
- The arithmetic-only BIDS conversion pipeline lives in `conversion_package/`; follow `conversion_package/README.md` to rebuild `bids_arithmetic` from the XDFs and demographics.
- The QC plotter is included in `conversion_package/qc` for inspecting EEG/ECG/pupil with event overlays after conversion; see `conversion_package/qc/README.md` for usage.

## Requirements
- Python 3.11+ 
- OS: Windows (developed on Windows 11 Professional)
- PyQt5, pylsl, numpy (see `requirements.txt`)
- An LSL recorder/receiver if you want to capture markers
- On some Linux setups, `pylsl` may require a separate `liblsl` install

## Reproducibility notes
- The experiment loads a pickled question set from a file named `GeneratedQuestions` in the current working directory.
- To keep runs reproducible, run from `Arithmetic_Difficulty` and use `Arithmetic_Difficulty/GeneratedQuestions`.
- `MakeQuestions.py` regenerates a question set using Python's `random` module with no fixed seed. Use the bundled `GeneratedQuestions` file if you need identical stimuli.
- The app does not write output files; all trial information is emitted as LSL string markers on the `arithmetic-Markers` stream.

## Question file format
`GeneratedQuestions` is a pickled Python list. Each entry contains a set of questions and the Q range:
`[ [ [n1, n2], [n1, n2], ... ], [qmin, qmax] ]`

## Marker format
Markers are single strings on the `arithmetic-Markers` LSL stream.
- "Started tutorial artihmetic" and "Started arithmetic" are emitted at block start.
- For each trial, a difficulty marker is emitted as `qmin-qmax` (one decimal place).
- After an answer, a result marker is emitted as `qmin-qmax Correct` or `qmin-qmax Wrong` (plus ` tutorial` during the tutorial).
- "Finished tutorial Arithmetic" and "Finished Arithmetic" are emitted at block end.

## Timing parameters
- Tutorial starts with a 60 second fixation ("x") period.
- Each question is displayed for 6 seconds before the input box appears.
- After an answer, the next trial begins after 0.5 seconds.

## References
- H. B. G. Thomas (1963), Communication theory and the constellation hypothesis of calculation, Quarterly Journal of Experimental Psychology, 15:3, 173-191. DOI: 10.1080/17470216308416323

## Regenerating question sets
```powershell
cd Arithmetic_Difficulty
python MakeQuestions.py
```

## Docker (optional)
This example targets Linux hosts with an X11 server. You may need an X server or WSLg on Windows/macOS.

```bash
docker build -t qt-arithmetic .
xhost +local:docker
docker run --rm -it --network host \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  qt-arithmetic
```

## Citation
See `CITATION.cff` for citation metadata. Update it with a DOI after creating the Zenodo record.

## License
This repository is released under CC0 1.0 Universal. See `LICENSE`.
