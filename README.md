# QT-arithmetic_study
Qt/PyQt5 arithmetic difficulty study that sends LSL markers for each trigger.

## Quickstart
1. Create a virtual environment and install dependencies.
2. Run the experiment from the `Arithmetic_Difficulty` directory so the question set is found.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cd Arithmetic_Difficulty
python mainExperiment.py
```

## Requirements
- Python 3.12+ (developed on Windows; update if your environment differs)
- PyQt5, pylsl, numpy (see `requirements.txt`)
- A GUI display server (X11/Wayland on Linux)
- An LSL recorder/receiver if you want to capture markers
- On some Linux setups, `pylsl` may require a separate `liblsl` install

## Reproducibility notes
- The experiment loads a pickled question set from a file named `GeneratedQuestions` in the current working directory.
- To keep runs reproducible, run from `Arithmetic_Difficulty` and use `Arithmetic_Difficulty/GeneratedQuestions`.
- `MakeQuestions.py` regenerates a question set using Python's `random` module with no fixed seed. Use the bundled `GeneratedQuestions` file if you need identical stimuli.
- The app does not write output files; all trial information is emitted as LSL string markers on the `arithmetic-Markers` stream.

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
