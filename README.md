# Odysse take-home: right trip, right driver

**Question:** can Odysse cut the 61% of offers that go unaccepted by sending each trip to the driver
most likely to take it, without giving anyone more work?

**Answer in short.** The trips don't explain it: they predict 55–64% not accepted for every driver,
against 8–84% observed. Each driver's pattern is stable: three days predict the other three
(r = 0.83). Drivers prefer different trip lengths, in opposite directions, so the fleet average hides
it (r ≈ 0.77, p ≈ 0.001). That is a lever dispatch isn't pulling. A prototype router shows the
mechanism but cannot size it offline: checked against routers built on scrambled driver histories,
only about +0.2 points of acceptance survives. An A/B test at equal workload is what would measure it.

## Just want to read it?

Nothing to install. GitHub renders both of these in the browser:

- [The analysis notebook](notebooks/Odysse_Ride_Analysis.ipynb), already executed, with every chart and number.
- [The deck (PDF)](deck/Odysse_deck.pdf).

## Run it in your browser

You need **Python 3.11 or newer** and git. Everything else is installed by the commands below.

```bash
git clone https://github.com/retinapeg/odysse-take-home.git
cd odysse-take-home
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/Odysse_Ride_Analysis.ipynb
```

On Windows, replace the `source` line with `.venv\Scripts\activate`.

JupyterLab opens the notebook in your browser. Choose **Run → Run All Cells**; it takes about a
minute and a half. The last cells re-assert every number quoted in the text and print
`All quoted numbers reproduce.`

To run the whole notebook from the terminal instead, without opening a browser:

```bash
python run_notebook.py
```

Dependencies (all in `requirements.txt`): pandas, numpy, scipy, scikit-learn, matplotlib, openpyxl,
jupyterlab, nbformat, nbclient, ipykernel. The notebook reads only the workbook in `data/raw/`, and
every random step starts from the one `SEED` in the first code cell, so reruns match.

## What's here

| Path | What it is |
|---|---|
| `notebooks/Odysse_Ride_Analysis.ipynb` | The analysis, start to finish. Each step names the deck slides it backs. |
| `deck/Odysse_deck.pdf`, `deck/Odysse_deck.pptx` | The walkthrough deck. |
| `deck/build/charts/hires/` | The notebook's charts as used in the deck, named by notebook cell. |
| `data/raw/interview-take-home-dataset.xlsx` | The dataset as supplied. |
| `verification/` | Standalone scripts that check the notebook's claims (below). |
| `run_notebook.py` | Runs the notebook in a fresh kernel from the terminal. |
| `requirements.txt` | The dependencies. |

## Checking the claims

Each script in `verification/` rebuilds its part of the analysis from the raw workbook and prints
its numbers. Run them from the repo root, with the virtualenv active:

```bash
python verification/robustness_checks.py        # step 1: do the trips explain it, with better models of the request?
python verification/router_checks.py            # step 5: the router's self-scored +4 against scrambled driver histories
python verification/scrambled_capped_router.py  # step 5: the capped, cross-graded +1.2 against the same scramble
```

## How to read the notebook

Seven short steps, each ending with a plain-English takeaway: what counts as unaccepted; is it the
trips (no); does each driver have a consistent pattern (yes); do drivers prefer different trips (yes,
in opposite directions); two signals tested on unseen days; a prototype router; what a trip is worth
per minute of driving. Then the recommendations and the limitations.
