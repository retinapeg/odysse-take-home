# Odysse take-home: right trip, right driver

**Question:** can Odysse cut the 61% of offers that go unaccepted by sending each trip to the driver
most likely to take it, without giving anyone more work?

**Answer in short.** It isn't explained by the trips (they predict 55–64% not accepted for every
driver, against 8–84% observed). Each driver's pattern is stable: three days predict the other three
(r = 0.83). Drivers prefer different trip lengths, in opposite directions, so the fleet average hides
it (r ~ 0.77, p ~ 0.001). A prototype router on two unseen days suggests about 1–2 points more
acceptance at near-equal workload: a modelled estimate, enough to justify an A/B test.

## What's here

- `notebooks/Odysse_Ride_Analysis.ipynb` — the analysis, start to finish, already executed.
- `deck/Odysse_deck.pptx`, `deck/Odysse_deck.pdf` — the walkthrough deck. Each notebook step names the slides it backs.
- `data/raw/interview-take-home-dataset.xlsx` — the dataset as supplied.
- `requirements.txt`, `run_notebook.py` — everything needed to reproduce it.

## Reproducing it

    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    python run_notebook.py

Or open the notebook and Run All. It reads only the workbook in `data/raw/`, and every random step
starts from the one SEED in the first code cell, so reruns match. The last four code cells re-assert every
number quoted in the text and print "All quoted numbers reproduce."

## How to read it

Seven short steps, each ending with a plain-English takeaway: what counts as unaccepted; is it the
trips (no); does each driver have a consistent pattern (yes); do drivers prefer different trips (yes,
in opposite directions); two signals tested on unseen days; a prototype router; what a trip is worth
per minute of driving. Then the recommendations and the limitations.
