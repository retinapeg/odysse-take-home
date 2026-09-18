'''Does the capped, cross-scored router gain (+1.2 points) beat a router trained on scrambled
driver histories?

Rebuilds the prototype router from notebooks/Odysse_Ride_Analysis.ipynb (data prep from cells
6-8, router from cells 55-61), checks it reproduces the notebook's asserted numbers (cell 75),
then reruns the same pipeline with both models trained on scrambled driver identities: the
training driver ids are relabelled by a random derangement of the known drivers, so each
driver's history stays internally coherent but is attributed to a different driver. Test
offers, candidate lists and workloads stay real.

Run from the repo root:  python verification/scrambled_capped_router.py
'''
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

SEED = 42
N_SCRAMBLES = 20
ROOT = Path(__file__).resolve().parents[1]

# ---- Data prep (notebook cells 6-8, only the columns the router needs) --------------------
offers = pd.read_excel(ROOT / 'data/raw/interview-take-home-dataset.xlsx', sheet_name='Trips')
offers = offers.sort_values(['driver_id', 'dispatched_datetime'], kind='stable')
offers = offers.reset_index(drop=True)
offers['accepted'] = offers.order_accepted_timestamp.notna().astype(int)


def km_between(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    half_chord = (np.sin((lat2 - lat1) / 2) ** 2
                  + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0088 * np.arcsin(np.sqrt(half_chord))


CENTRE = (51.5074, -0.1278)
pickup = offers.pickup_lat_long.str.split(',', expand=True).astype(float)
dropoff = offers.dropoff_latlong.str.split(',', expand=True).astype(float)
km_from_centre = km_between(pickup[0], pickup[1], *CENTRE)
offers['trip_km'] = km_between(pickup[0], pickup[1], dropoff[0], dropoff[1])
offers['log_km'] = np.log1p(offers.trip_km)
offers['pickup_ring'] = pd.cut(km_from_centre, [0, 3, 8, 15, np.inf], right=False,
                               labels=['Central', 'Inner', 'Outer', 'Far']).astype(str)
offers['hour_bin'] = (offers.dispatched_datetime.dt.hour // 3).astype(str)
offers['day'] = offers.dispatched_datetime.dt.day

# ---- Router (notebook cells 55-61) ------------------------------------------------------
train = offers[offers.day <= 20]
known_drivers = sorted(train.driver_id.unique())
test = offers[(offers.day >= 21) & offers.driver_id.isin(known_drivers)]
test = test.sort_values('dispatched_datetime')
ring_and_hour = pd.get_dummies(offers[['pickup_ring', 'hour_bin']], dtype=float)
km_scaled = (offers.log_km - train.log_km.mean()) / train.log_km.std()
finished_trips = offers[offers.status == 'Finished']


def router_inputs(offer_ids, driver_ids, with_trip_length):
    who = np.equal.outer(np.asarray(driver_ids), known_drivers).astype(float)
    km = km_scaled[offer_ids].to_numpy()[:, None]
    blocks = [ring_and_hour.loc[offer_ids].to_numpy(), km, who]
    if with_trip_length:
        blocks.append(who * km)
    return np.hstack(blocks)


def nearby_candidates(offer):
    seconds_apart = (test.dispatched_datetime - offer.dispatched_datetime).dt.total_seconds().abs()
    nearby = test[(test.pickup_ring == offer.pickup_ring) & (seconds_apart <= 600)]
    others = []
    for driver in sorted(nearby.driver_id.unique()):
        driver_trips = finished_trips[finished_trips.driver_id == driver]
        on_trip = ((driver_trips.order_accepted_timestamp < offer.dispatched_datetime)
                   & (driver_trips.dropoff_trip_datetime > offer.dispatched_datetime)).any()
        if driver != offer.driver_id and not on_trip:
            others.append(driver)
    return [offer.driver_id] + others


# Candidates and workloads depend only on the real test data, so they are built once.
candidates = {offer_id: nearby_candidates(offer) for offer_id, offer in test.iterrows()}
workload = test.driver_id.value_counts()
pairs = pd.DataFrame([(o, d) for o, ds in candidates.items() for d in ds],
                     columns=['offer', 'driver'])
MODELS = [('by driver only', False), ('by driver and trip', True)]
pair_inputs = {name: router_inputs(pairs.offer, pairs.driver, trip) for name, trip in MODELS}


def build_scores(train_driver_ids):
    '''Fit both router models with the given training driver labels and score every
    offer x candidate pair (cells 57 and 59).'''
    frame = pairs.copy()
    for name, with_trip_length in MODELS:
        inputs = router_inputs(train.index, train_driver_ids, with_trip_length)
        model = LogisticRegression(C=0.3, max_iter=3000).fit(inputs, train.accepted)
        frame[name] = model.predict_proba(pair_inputs[name])[:, 1]
    return frame.set_index(['offer', 'driver']).to_dict('index')


def route(score, policy, slack=None):
    '''Cell 60, with the score table passed in.'''
    given = dict.fromkeys(workload.index, 0)
    seen = dict.fromkeys(workload.index, 0)
    rng = np.random.default_rng(SEED)
    chosen = {}
    for offer_id, drivers_nearby in candidates.items():
        seen[drivers_nearby[0]] += 1
        free = []
        for driver in drivers_nearby:
            if slack is None:
                has_room = given[driver] < workload[driver]
            else:
                has_room = given[driver] < seen[driver] + slack
            if has_room:
                free.append(driver)
        if policy == 'actual dispatch' or not free:
            pick = drivers_nearby[0]
        elif policy == 'random nearby driver':
            pick = rng.choice(free)
        else:
            pick = max(free, key=lambda driver: score[(offer_id, driver)][policy])
        given[pick] += 1
        chosen[offer_id] = pick
    return pd.Series(chosen)


def scored(score, assignment, model):
    return np.mean([score[pair][model] for pair in assignment.items()])


def gains(score, judge=None):
    '''Gains in points over actual dispatch, each judged by the named scoring model. Routing
    uses `score`; grading uses `judge` (default: the same, possibly scrambled, models). The
    baseline is rescored by the grading model every time.'''
    judge = score if judge is None else judge
    actual = route(score, 'actual dispatch')
    trip_capped = route(score, 'by driver and trip', slack=3)
    driver_capped = route(score, 'by driver only', slack=3)
    trip_free = route(score, 'by driver and trip')

    def gain(assignment, model):
        return 100 * (scored(judge, assignment, model) - scored(judge, actual, model))

    return {'trip router, capped, scored by driver-only (headline)': gain(trip_capped, 'by driver only'),
            'trip router, capped, self-scored': gain(trip_capped, 'by driver and trip'),
            'driver-only router, capped, self-scored': gain(driver_capped, 'by driver only'),
            'driver-only router, capped, scored by trip model': gain(driver_capped, 'by driver and trip'),
            'trip router, uncapped, self-scored (the +4.1)': gain(trip_free, 'by driver and trip')}


# ---- Gate: the real pipeline must reproduce the notebook -----------------------------------
real_score = build_scores(train.driver_id)
policies = {'actual dispatch': {}, 'random nearby driver': {}, 'by driver only': {},
            'by driver and trip': {}}
check = {p: route(real_score, p) for p in policies}
check['driver only capped'] = route(real_score, 'by driver only', slack=3)
check['trip capped'] = route(real_score, 'by driver and trip', slack=3)
trip_scores = [round(float(scored(real_score, a, 'by driver and trip')), 3) for a in check.values()]
driver_scores = [round(float(scored(real_score, a, 'by driver only')), 3) for a in check.values()]
assert len(test) == 2221
assert trip_scores == [0.371, 0.371, 0.395, 0.412, 0.383, 0.395], trip_scores
assert driver_scores[2:] == [0.398, 0.394, 0.384, 0.383], driver_scores
print('Gate passed: rebuilt pipeline reproduces the notebook (cell 75 asserts).')
print(f'  scored by trip model:        {trip_scores}')
print(f'  scored by driver-only model: {driver_scores}\n')

real = gains(real_score)

# ---- Scrambled driver histories -------------------------------------------------------------
rng = np.random.default_rng(SEED)


def derangement(items):
    '''Random permutation with no fixed points, so every history goes to a different driver.'''
    while True:
        shuffled = rng.permutation(items)
        if not np.any(shuffled == np.asarray(items)):
            return dict(zip(items, shuffled))


scrambled, judged_by_real = [], []
for i in range(N_SCRAMBLES):
    mapping = derangement(known_drivers)
    scrambled_score = build_scores(train.driver_id.map(mapping))
    scrambled.append(gains(scrambled_score))
    # Same scrambled routes, graded by the REAL models: isolates self-agreement from signal.
    judged_by_real.append(gains(scrambled_score, judge=real_score))
    print(f'  scramble {i + 1:2d}: headline {scrambled[-1][next(iter(real))]:+.2f} points '
          f'(graded by real models: {judged_by_real[-1][next(iter(real))]:+.2f})')
scrambled, judged_by_real = pd.DataFrame(scrambled), pd.DataFrame(judged_by_real)

# ---- Report -----------------------------------------------------------------------------------
table = pd.DataFrame({'real': pd.Series(real),
                      'scrambled mean': scrambled.mean(),
                      'scrambled min': scrambled.min(),
                      'scrambled max': scrambled.max(),
                      'scrambled sd': scrambled.std()})
table['real − scrambled mean'] = table.real - table['scrambled mean']
table['real rank / 21'] = [1 + (scrambled[c] >= real[c]).sum() for c in table.index]
pd.set_option('display.width', 200)
print(f'\nGain over actual dispatch, points ({N_SCRAMBLES} scrambles, both models trained on '
      'deranged driver ids; rank 1 = real beats every scramble):')
print(table.round(3).to_string())
print('\nScrambled routes graded by the real models instead (mean, min, max):')
print(judged_by_real.agg(['mean', 'min', 'max']).T.round(3).to_string())
