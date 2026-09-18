"""Is the router's self-scored gain evidence of anything?

The first version of the router scored its own choices about +4 points above actual dispatch.
That number is not evidence. The router picks the candidate its model likes best and is then
graded by that same model, so it approves of itself by construction.

This script measures the size of that effect directly: it rebuilds the whole router on driver
histories that have been scrambled -- the driver labels in training are permuted, so whatever
the model believes about a driver belongs to somebody else. A router that has learned nothing
real should gain nothing. If it still scores itself about +4, then +4 is the floor, not the
finding.

Run from the repository root:  python verification/router_checks.py
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).parent))
from _common import SEED, load_offers

SHUFFLES = 20        # scrambled rebuilds; each one is a complete router


def scramble(driver_ids, rng, mode):
    """Two ways to destroy what the model knows about a driver.

    'relabel' permutes the mapping between drivers, so each history stays coherent but belongs
    to the wrong person: the model still learns that some drivers accept far more than others,
    it has simply attached that to the wrong names. 'shuffle' permutes the labels row by row,
    which destroys the between-driver differences as well.
    """
    if mode == 'relabel':
        names = np.array(sorted(pd.unique(driver_ids)))
        mapping = dict(zip(names, rng.permutation(names)))
        return pd.Series(driver_ids).map(mapping).to_numpy()
    return rng.permutation(np.asarray(driver_ids))


def scramble(driver_ids, rng, whole_histories):
    """Two ways to destroy what the model knows about drivers, both worth reporting.

    whole_histories=True swaps identities: driver A's entire history is relabelled as driver B.
    Each driver still looks like a coherent person -- the same spread of acceptance rates
    survives -- but every history belongs to the wrong one. This is the harder floor, because a
    router can still gain simply by preferring whoever looks keen.

    whole_histories=False permutes the labels row by row, which dissolves the drivers entirely.
    """
    if not whole_histories:
        return rng.permutation(np.asarray(driver_ids))
    unique = np.array(sorted(pd.unique(driver_ids)))
    mapping = dict(zip(unique, rng.permutation(unique)))
    return np.array([mapping[d] for d in driver_ids])


def build(offers):
    """Everything the router needs: the train/test split, candidates, and the input builder."""
    train = offers[offers.day <= 20]
    known = sorted(train.driver_id.unique())
    test = offers[(offers.day >= 21) & offers.driver_id.isin(known)]
    test = test.sort_values('dispatched_datetime')

    ring_and_hour = pd.get_dummies(offers[['pickup_ring', 'hour_bin']], dtype=float)
    km_scaled = (offers.log_km - train.log_km.mean()) / train.log_km.std()
    finished = offers[offers.status == 'Finished']

    def inputs(offer_ids, driver_ids, with_trip_length):
        who = np.equal.outer(np.asarray(driver_ids), known).astype(float)
        km = km_scaled[offer_ids].to_numpy()[:, None]
        blocks = [ring_and_hour.loc[offer_ids].to_numpy(), km, who]
        if with_trip_length:
            blocks.append(who * km)
        return np.hstack(blocks)

    def nearby(offer):
        apart = (test.dispatched_datetime - offer.dispatched_datetime).dt.total_seconds().abs()
        close = test[(test.pickup_ring == offer.pickup_ring) & (apart <= 600)]
        others = []
        for driver in sorted(close.driver_id.unique()):
            trips = finished[finished.driver_id == driver]
            on_trip = ((trips.order_accepted_timestamp < offer.dispatched_datetime)
                       & (trips.dropoff_trip_datetime > offer.dispatched_datetime)).any()
            if driver != offer.driver_id and not on_trip:
                others.append(driver)
        return [offer.driver_id] + others

    candidates = {offer_id: nearby(offer) for offer_id, offer in test.iterrows()}
    return train, test, candidates, inputs


def route_and_score(train, test, candidates, inputs, labels, slack=None, cross=False):
    """Train on `labels` (real or scrambled), route the test days, and score the result.

    Returns the router's self-scored gain in predicted acceptance over actual dispatch, in
    percentage points, plus the share of offers it moved.
    """
    chooser = LogisticRegression(C=0.3, max_iter=3000).fit(
        inputs(train.index, labels, True), train.accepted)
    judge = chooser
    if cross:      # judged by the OTHER model, so nothing is graded by the model that chose it
        judge = LogisticRegression(C=0.3, max_iter=3000).fit(
            inputs(train.index, labels, False), train.accepted)

    pairs = [(offer_id, driver) for offer_id, drivers in candidates.items() for driver in drivers]
    frame = pd.DataFrame(pairs, columns=['offer', 'driver'])
    frame['p'] = chooser.predict_proba(inputs(frame.offer, frame.driver, True))[:, 1]
    score = {(o, d): p for o, d, p in frame.itertuples(index=False)}
    if cross:
        frame['q'] = judge.predict_proba(inputs(frame.offer, frame.driver, False))[:, 1]
        graded = {(o, d): q for o, d, q in
                  frame[['offer', 'driver', 'q']].itertuples(index=False)}
    else:
        graded = score

    workload = test.driver_id.value_counts()
    given = dict.fromkeys(workload.index, 0)
    seen = dict.fromkeys(workload.index, 0)
    chosen = {}
    for offer_id, drivers in candidates.items():
        seen[drivers[0]] += 1
        def has_room(d):
            if slack is None:
                return given[d] < workload[d]
            return given[d] < seen[d] + slack
        free = [d for d in drivers if has_room(d)]
        pick = max(free, key=lambda d: score[(offer_id, d)]) if free else drivers[0]
        given[pick] += 1
        chosen[offer_id] = pick

    assignment = pd.Series(chosen)
    routed = np.mean([graded[(o, d)] for o, d in assignment.items()])
    actual = np.mean([graded[(o, test.driver_id[o])] for o in assignment.index])
    moved = (assignment != test.driver_id[assignment.index]).mean()
    return (routed - actual) * 100, moved * 100


def main():
    offers = load_offers()
    train, test, candidates, inputs = build(offers)
    sizes = pd.Series({o: len(c) for o, c in candidates.items()})
    print(f'{len(test):,} test offers, {test.driver_id.nunique()} drivers | '
          f'median candidates {sizes.median():.0f} | with 2 or more: {(sizes >= 2).mean():.0%}\n')

    real_gain, real_moved = route_and_score(train, test, candidates, inputs, train.driver_id)
    print(f'Router on REAL driver histories, no workload cap: '
          f'scores itself {real_gain:+.1f} points, moves {real_moved:.0f}% of offers')

    rng = np.random.default_rng(SEED)
    results = {}
    for label, whole in [('identities swapped', True), ('labels dissolved', False)]:
        gains, moves = [], []
        for _ in range(SHUFFLES):
            scrambled = scramble(train.driver_id.to_numpy(), rng, whole)
            gain, moved = route_and_score(train, test, candidates, inputs, scrambled)
            gains.append(gain)
            moves.append(moved)
        results[label] = (np.array(gains), np.array(moves))
        gains, moves = results[label]
        print(f'Router on SCRAMBLED histories ({label}), {SHUFFLES} rebuilds: '
              f'{gains.mean():+.1f} points on average\n'
              f'{" " * 4}(range {gains.min():+.1f} to {gains.max():+.1f}), '
              f'moving {moves.mean():.0f}% of offers.')
    gains = results['identities swapped'][0]
    print('\nNeither has learned anything real about who accepts what, and both approve of '
          'themselves anyway.\n')

    print('Now the same two with a workload cap that holds (nobody ever more than 3 offers '
          'ahead\nof the work that actually came their way):\n')
    capped_real, capped_moved = route_and_score(train, test, candidates, inputs,
                                                train.driver_id, slack=3)
    capped_gains = []
    for _ in range(SHUFFLES):
        scrambled = scramble(train.driver_id.to_numpy(), rng, 'relabel')
        capped_gains.append(route_and_score(train, test, candidates, inputs,
                                            scrambled, slack=3)[0])
    capped_gains = np.array(capped_gains)
    print(f'  real histories      {capped_real:+.1f} points, moves {capped_moved:.0f}% of offers')
    print(f'  scrambled histories {capped_gains.mean():+.1f} points on average '
          f'(range {capped_gains.min():+.1f} to {capped_gains.max():+.1f})')

    print('\nAnd the figure the deck actually quotes, where the driver-and-trip router is '
          'graded by the\nDRIVER-ONLY model, so nothing is judged by the model that chose it:\n')
    cross_real = route_and_score(train, test, candidates, inputs, train.driver_id,
                                 slack=3, cross=True)[0]
    cross_null = []
    for _ in range(SHUFFLES):
        scrambled = scramble(train.driver_id.to_numpy(), rng, 'relabel')
        cross_null.append(route_and_score(train, test, candidates, inputs, scrambled,
                                          slack=3, cross=True)[0])
    cross_null = np.array(cross_null)
    print(f'  real histories      {cross_real:+.1f} points')
    print(f'  scrambled histories {cross_null.mean():+.1f} points on average '
          f'(range {cross_null.min():+.1f} to {cross_null.max():+.1f})')
    print(f'  difference          {cross_real - cross_null.mean():+.1f} points above a router '
          'that knows nothing')

    print(f'\nWhat this is for: a self-scored gain of {real_gain:+.1f} cannot be quoted as an '
          'improvement when a\nrouter that knows nothing scores itself '
          f'{gains.mean():+.1f}. The figure reported in the deck is the capped\none, held against '
          'this floor, and it is still a modelled offline estimate rather than a measured\nuplift. '
          'Only an A/B test at equal workload measures it.')


if __name__ == '__main__':
    main()
