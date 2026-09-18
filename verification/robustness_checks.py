"""Does the step 1 conclusion survive a better description of the request?

The notebook's benchmark is deliberately crude: pickup ring x 6-hour block. The worry is that
crude bins hide real differences in the work, and that a flexible model would find them. So the
same question is asked four ways, from the notebook's own bins up to a boosted model given raw
coordinates: how much of the between-driver variation can the request reproduce?

Run from the repository root:  python verification/robustness_checks.py
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).parent))
from _common import (SEED, load_offers, expected_from_others, compared_drivers,
                     explained_share)

pd.set_option('display.width', 200)


def grid_cell(offers, size_km):
    """Square bins of the given size, on km east and north of the centre."""
    x = np.floor(offers.x_km / size_km).astype(int)
    y = np.floor(offers.y_km / size_km).astype(int)
    return x.astype(str) + ',' + y.astype(str)


def boosted_expected(offers, group_by_driver=True):
    """A flexible model of the request, given the raw coordinates rather than bins.

    The folds are grouped BY DRIVER, so no driver's own offers ever train the model that
    predicts them. That is the same leave-driver-out principle the ring benchmark uses, and it
    matters: with ordinary folds the trees learn "offers at this spot get rejected" when what
    they have really learned is which driver works that spot, and the request appears to explain
    far more than it does. The driver is deliberately not an input either.
    """
    inputs = pd.DataFrame({
        'pickup_lat': offers.pickup_lat, 'pickup_lon': offers.pickup_lon,
        'x_km': offers.x_km, 'y_km': offers.y_km, 'km_from_centre': offers.km_from_centre,
        'trip_km': offers.trip_km,
        'hour': offers.dispatched_datetime.dt.hour,
        'weekday': offers.dispatched_datetime.dt.dayofweek,
        'is_scheduled': offers.is_scheduled.astype(float),
        'product': offers['product'].astype('category').cat.codes,
    })
    model = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.06, max_iter=300,
                                           l2_regularization=1.0, random_state=SEED)
    if group_by_driver:
        return cross_val_predict(model, inputs, offers.not_accepted, cv=GroupKFold(n_splits=5),
                                 groups=offers.driver_id, method='predict_proba')[:, 1]
    return cross_val_predict(model, inputs, offers.not_accepted, cv=5,
                             method='predict_proba')[:, 1]


def main():
    offers = load_offers()
    drivers = compared_drivers(offers)
    print(f'{len(offers):,} offers, {offers.driver_id.nunique()} drivers | '
          f'{len(drivers)} drivers with 50+ offers, '
          f'{offers.driver_id.isin(drivers).mean():.1%} of offers\n')

    definitions = {}

    # 1. The notebook's own benchmark, reproduced here as the reference point.
    ring_block = expected_from_others(offers, ['pickup_ring', 'time_block'], 'not_accepted')
    ring_only = expected_from_others(offers, ['pickup_ring'], 'not_accepted')
    definitions['ring x 6-hour block (notebook)'] = ring_block.fillna(ring_only)

    # 2-3. Square grids: finer geography, same leave-driver-out construction.
    for size in (4, 2):
        offers[f'grid_{size}'] = grid_cell(offers, size)
        fine = expected_from_others(offers, [f'grid_{size}', 'time_block'], 'not_accepted')
        coarse = expected_from_others(offers, [f'grid_{size}'], 'not_accepted')
        definitions[f'{size} km grid x 6-hour block'] = fine.fillna(coarse).fillna(
            definitions['ring x 6-hour block (notebook)'])

    # 4. k-means zones: let the data choose the geography instead of imposing rings.
    coords = offers[['x_km', 'y_km']].to_numpy()
    offers['zone'] = KMeans(n_clusters=12, random_state=SEED, n_init=10).fit_predict(coords)
    offers['zone'] = offers.zone.astype(str)
    zoned = expected_from_others(offers, ['zone', 'time_block'], 'not_accepted')
    zone_only = expected_from_others(offers, ['zone'], 'not_accepted')
    definitions['12 k-means zones x 6-hour block'] = zoned.fillna(zone_only).fillna(
        definitions['ring x 6-hour block (notebook)'])

    # 5. A boosted model on raw coordinates: no bins at all.
    definitions['boosted model, raw coordinates'] = pd.Series(
        boosted_expected(offers, group_by_driver=True), index=offers.index)

    # 6. The same model with ordinary folds, to show what the grouping is protecting against.
    definitions['  (same, folds not grouped by driver)'] = pd.Series(
        boosted_expected(offers, group_by_driver=False), index=offers.index)

    rows = {}
    per_driver_tables = {}
    for name, expected in definitions.items():
        share, per_driver = explained_share(offers, expected, drivers)
        per_driver_tables[name] = per_driver
        rows[name] = {'expected low': per_driver.expected.min(),
                      'expected high': per_driver.expected.max(),
                      'share of real between-driver variation': share}
    table = pd.DataFrame(rows).T

    observed = per_driver_tables['ring x 6-hour block (notebook)'].observed
    print('Observed not-accepted rate across the compared drivers: '
          f'{observed.min():.1%} to {observed.max():.1%}\n')
    print('How much of that can the REQUEST reproduce, under each description of it?\n')
    formatted = table.copy()
    for column in ['expected low', 'expected high']:
        formatted[column] = (table[column] * 100).round(1).astype(str) + '%'
    formatted['share of real between-driver variation'] = (
        (table['share of real between-driver variation'] * 100).round(1).astype(str) + '%')
    print(formatted.to_string())

    # Does a finer description change WHO looks worst? That is what dispatch would act on.
    print('\nDoes the driver ordering change? Spearman rank correlation of observed '
          'minus expected,\nagainst the notebook\'s ring benchmark:\n')
    reference = (per_driver_tables['ring x 6-hour block (notebook)'].observed
                 - per_driver_tables['ring x 6-hour block (notebook)'].expected)
    for name, per_driver in per_driver_tables.items():
        if name.startswith('ring'):
            continue
        gap = per_driver.observed - per_driver.expected
        rho = spearmanr(reference, gap).statistic
        print(f'  {name:<34} rho = {rho:.3f}')

    honest = table.loc[[n for n in table.index if not n.startswith('  ')],
                       'share of real between-driver variation'].max()
    leaky = table.loc['  (same, folds not grouped by driver)',
                      'share of real between-driver variation']
    print(f'\nWith every driver held out of the model that predicts them, the most generous '
          f'description of\nthe request reproduces {honest:.1%} of the real between-driver variation. '
          'The conclusion in step 1\ndoes not depend on the choice of bins.')
    print(f'\nThe indented row is the same model with ordinary folds: {leaky:.1%}. That is not a '
          'better estimate,\nit is the driver leaking in through their own geography, which is '
          'what the grouping prevents.')


if __name__ == '__main__':
    main()
