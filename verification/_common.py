"""Shared loading and preparation, identical to notebooks/Odysse_Ride_Analysis.ipynb cells 3-17.

Kept separate so the two check scripts start from exactly the same table the notebook uses.
Run either check from the repository root.
"""
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42
MIN_OFFERS = 50
MIN_CELL = 20
CENTRE = (51.5074, -0.1278)          # Charing Cross


def km_between(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    half_chord = (np.sin((lat2 - lat1) / 2) ** 2
                  + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0088 * np.arcsin(np.sqrt(half_chord))


def load_offers():
    workbook = 'interview-take-home-dataset.xlsx'
    paths = [Path('data/raw') / workbook, Path('../data/raw') / workbook, Path(workbook)]
    data_file = next(p for p in paths if p.exists())
    offers = pd.read_excel(data_file, sheet_name='Trips')
    offers = offers.sort_values(['driver_id', 'dispatched_datetime'], kind='stable')
    offers = offers.reset_index(drop=True)

    offers['turned_down'] = offers.status.eq('Driver rejected').astype(int)
    offers['no_response'] = offers.status.eq('Driver did not respond').astype(int)
    offers['not_accepted'] = offers.turned_down + offers.no_response
    offers['accepted'] = offers.order_accepted_timestamp.notna().astype(int)

    pickup = offers.pickup_lat_long.str.split(',', expand=True).astype(float)
    dropoff = offers.dropoff_latlong.str.split(',', expand=True).astype(float)
    offers['pickup_lat'], offers['pickup_lon'] = pickup[0], pickup[1]
    km_from_centre = km_between(pickup[0], pickup[1], *CENTRE)
    offers['km_from_centre'] = km_from_centre
    offers['trip_km'] = km_between(pickup[0], pickup[1], dropoff[0], dropoff[1])
    offers['long_trip'] = offers.trip_km >= 8
    offers['log_km'] = np.log1p(offers.trip_km)

    offers['pickup_ring'] = pd.cut(km_from_centre, [0, 3, 8, 15, np.inf], right=False,
                                   labels=['Central', 'Inner', 'Outer', 'Far']).astype(str)
    offers['x_km'] = (pickup[1] - CENTRE[1]) * np.cos(np.radians(CENTRE[0])) * 111.32
    offers['y_km'] = (pickup[0] - CENTRE[0]) * 110.574

    dispatch_hour = offers.dispatched_datetime.dt.hour
    offers['time_block'] = pd.cut(dispatch_hour, [0, 6, 12, 18, 24], right=False,
                                  labels=['00–06', '06–12', '12–18', '18–24']).astype(str)
    offers['hour_bin'] = (dispatch_hour // 3).astype(str)
    offers['day'] = offers.dispatched_datetime.dt.day
    offers['day_set'] = np.where(offers.day.isin([17, 19, 21]), 'A', 'B')
    offers['product'] = offers.category.str.replace(r'\s*\[.*$', '', regex=True).str.strip()
    return offers


def expected_from_others(offers, cell_columns, outcome, min_cell=MIN_CELL):
    """For each offer, how often OTHER drivers' offers in the same cell had this outcome."""
    by_cell = offers.groupby(cell_columns)[outcome]
    cell_totals = by_cell.agg(cell_outcomes='sum', cell_offers='count')
    by_cell_and_driver = offers.groupby(cell_columns + ['driver_id'])[outcome]
    own_totals = by_cell_and_driver.agg(own_outcomes='sum', own_offers='count')

    beside = offers[cell_columns + ['driver_id']].join(cell_totals, on=cell_columns)
    beside = beside.join(own_totals, on=cell_columns + ['driver_id'])
    others_outcomes = beside.cell_outcomes - beside.own_outcomes
    others_offers = beside.cell_offers - beside.own_offers
    expected = others_outcomes / others_offers
    expected[others_offers < min_cell] = np.nan
    return expected


def compared_drivers(offers):
    """The drivers with enough offers to be compared with the others."""
    counts = offers.groupby('driver_id').size()
    return sorted(counts[counts >= MIN_OFFERS].index)


def explained_share(offers, expected, drivers):
    """Share of the real between-driver variation reproduced by this set of expected rates.

    Denominator is the observed between-driver variance with average sampling noise removed,
    so the comparison is against variation that is actually there rather than against noise.
    """
    frame = pd.DataFrame({'driver_id': offers.driver_id,
                          'observed': offers.not_accepted,
                          'expected': expected})
    frame = frame[frame.driver_id.isin(drivers)]
    per_driver = frame.groupby('driver_id').agg(n=('observed', 'size'),
                                                observed=('observed', 'mean'),
                                                expected=('expected', 'mean'))
    noise = (per_driver.observed * (1 - per_driver.observed) / per_driver.n).mean()
    real = per_driver.observed.var() - noise
    return per_driver.expected.var() / real, per_driver
