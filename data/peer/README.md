# PEER NGA-West2 records go here

The load generator reads every `.AT2` file in this folder when called with
`generate_loads(n, records="peer")`. The files are not committed to git (see `.gitignore`).

## How to download

1. Create a free account at the PEER Ground Motion Database (NGA-West2):
   https://ngawest2.berkeley.edu
2. Choose the search without a target spectrum and filter roughly as follows:
   - Magnitude: 5.5 to 7.8
   - Rupture distance (Rrup): 0 to 60 km
   - Vs30: 360 to 1500 m/s (stiff soil or rock; our model adds the soft soil itself)
   - Spectral ordinates / PGA: leave open
3. Aim for about 100 to 200 records from at least 20 different earthquakes. Many stations
   from one earthquake are fine, but more distinct earthquakes make the held-out test fairer.
4. Download the acceleration files (`.AT2`), unzip, and copy them into this folder.
   Also keep every `_SearchResults*.csv` file here: the magnitudes in them are used to
   stratify the train/val/test split (`split_by_event`).

Vertical components (`UP`, `DWN`, `-V` in the name) are skipped automatically.

## Check

```
.venv\Scripts\python.exe -c "from loads.earthquakes import load_peer; r = load_peer(); print(len(r), 'records from', len({x.event for x in r}), 'earthquakes')"
```
