"""Validated NHL schedules and compact MoneyPuck xG; never substitutes sample data."""
import csv
import datetime as dt
import hashlib
import io
import json
import math
import os
import pathlib
import time
import urllib.error
import urllib.request

API = 'https://api-web.nhle.com/v1'
MONEYPUCK = 'https://moneypuck.com/moneypuck/playerData/careers/gameByGame/all_teams.csv'
UTC = dt.timezone.utc
FINAL = {'OFF', 'FINAL'}
# NHL published season lengths; do not silently assume a future season's format.
SEASON_GAMES = {2023: 82, 2024: 82, 2025: 82, 2026: 84}


class DataError(RuntimeError):
    pass


def date_string(value):
    if dt.date.fromisoformat(value).isoformat() != value:
        raise ValueError('Use an explicit YYYY-MM-DD NHL game date')
    return value


def season(value):
    day = dt.date.fromisoformat(date_string(value))
    return day.year if day.month >= 7 else day.year - 1


def team(value):
    # Utah franchise history is kept together; ARI is intentionally separate.
    return {'UTH': 'UTA', 'LA': 'LAK', 'SJ': 'SJS', 'TB': 'TBL', 'NJ': 'NJD'}.get(value, value)


def atomic_json(path, payload):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(payload, allow_nan=False))
    os.replace(temp, path)


class Client:
    def __init__(self, cache='data/cache/v3', timeout=20, attempts=3,
                 opener=urllib.request.urlopen, sleeper=time.sleep, clock=time.time):
        self.cache = pathlib.Path(cache)
        self.timeout, self.attempts = timeout, attempts
        self.opener, self.sleeper, self.clock = opener, sleeper, clock
        self.events = []

    def download(self, url):
        for attempt in range(self.attempts):
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'NHL-MOD/3.0', 'Accept': '*/*'})
                with self.opener(req, timeout=self.timeout) as response:
                    return response.read()
            except urllib.error.HTTPError as exc:
                if exc.code not in {408, 429, 500, 502, 503, 504}:
                    raise DataError(f'HTTP {exc.code}: {url}') from exc
                error = exc
            except (OSError, TimeoutError) as exc:
                error = exc
            if attempt + 1 < self.attempts:
                self.sleeper(min(2 ** attempt, 8))
        raise DataError(f'Failed after {self.attempts} attempts: {url}: {error}')

    def cached(self, key, loader, validator, ttl):
        path = self.cache / (key + '.json')
        if path.exists():
            try:
                entry = json.loads(path.read_text())
                validator(entry['payload'])
                if ttl is None or 0 <= self.clock() - entry['fetched_at'] < ttl:
                    self.events.append({'key': key, 'cache': 'hit', 'fetched_at': entry['fetched_at']})
                    return entry['payload']
            except (ValueError, KeyError, TypeError, DataError):
                pass
        payload = loader()
        validator(payload)  # Invalid/partial responses never replace a valid cache.
        atomic_json(path, {'fetched_at': self.clock(), 'payload': payload})
        self.events.append({'key': key, 'cache': 'refresh', 'fetched_at': self.clock()})
        return payload

    def get(self, endpoint, validator, ttl=3600):
        key = hashlib.sha256(endpoint.encode()).hexdigest()
        def loader():
            try:
                return json.loads(self.download(API + endpoint))
            except (ValueError, UnicodeError) as exc:
                raise DataError(f'Invalid JSON: {endpoint}') from exc
        return self.cached(key, loader, validator, ttl)


def normalize_game(g):
    try:
        date = date_string(g['gameDate'])
        start = dt.datetime.fromisoformat(g['startTimeUTC'].replace('Z', '+00:00'))
        if start.tzinfo is None:
            raise ValueError('No start timezone')
        home, away = team(g['homeTeam']['abbrev']), team(g['awayTeam']['abbrev'])
        if not home or not away or home == away:
            raise ValueError('Invalid teams')
        row = dict(id=str(g['id']), date=date, start_utc=g['startTimeUTC'],
                   home=home, away=away, state=g['gameState'], game_type=g['gameType'])
        if row['state'] in FINAL:
            for key, side in [('hg', 'homeTeam'), ('ag', 'awayTeam')]:
                score = g[side]['score']
                if isinstance(score, bool) or int(score) != score or score < 0:
                    raise ValueError('Invalid final score')
                row[key] = int(score)
            if row['hg'] == row['ag']:
                raise ValueError('Final NHL game cannot be tied')
        return row
    except (ValueError, TypeError, KeyError) as exc:
        raise DataError(f'Incomplete game {g.get("id")}: {exc}') from exc


def validate_club(payload):
    if not isinstance(payload, dict) or not isinstance(payload.get('games'), list) or not payload['games']:
        raise DataError('Missing or empty club schedule')
    ids = set()
    for raw in payload['games']:
        if raw.get('gameType') != 2:
            continue
        row = normalize_game(raw)
        if row['id'] in ids:
            raise DataError('Duplicate club game')
        ids.add(row['id'])
    if not ids:
        raise DataError('No regular-season club games')


def collect_season(client, year, requested_date):
    """Cross-check all 32 club schedules, not just the teams that answered."""
    clubs = 'ANA BOS BUF CAR CBJ CGY CHI COL DAL DET EDM FLA LAK MIN MTL NJD NSH NYI NYR OTT PHI PIT SEA SJS STL TBL TOR VAN VGK WPG WSH'.split()
    clubs.append('ARI' if year <= 2023 else 'UTA')
    games, appearances = {}, {}
    closed = year < season(requested_date)
    if year not in SEASON_GAMES:
        raise DataError(f'Add the published game-count expectation for season {year} before running')
    def validate(payload):
        validate_club(payload)
        regular = [g for g in payload['games'] if g.get('gameType') == 2]
        if len(regular) != SEASON_GAMES[year]:
            raise DataError(f'Incomplete {year} club schedule: {len(regular)}/{SEASON_GAMES[year]}')
        if closed and any(g['gameState'] not in FINAL for g in regular):
            raise DataError(f'Incomplete final results for historical season {year}')
    for club in clubs:
        endpoint = f'/club-schedule-season/{club}/{year}{year+1}'
        payload = client.get(endpoint, validate, ttl=None if closed else 3600)
        for raw in payload['games']:
            if raw.get('gameType') != 2:
                continue
            row = normalize_game(raw)
            if season(row['date']) != year or club not in (row['home'], row['away']):
                raise DataError(f'Season/team mismatch: {club}, {row["id"]}')
            gid = row['id']
            if gid in games and games[gid] != row:
                raise DataError(f'Conflicting club records for {gid}; refresh after score updates')
            games[gid] = row
            appearances.setdefault(gid, set()).add(club)
    if len(games) != 32 * SEASON_GAMES[year] // 2 or any(appearances[g] != {r['home'], r['away']} for g, r in games.items()):
        raise DataError('Incomplete league schedule: games must appear in both club feeds')
    if closed:
        if year in {2023, 2024, 2025} and len(games) != 1312:
            raise DataError(f'Incomplete historical season {year}: {len(games)}/1312 games')
        if any(g['state'] not in FINAL for g in games.values()):
            raise DataError(f'Historical season {year} contains unfinished games')
    overdue = [g['id'] for g in games.values() if g['date'] < requested_date and g['state'] not in FINAL]
    if overdue:
        raise DataError(f'Unresolved prior-date games (possibly postponed): {overdue[:10]}')
    return sorted(games.values(), key=lambda r: (r['date'], r['id']))


def schedule(client, requested_date):
    date_string(requested_date)
    def validate(payload):
        days = payload.get('gameWeek')
        if not isinstance(days, list):
            raise DataError('Missing gameWeek')
        matching = [d for d in days if d.get('date') == requested_date]
        if len(matching) != 1:
            raise DataError('Schedule omitted requested NHL date')
        day = matching[0]
        if not isinstance(day.get('games'), list) or day.get('numberOfGames') != len(day['games']):
            raise DataError('Schedule game count mismatch')
        ids = set()
        for g in day['games']:
            # Weekly API carries gameDate on the parent day; club feeds carry it per game.
            row = normalize_game({**g, 'gameDate': g.get('gameDate', day['date'])})
            if row['date'] != requested_date or row['id'] in ids:
                raise DataError('Schedule date mismatch or duplicate game')
            ids.add(row['id'])
    payload = client.get('/schedule/' + requested_date, validate, ttl=300)
    return [normalize_game({**g, 'gameDate': g.get('gameDate', d['date'])}) for d in payload['gameWeek'] if d['date'] == requested_date
            for g in d['games'] if g['gameType'] == 2]


def history_before(games, requested_date):
    date_string(requested_date)
    return sorted((g for g in games if g['date'] < requested_date and g['state'] in FINAL),
                  key=lambda r: (r['date'], r['id']))


def compact_xg(stream, first_season=2023):
    rows = []
    reader = csv.DictReader(stream)
    required = {'season', 'situation', 'gameId', 'team', 'gameDate', 'xGoalsFor', 'xGoalsAgainst'}
    if not required.issubset(reader.fieldnames or []):
        raise DataError('MoneyPuck columns missing')
    seen = set()
    for r in reader:
        if int(r['season']) < first_season or r['situation'] != 'all':
            continue
        # NHL regular-season game IDs contain game type 02.
        if str(r['gameId'])[4:6] != '02':
            continue
        date = dt.datetime.strptime(r['gameDate'], '%Y%m%d').date().isoformat()
        xf, xa = float(r['xGoalsFor']), float(r['xGoalsAgainst'])
        key = (str(r['gameId']), team(r['team']))
        if key in seen or any(not math.isfinite(v) or v < 0 for v in (xf, xa)):
            raise DataError('Duplicate or invalid MoneyPuck xG row')
        seen.add(key)
        rows.append(dict(id=key[0], team=key[1], date=date, xf=xf, xa=xa))
    if not rows:
        raise DataError('No usable MoneyPuck rows')
    return rows


def load_xg(client, local_csv=None):
    def loader():
        if local_csv:
            with open(local_csv, newline='') as f:
                return compact_xg(f)
        return compact_xg(io.StringIO(client.download(MONEYPUCK).decode('utf-8-sig')))
    def validate(rows):
        if not isinstance(rows, list) or not rows:
            raise DataError('Empty xG cache')
        for r in rows:
            if not {'id', 'team', 'date', 'xf', 'xa'}.issubset(r):
                raise DataError('Malformed xG cache')
            date_string(r['date'])
            if any(not math.isfinite(r[k]) or r[k] < 0 for k in ('xf', 'xa')):
                raise DataError('Invalid cached xG')
    rows = client.cached('moneypuck-compact-2023-v1', loader, validate, ttl=86400)
    return {(r['id'], r['team']): r for r in rows}
