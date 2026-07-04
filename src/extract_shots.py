# Extraction script for a local clone of https://github.com/statsbomb/open-data
# (git clone --depth 1 --filter=blob:none, then checkout the needed matches/events paths)
"""
Extracts every "Shot" event from StatsBomb open-data event files across
multiple competitions and builds a flat shots_raw.csv with engineered features,
including a simple assist-type feature derived from the key pass event,
plus player and team names for downstream aggregate analysis.

Each shot is tagged with competition_name AND season_name, since StatsBomb
reuses the same competition_name ("FIFA World Cup") for both the 2018 and
2022 tournaments - without the season field the two would silently merge
under one filter.
"""
import json, os, math, csv

EVENTS_DIR = "/tmp/sbtest/data/events"
COMPETITIONS = [
    (43, 3),      # FIFA World Cup 2018
    (72, 30),     # Women's World Cup 2019
    (55, 43),     # UEFA Euro 2020
    (43, 106),    # FIFA World Cup 2022
    (53, 106),    # UEFA Women's Euro 2022
    (223, 282),   # Copa America 2024
    (1267, 107),  # Africa Cup of Nations 2023
    (11, 27),     # La Liga 2015/2016 (full season, club football)
]
MATCHES_DIR = "/tmp/sbtest/data/matches"
OUT_CSV = "/sessions/peaceful-exciting-albattani/mnt/outputs/xg_model_v2/data/shots_raw.csv"

match_info = {}
for comp_id, season_id in COMPETITIONS:
    path = f"{MATCHES_DIR}/{comp_id}/{season_id}.json"
    for m in json.load(open(path)):
        match_info[m["match_id"]] = m
print(f"Loaded metadata for {len(match_info)} matches across {len(COMPETITIONS)} competitions")

GOAL_X, GOAL_Y = 120.0, 40.0
POST1 = (120.0, 36.0)
POST2 = (120.0, 44.0)

def angle_to_goal(x, y):
    a = math.dist((x,y), POST1)
    b = math.dist((x,y), POST2)
    c = 8.0
    try:
        cos_angle = (a**2 + b**2 - c**2) / (2*a*b)
        cos_angle = max(-1,min(1,cos_angle))
        return math.degrees(math.acos(cos_angle))
    except ZeroDivisionError:
        return 0.0

rows = []
parse_failures = []
for match_id, minfo in match_info.items():
    path = os.path.join(EVENTS_DIR, f"{match_id}.json")
    if not os.path.exists(path):
        parse_failures.append((match_id, "missing file"))
        continue
    try:
        events = json.load(open(path))
    except Exception as e:
        parse_failures.append((match_id, str(e)))
        continue

    by_id = {e["id"]: e for e in events}
    competition_name = minfo.get("competition", {}).get("competition_name", "")
    competition_stage = minfo.get("competition_stage", {}).get("name", "")
    season_name = minfo.get("season", {}).get("season_name", "")
    gender = minfo.get("home_team", {}).get("home_team_gender", "")

    for e in events:
        if e.get('type',{}).get('name') != 'Shot':
            continue
        shot = e.get('shot', {})
        loc = e.get('location')
        if not loc or len(loc) < 2:
            continue
        x, y = loc[0], loc[1]
        outcome = shot.get('outcome', {}).get('name', '')
        is_goal = 1 if outcome == 'Goal' else 0
        body_part = shot.get('body_part', {}).get('name', '')
        technique = shot.get('technique', {}).get('name', '')
        shot_type = shot.get('type', {}).get('name', '')
        under_pressure = bool(e.get('under_pressure', False))
        first_time = bool(shot.get('first_time', False))
        one_on_one = bool(shot.get('one_on_one', False))
        aerial_won = bool(shot.get('aerial_won', False))
        statsbomb_xg = shot.get('statsbomb_xg', None)
        play_pattern = e.get('play_pattern', {}).get('name', '')
        freeze = shot.get('freeze_frame', []) or []
        n_opponents_close = 0
        gk_positioned = 0
        for ff in freeze:
            if ff.get('teammate') is False:
                floc = ff.get('location', [None,None])
                if floc[0] is not None:
                    d = math.dist((x,y), (floc[0], floc[1]))
                    if d < 3.0:
                        n_opponents_close += 1
                if ff.get('position', {}).get('name') == 'Goalkeeper':
                    gk_positioned = 1

        assist_type = "None"
        key_pass_id = shot.get("key_pass_id")
        if key_pass_id and key_pass_id in by_id:
            kp = by_id[key_pass_id].get("pass", {})
            if kp.get("cross"):
                assist_type = "Cross"
            elif kp.get("cut_back") or kp.get("technique", {}).get("name") == "Cut Back":
                assist_type = "CutBack"
            elif kp.get("through_ball"):
                assist_type = "ThroughBall"
            elif kp.get("type", {}).get("name") == "Corner":
                assist_type = "Corner"
            elif kp.get("type", {}).get("name") == "Free Kick":
                assist_type = "FreeKickPass"
            else:
                assist_type = "OtherPass"

        distance = math.dist((x,y), (GOAL_X, GOAL_Y))
        angle = angle_to_goal(x,y)
        player_name = e.get('player', {}).get('name', '')
        team_name = e.get('team', {}).get('name', '')
        rows.append({
            'match_id': match_id,
            'competition_name': competition_name,
            'competition_stage': competition_stage,
            'season_name': season_name,
            'gender': gender,
            'player_name': player_name,
            'team_name': team_name,
            'x': x, 'y': y,
            'distance': round(distance,3),
            'angle_deg': round(angle,3),
            'body_part': body_part,
            'technique': technique,
            'shot_type': shot_type,
            'play_pattern': play_pattern,
            'assist_type': assist_type,
            'under_pressure': int(under_pressure),
            'first_time': int(first_time),
            'one_on_one': int(one_on_one),
            'aerial_won': int(aerial_won),
            'n_opponents_close': n_opponents_close,
            'gk_positioned': gk_positioned,
            'statsbomb_xg': statsbomb_xg,
            'is_goal': is_goal,
        })

print("total shots extracted:", len(rows))
if parse_failures:
    print(f"{len(parse_failures)} matches failed to parse:", parse_failures[:5])
if rows:
    keys = list(rows[0].keys())
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("saved to", OUT_CSV)
