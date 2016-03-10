import re
import json
import requests
from bs4 import BeautifulSoup
from docopt import docopt
import pandas as pd


URL = 'http://www.numberfire.com/nba/fantasy/full-fantasy-basketball-projections'


def get_data():
    r = requests.get(URL)
    r.raise_for_status()

    script = BeautifulSoup(r.text, 'lxml').find_all('script')[3]
    return json.loads(re.search(r'\tvar NF_DATA = ([^;]*);', script.text).groups(1)[0])


def get_team_analytics(raw_data):
    team_analytics = pd.DataFrame(raw_data['team_analytics']).transpose()
    team_analytics.index = team_analytics.index.astype(int)
    team_analytics = team_analytics.sort_index()

    float_cols = [
        'champs',
        'drtg',
        'drtg_cdf',
        'nerd',
        'nerd_cdf',
        'ortg',
        'ortg_cdf',
        'pace',
        'pace_cdf',
        'playoffs',
        'proj_l',
        'proj_w',
        'ptdiff',
        'sd_differential',
    ]
    int_cols = [
        'losses',
        'nba_team_id',
        'ovr_rank',
        'season',
        'wins',
    ]

    team_analytics[float_cols] = team_analytics[float_cols].astype(float)
    team_analytics[int_cols] = team_analytics[int_cols].astype(int)
    team_analytics['date'] = pd.to_datetime(team_analytics['date'])

    teams = pd.DataFrame(raw_data['teams']).transpose()
    teams.index = teams.index.astype(int)
    teams = teams.sort_index()

    int_cols = [
        'espn_id',
        'id',
        'is_active',
        'league_id',
    ]
    teams[int_cols] = teams[int_cols].astype(int)

    return pd.concat([team_analytics, teams], axis='columns')


def get_player_projections(raw_data):
    players = pd.DataFrame(raw_data['players']).transpose()

    int_cols = [
        'age',
        'category_id',
        'depth_rank',
        'espn_id',
        'experience',
        'height',
        'id',
        'league_id',
        'number',
        'position_group',
        'race',
        'role_id',
        'team_id',
        'weight',
        'yahoo_id',
    ]
    players.loc[players['height'] == '', 'height'] = '0'
    compound_height_ixs = pd.Series(['-' in height for height in players['height']], index=players.index)
    compound_heights = players.loc[compound_height_ixs, 'height']
    players.loc[compound_height_ixs, 'height'] = (
        compound_heights.str.split('-', expand=True).astype(int) * [12, 1]
    ).sum(axis=1).astype(str)
    players.loc[players['league_id'] == '', 'league_id'] = '0'
    players[int_cols] = players[int_cols].astype(int)

    players.loc[players['dob'] == '0000-00-00', 'dob'] = '1900-01-01'
    players['dob'] = pd.to_datetime(players['dob'])

    bool_cols = [
        'is_active',
        'is_rookie',
    ]
    players[bool_cols] = players[bool_cols].astype(int).astype(bool)

    players = players.set_index('id').sort_index()

    analytics = pd.DataFrame(raw_data['daily_projections'])

    float_cols = [
        'ast',
        'blk',
        'draft_kings_fp',
        'draft_kings_ratio',
        'draft_street_daily_fp',
        'draft_street_daily_ratio',
        'draftday_fp',
        'draftday_ratio',
        'draftster_fp',
        'draftster_ratio',
        'dreb',
        'drtg',
        'efg',
        'fanduel_fp',
        'fanduel_ratio',
        'fantasy_aces_fp',
        'fantasy_aces_ratio',
        'fantasy_feud_fp',
        'fantasy_feud_ratio',
        'fantasy_score_fp',
        'fantasy_score_ratio',
        'fanthrowdown_fp',
        'fanthrowdown_ratio',
        'fga',
        'fgm',
        'fta',
        'ftm',
        'game_play_probability',
        'game_start',
        'minutes',
        'nerd',
        'oreb',
        'ortg',
        'p3a',
        'p3m',
        'pf',
        'pts',
        'star_street_fp',
        'star_street_ratio',
        'stl',
        'tov',
        'treb',
        'ts',
        'usg',
    ]
    analytics[float_cols] = analytics[float_cols].astype(float)

    pct_cols = [col for col in analytics.columns if col.endswith('_pct')]
    analytics[pct_cols] = analytics[pct_cols].astype(float) / 100

    int_cols = [
        'draft_kings_salary',
        'draft_street_daily_salary',
        'draftday_salary',
        'draftster_salary',
        'fanduel_salary',
        'fantasy_aces_salary',
        'fantasy_feud_salary',
        'fantasy_score_salary',
        'fanthrowdown_salary',
        'nba_game_id',
        'nba_player_id',
        'nba_team_id',
        'season',
        'star_street_salary',
    ]
    analytics[int_cols] = analytics[int_cols].astype(int)

    analytics['date'] = pd.to_datetime(analytics['date'])

    analytics = analytics.set_index('nba_player_id').sort_index()
    analytics.index.name = 'id'

    return pd.concat([players, analytics], axis='columns')


def main():
    raw_data = get_data()
    get_team_analytics(raw_data).to_csv('team_analytics.csv')
    get_player_projections(raw_data).to_csv('daily_projections.csv')


if __name__ == '__main__':
    main()
