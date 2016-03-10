import player as data_pl
import team as data_te
import game as data_ga
import utils as anal_ut
import multiprocessing as mp
import numpy as np
import scipy
import scipy.stats as scst
import datetime
from numpy import linalg
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd
import time
from sklearn import cluster
from sklearn import linear_model
from collections import Counter
import math
import statsmodels.api as sm
import urllib2
import re
import os
import unicodedata
import pulp_lineup_optimization as lopt
from scipy import special
import cvxopt
import pulp

CURRENT_SEASON = '2015-16'

DFS_SITES = ['FD','DK','DP','YH','DD','FA','FF']

MASCOT2ABBREV = {
    'Hawks' : 'ATL',
    'Suns' : 'PHX',
    'Celtics' : 'BOS',
    '76ers' : 'PHI',
    'Hornets' : 'CHA',
    'Knicks' : 'NYK',
    'Bulls' : 'CHI',
    'Cavaliers' : 'CLE',
    'Nuggets' : 'DEN',
    'Pistons' : 'DET',
    'Pacers' : 'IND',
    'Kings' : 'SAC',
    'Lakers' : 'LAL',
    'Trail Blazers': 'POR',
    'Grizzlies' : 'MEM',
    'Timberwolves' : 'MIN',
    'Pelicans' : 'NOP',
    'Bucks' : 'MIL',
}

PlayerList = data_pl.PlayerList(season=CURRENT_SEASON, only_current=1).info()
PlayerList = PlayerList.set_index('PERSON_ID')

COMMA_NAME_CORRECTIONS = {
    'Smith, Joshua': 'Smith, Josh',
    'Neto, Raulzinho': 'Neto, Raul',
    'Harkless, Moe': 'Harkless, Maurice',
    'Amundson, Louis': 'Amundson, Lou',
    'Barea, J.J.': 'Barea, Jose Juan',
    'Barea, Juan Jose': 'Barea, Jose Juan',
    'Juan Barea, Jose': 'Barea, Jose Juan',
    'Jones, Bryce' : 'Dejean-Jones, Bryce',
    'Beal, Brad': 'Beal, Bradley',
    'Brooks, Aar**n': 'Brooks, Aaron',
    'Calder**n, Jos**': 'Calderon, Jose',
    'Fel**cio, Cristiano': 'Felicio, Cristiano',
    'Gin**bili, Manu': 'Ginobili, Manu',
    'Hairston, P.J.': 'Hairston, PJ',
    'Hayes, Chuck': 'Hayes, Charles',
    'Hickson, J.J.': 'Hickson, JJ',
    'Hilario, Nene': 'Nene',
    'Hunter, R.J.': 'Hunter, RJ',
    'Robinson III, Glenn': 'Robinson, Glenn',
    'Oubre Jr., Kelly': 'Oubre, Kelly',
    'Devyn Marble, Roy': 'Marble, Devyn',
    'Mart**n, Kevin': 'Martin, Kevin',
    'McCollum, C.J.': 'McCollum, CJ',
    'McConnell, T.J.': 'McConnell, TJ',
    'McDaniels, K.J.': 'McDaniels, KJ',
    'Miles, C.J.': 'Miles, CJ',
    'Miller, Andr**': 'Miller, Andre',
    'Mills, Patrick': 'Mills, Patty',
    'Mbah a Moute, Luc Richard': 'Mbah a Moute, Luc',
    'Richard Mbah a Moute, Luc': 'Mbah a Moute, Luc',
    'Pressey, Phil (Flip)': 'Pressey, Phil',
    'Redick, J.J.': 'Redick, JJ',
    'Schroder, Denis': 'Schroder, Dennis',
    'Smith, Ishmael': 'Smith, Ish',
    'Tucker, P.J.': 'Tucker, PJ',
    'Warren, T.J.': 'Warren, TJ',
    'Watson, C.J.': 'Watson, CJ',
    'Wilcox, C.J.': 'Wilcox, CJ',
    'Williams, Louis': 'Williams, Lou',
    'Young, Joseph': 'Young, Joe',
    'Porter Jr., Otto': 'Porter, Otto',
    'O\'Bryant III, Johnny': 'O\'Bryant, Johnny',
    'Kaminsky III, Frank': 'Kaminsky, Frank',
    'Michael McAdoo, James': 'McAdoo, James Michael',
    'Da Silva Felicio, Cristiano': 'Felicio, Cristiano',
    'Nance, Larry': 'Nance Jr., Larry',
}


def get_pid(lcf):
    '''Take a LAST_COMMA_FIRST and return its Player_ID'''
    try:
        return PlayerList[PlayerList.DISPLAY_LAST_COMMA_FIRST == lcf].iloc[0].name
    except IndexError:
        print lcf
        return np.nan

def get_lcf(pid):
    '''Take a Player_ID and return its LAST_COMMA_FIRST'''
    try:
        return PlayerList.loc[pid].DISPLAY_LAST_COMMA_FIRST
    except IndexError:
        print lcf
        return np.nan

def params2rates(params):
    '''Take beta and gamma distributions and convert them to rates.'''
    dic = {'alpha': {}, 'beta': {}}
    for key in params:
        param, st = key.split('_')
        val = params[key].tolist()[0]
        dic[param][st] = val
    tmp = pd.DataFrame(dic)
    
    lm = []
    for st in ('AST','BLK','OREB','DREB','STL','TOV','FG2A','FG3A','FTA'):
        lm.append(tmp.alpha.loc[st] / tmp.beta.loc[st])
    
    p = []
    for st in ('FG2%','FG3%','FT%'):
        p.append(tmp.alpha.loc[st] / (tmp.alpha.loc[st] + tmp.beta.loc[st]))
    
    lm.extend(p)
    return pd.Series(lm, index=('AST','BLK','OREB','DREB','STL','TOV','FG2A','FG3A','FTA','FG2%','FG3%','FT%'))

def rates_mins2simple_projs(rates, mins):
    '''
    Take rates from the gamma and beta distributions with mins.
    '''
    projs = mins * rates[['AST','BLK','FG2A','FG3A','FTA','OREB','DREB','STL','TOV']]
    projs['REB'] = projs.OREB + projs.DREB
    projs['FGM'] = rates['FG2%']*projs['FG2A'] + rates['FG3%']*projs['FG3A']
    projs['FG3M'] = rates['FG3%']*projs['FG3A']
    projs['FGA'] = projs['FG2A'] + projs['FG3A']
    projs['FG2M'] = rates['FG2%']*projs['FG2A']
    projs['FTM'] = projs['FTA'] * rates['FT%']
    projs['MIN'] = mins
    projs['PTS'] = 2*projs['FGM'] + projs['FG3M'] + projs['FTM']
    
    fantasy = results2fp(projs)
    return fantasy


def rates_mins2projs(rates, mins, adj=None):
    '''
    Take rates from the gamma and beta distributions with mins.
    '''
    projs = mins * rates[['AST','BLK','FG2A','FG3A','FTA','OREB','DREB','STL','TOV']]
    projs['REB'] = projs.OREB + projs.DREB
    projs['FGM'] = rates['FG2%']*projs['FG2A'] + rates['FG3%']*projs['FG3A']
    projs['FG3M'] = rates['FG3%']*projs['FG3A']
    projs['FGA'] = projs['FG2A'] + projs['FG3A']
    del projs['FG2A']
    projs['FTM'] = projs['FTA'] * rates['FT%']
    projs['MIN'] = mins
    
    # ('FGM','FGA','FG3M','FG3A','FTM','FTA','AST','OREB','DREB','BLK','STL','TOV')
    if adj is not None:
        for st, coef in adj.iteritems():
            tmp_proj = coef[0] + coef[1]*projs[st]
            projs[st] = tmp_proj if tmp_proj > 0 else projs[st]
    projs['REB'] = projs.OREB + projs.DREB
    projs['PTS'] = 2*projs['FGM'] + projs['FG3M'] + projs['FTM']
    
    p2_2 = 0.0
    p3_2 = 0.0
    
    if mins > 0:
        p = 0.0
        for pt in range(15):
            for p3 in range(15):
                p += scst.poisson.sf((10-pt-3*p3)/2-1, np.maximum(1e-6 , projs['FGM'] - projs['FG3M'])) \
                    * scst.poisson.pmf(pt, np.maximum(1e-6 , projs['FTM'])) \
                    * scst.poisson.pmf(p3, np.maximum(1e-6 , projs['FG3M']))
        probs = {'PTS': p,
                 'REB': scst.poisson.sf(9, np.maximum(1e-6 , projs['OREB']+projs['DREB'])),
                 'AST': scst.poisson.sf(9, np.maximum(1e-6 , projs['AST'])),
                 'STL': scst.poisson.sf(9, np.maximum(1e-6 , projs['STL'])),
                 'BLK': scst.poisson.sf(9, np.maximum(1e-6 , projs['BLK']))
                }
    
        sts = probs.keys()
        for k, st in enumerate(sts):
            v = probs[st]
            for k2, st2 in enumerate(sts[k+1:]):
                v2 = probs[st2]
                p2_2 += v*v2
                for st3 in sts[k+k2+2:]:
                    v3 = probs[st3]
                    p3_2 += v*v2*v3
    
    projs['DD'] = p2_2 - p3_2
    projs['TD'] = p3_2
    
    fantasy = results2fp(projs)
    return fantasy

def results2fp(res):
    """
    NBA:
    Take a DataFrame with stats and add to it columns with fantasy scores from each site.
    """
    tmp = res.copy()
    fd = tmp.PTS + 1.2*tmp.REB + 1.5*tmp.AST + 2*tmp.BLK + 2*tmp.STL - tmp.TOV
    ff = tmp.PTS + 1.25*tmp.REB + 1.5*tmp.AST + 2*tmp.BLK + 2*tmp.STL - tmp.TOV
    if 'DD' in tmp.keys() and 'TD' in tmp.keys():
        dd = tmp.DD
        td = tmp.TD
    else:
        dd_td = np.array(tmp.PTS>=10) + np.array(tmp.REB>=10) + np.array(tmp.AST>=10) + np.array(tmp.BLK>=10) + np.array(tmp.STL>=10)
        dd = np.array(dd_td==2)
        td = np.array(dd_td>2)
    dk = tmp.PTS + 1.25*tmp.REB + 1.5*tmp.AST + 2*tmp.BLK + 2*tmp.STL - .5*tmp.TOV + .5*tmp.FG3M + 1.5*dd + 3.0*td
    dp = tmp.PTS + 1.25*tmp.REB + 1.5*tmp.AST + 2*tmp.BLK + 2*tmp.STL - tmp.TOV + 1.5*dd + 3.0*td
    yh = tmp.PTS + 1.2*tmp.REB + 1.5*tmp.AST + 2*tmp.BLK + 2*tmp.STL - tmp.TOV + .5*tmp.FG3M
    dd = tmp.PTS + 1.25*tmp.REB + 1.5*tmp.AST + 2*tmp.BLK + 2*tmp.STL - .5*tmp.TOV + tmp.FG3M - .25*(tmp.FGA-tmp.FGM) - .25*(tmp.FTA-tmp.FTM) + 2*dd + 2*td
    fa = tmp.PTS + 1.25*tmp.REB + 1.5*tmp.AST + 2*tmp.BLK + 2*tmp.STL - tmp.TOV
    tmp['FD'] = fd
    tmp['DK'] = dk
    tmp['DP'] = dp
    tmp['YH'] = yh
    tmp['DD'] = dd
    tmp['FA'] = fa
    tmp['FF'] = ff
    return tmp

def get_optimizer(proj, site, *args, **kwargs):
    #if site=='DP':
    #    return eval('get_optimizer_%s(proj, mode="%s", *args, **kwargs)'%(site,mode))
    return eval('get_optimizer_%s(proj, *args, **kwargs)'%site)

def get_optimizer_FF(PROJ, *args, **kwargs):
    games = np.unique(PROJ.Game).tolist()

    ## FantasyFeud
    # {'G': 3, 'F': 3, 'C': 2, 'UTIL': 2}
    nslots = 10
    salary_cap = 1000000.
    
    tab = PROJ.copy()
    print mode, style
    try:
        tab.Proj
    except:
        print 'Cannot find default column "Proj", defaulting to column "PROJ_FF"'
        tab['Proj'] = tab.PROJ_FF
    tab['Pos'] = tab.Position

    # Use only players in the selected games
    teams = [x for y in games for x in y.split('@')]
    games_dic = [x.split('@') for x in games]
    games_dic.extend([x.split('@')[::-1] for x in games])
    games_dic = dict(games_dic)
    tab = tab[np.array([t in teams for t in tab.Team], dtype=bool)]

    # This is for ordering players in the table by position
    pos2num = {'PG': 1, 'SG': 1, 'SF': 3, 'PF': 4, 'C': 5}
    tab['PosNum'] = [min([pos2num[y] for y in x.split(',')]) for x in tab.Position]

    tab = tab.reset_index().set_index('TAG')
    
    ## Create Optimizer
    lo = lopt.LineupOptimizer(tab, nslots, salary_cap)

    # These constraints are specified on the websites
    # Roster Constraint
    lo.addPositionConstraint('PG/SG', 'ge', 3)
    lo.addPositionConstraint('SF/PF', 'ge', 3)
    lo.addPositionConstraint('C', 'ge', 2)
    lo.addPositionConstraint('PG/SG/SF/PF/C', 'eq', nslots)
    
    return lo

def get_optimizer_FA(PROJ, *args, **kwargs):
    games = np.unique(PROJ.Game).tolist()
    try:
        mode = kwargs['mode']
    except KeyError:
        mode = ''
    try:
        style = kwargs['style']
    except KeyError:
        style = ''

    ## FantasyAces
    # {'G': 3, 'F': 3, 'C': 1, 'UTIL': 2}
    nslots = 9
    if mode == 'pro':
        salary_cap = None
    else:
        salary_cap = 45000.
    tab = PROJ.copy()
    print mode, style
    try:
        tab.Proj
    except:
        print 'Cannot find default column "Proj", defaulting to column "PROJ_FA"'
        tab['Proj'] = tab.PROJ_FA
    tab['Pos'] = tab.Position

    # Use only players in the selected games
    teams = [x for y in games for x in y.split('@')]
    games_dic = [x.split('@') for x in games]
    games_dic.extend([x.split('@')[::-1] for x in games])
    games_dic = dict(games_dic)
    tab = tab[np.array([t in teams for t in tab.Team], dtype=bool)]

    # This is for ordering players in the table by position
    pos2num = {'PG': 1, 'SG': 1, 'SF': 3, 'PF': 4, 'C': 5}
    tab['PosNum'] = [min([pos2num[y] for y in x.split(',')]) for x in tab.Position]

    tab = tab.reset_index().set_index('TAG')
    
    ## Create Optimizer
    lo = lopt.LineupOptimizer(tab, nslots, salary_cap)

    # These constraints are specified on the websites
    # Roster Constraint
    lo.addPositionConstraint('PG/SG', 'ge', 3)
    lo.addPositionConstraint('SF/PF', 'ge', 3)
    lo.addPositionConstraint('C', 'ge', 1)
    lo.addPositionConstraint('PG/SG/SF/PF/C', 'eq', nslots)
    
    if mode == 'pro':
        if style == 'under40k':
            lo.addTableConstraint('Salary', 'le', 40000.)
        elif style == 'overspend':
            lo.addTableConstraint('Salary', 'ge', 45000.)
            lo.addTableConstraint('Salary', 'le', 50000.)
            Delta = pulp.LpVariable('Delta')
            lo.prob += pulp.lpSum([lo.table.Salary[k]*lo.player_vars[k] for k in lo.players]) + Delta == 45000.,\
                        'Delta Salary Cap Constraint'
            lo.prob.setObjective(pulp.lpSum([lo.table.Proj[k]*lo.player_vars[k] for k in lo.players]) -(.8/50)*Delta)
        elif style == 'mid_underspend':
            lo.addTableConstraint('Salary', 'le', 44750.)
            lo.addTableConstraint('Salary', 'ge', 40001.)
            Delta = pulp.LpVariable('Delta')
            lo.prob += pulp.lpSum([lo.table.Salary[k]*lo.player_vars[k] for k in lo.players]) + Delta == 45000.,\
                        'Delta Salary Cap Constraint'
            lo.prob.setObjective(pulp.lpSum([lo.table.Proj[k]*lo.player_vars[k] for k in lo.players]) + (.4/50)*Delta)
        elif style == 'mid_nopenalty':
            lo.addTableConstraint('Salary', 'le', 45000.)
            lo.addTableConstraint('Salary', 'ge', 44751.)
    
    return lo

def get_optimizer_DD(PROJ, *args, **kwargs):
    games = np.unique(PROJ.Game).tolist()

    ## DraftPot
    # {'PG': 1, 'SG': 1, 'SF': 1, 'PF': 1, 'C': 1, 'UTIL': 5}
    nslots = 9
    salary_cap = 100000.0
    tab = PROJ.copy()
    try:
        tab.Proj
    except:
        print 'Cannot find default column "Proj", defaulting to column "PROJ_DD"'
        tab['Proj'] = tab.PROJ_DD
    tab['Pos'] = tab.Position

    # Use only players in the selected games
    teams = [x for y in games for x in y.split('@')]
    games_dic = [x.split('@') for x in games]
    games_dic.extend([x.split('@')[::-1] for x in games])
    games_dic = dict(games_dic)
    tab = tab[np.array([t in teams for t in tab.Team], dtype=bool)]

    # This is for ordering players in the table by position
    pos2num = {'PG': 1, 'PG,SG': 1.5, 'SG': 2, 'SG,SF': 2.5, 'SF': 3, 'SF,PF': 3.5, 'PF': 4, 'PF,C': 4.5, 'C': 5}
    tab['PosNum'] = [min([pos2num[y] for y in x.split(',')]) for x in tab.Position]
    tab['AllOnes'] = 1.0

    tab = tab.reset_index().set_index('TAG')
    
    ## Create Optimizer
    lo = lopt.LineupOptimizer(tab, nslots, salary_cap)

    # These constraints are specified on the websites
    # Roster Constraint
    lo.addPositionConstraint('PG/PG,SG/SG,PG', 'ge', 2)
    lo.addPositionConstraint('SG/PG,SG/SG,PG/SG,SF/SF,SG', 'ge', 2)
    lo.addPositionConstraint('SF/SF,SG/SG,SF/PF,SF/SF,PF', 'ge', 2)
    lo.addPositionConstraint('PF/SF,PF/PF,SF/C,SF/SF,C', 'ge', 2)
    lo.addPositionConstraint('C/C,PF/PF,C', 'ge', 1)
    lo.addPositionConstraint('PG/PG,SG/SG,PG/SG', 'le', 4)
    lo.addPositionConstraint('SG/SG,SF/SF,SG/SF', 'le', 4)
    lo.addPositionConstraint('SF/PF,SF/SF,PF/PF', 'le', 4)
    lo.addPositionConstraint('PF/C,PF/PF,C/C', 'le', 3)
    lo.addPositionConstraint('PG', 'le', 2)
    lo.addPositionConstraint('SG', 'le', 2)
    lo.addPositionConstraint('SF', 'le', 2)
    lo.addPositionConstraint('PF', 'le', 2)
    lo.addPositionConstraint('C', 'le', 1)
    lo.addPositionConstraint('PG/PG,SG/SG,PG/SG/SG,SF/SF,SG', 'ge', 4)
    lo.addPositionConstraint('SF/SF,SG/SG,SF/SG/PF,SF/SF,PF', 'ge', 4)
    lo.addPositionConstraint('SF/SF,PF/PF,SF/PF/PF,C/C,PF', 'ge', 4)
    lo.addTableConstraint('AllOnes', 'eq', nslots)

    return lo

def get_optimizer_DP(PROJ, *args, **kwargs):
    games = np.unique(PROJ.Game).tolist()

    try:
        mode = kwargs['mode']
    except KeyError:
        mode='GM'
    
    ## DraftPot
    # {'PG': 1, 'SG': 1, 'SF': 1, 'PF': 1, 'C': 1, 'UTIL': 5}
    nslots = 10
    salary_cap = 300.0 if mode=='GM' else None
    tab = PROJ.copy()
    try:
        tab.Proj
    except:
        print 'Cannot find default column "Proj", defaulting to column "PROJ_DP"'
        tab['Proj'] = tab.PROJ_DP
    tab['Pos'] = tab.Position

    # Use only players in the selected games
    teams = [x for y in games for x in y.split('@')]
    games_dic = [x.split('@') for x in games]
    games_dic.extend([x.split('@')[::-1] for x in games])
    games_dic = dict(games_dic)
    tab = tab[np.array([t in teams for t in tab.Team], dtype=bool)]

    # This is for ordering players in the table by position
    pos2num = {'PG': 1, 'SG': 2, 'SF': 3, 'PF': 4, 'C': 5}
    tab['PosNum'] = [pos2num[x] for x in tab.Position]

    tab = tab.reset_index().set_index('TAG')
    
    ## Create Optimizer
    lo = lopt.LineupOptimizer(tab, nslots, salary_cap)

    # These constraints are specified on the websites
    # Roster Constraint
    lo.addPositionConstraint('PG', 'ge', 1)
    lo.addPositionConstraint('SG', 'ge', 1)
    lo.addPositionConstraint('SF', 'ge', 1)
    lo.addPositionConstraint('PF', 'ge', 1)
    lo.addPositionConstraint('C', 'ge', 1)
    lo.addPositionConstraint('PG/SG/SF/PF/C', 'le', nslots)
    # At least players from 3 teams
    for k, team in enumerate(np.unique(tab.Team)):
        for team2 in np.unique(tab.Team)[k+1:]:
            lo.addTeamLimitConstraint('%s/%s' % (team,team2), 'le', nslots-1)
        # Max 6 players from a single team
        lo.addTeamLimitConstraint(team, 'le', 5)

    return lo    

def get_optimizer_YH(PROJ, *args, **kwargs):
    games = np.unique(PROJ.Game).tolist()

    ## YAHOO
    # {'PG': 1, 'SG': 1, 'SF': 1, 'PF': 1, 'C': 1, 'G': 1, 'F': 1, 'UTIL': 1}
    nslots = 8
    salary_cap = 200.
    tab = PROJ.copy()
    try:
        tab.Proj
    except:
        print 'Cannot find default column "Proj", defaulting to column "PROJ_YAHOO"'
        tab['Proj'] = tab.PROJ_YH
    tab['Pos'] = tab.Position

    # Use only players in the selected games
    teams = [x for y in games for x in y.split('@')]
    games_dic = [x.split('@') for x in games]
    games_dic.extend([x.split('@')[::-1] for x in games])
    games_dic = dict(games_dic)
    tab = tab[np.array([t in teams for t in tab.Team], dtype=bool)]

    # This is for ordering players in the table by position
    pos2num = {'PG': 1, 'SG': 2, 'SF': 3, 'PF': 4, 'C': 5}
    tab['PosNum'] = [pos2num[x] for x in tab.Position]

    tab = tab.reset_index().set_index('TAG')
    
    ## Create Optimizer
    lo = lopt.LineupOptimizer(tab, nslots, salary_cap)

    # These constraints are specified on the websites
    # Roster Constraint
    lo.addPositionConstraint('PG', 'ge', 1)
    lo.addPositionConstraint('SG', 'ge', 1)
    lo.addPositionConstraint('SF', 'ge', 1)
    lo.addPositionConstraint('PF', 'ge', 1)
    lo.addPositionConstraint('C', 'ge', 1)
    lo.addPositionConstraint('PG/SG', 'le', 3)
    lo.addPositionConstraint('SF/PF', 'le', 3)
    lo.addPositionConstraint('PG/SG', 'ge', 2)
    lo.addPositionConstraint('SF/PF', 'ge', 2)
    lo.addPositionConstraint('C', 'le', 2)
    # At least players from 3 teams (this is redundant to the next constraint)
    for k, team in enumerate(np.unique(tab.Team)):
        for team2 in np.unique(tab.Team)[k+1:]:
            lo.addTeamLimitConstraint('%s/%s' % (team,team2), 'le', 7)
        # Max 6 players from a single team
        lo.addTeamLimitConstraint(team, 'le', 6)

    return lo

def get_optimizer_DK(PROJ, *args, **kwargs):
    games = np.unique(PROJ.Game).tolist()

    ## DraftKings
    # {'PG': 1, 'SG': 1, 'SF': 1, 'PF': 1, 'C': 1, 'G': 1, 'F': 1, 'UTIL': 1}
    nslots = 8
    salary_cap = 50000
    tab = PROJ.copy()
    try:
        tab.Proj
    except:
        print 'Cannot find default column "Proj", defaulting to column "PROJ_DK"'
        tab['Proj'] = tab.PROJ_DK
    tab['Pos'] = tab.Position

    # Use only players in the selected games
    teams = [x for y in games for x in y.split('@')]
    games_dic = [x.split('@') for x in games]
    games_dic.extend([x.split('@')[::-1] for x in games])
    games_dic = dict(games_dic)
    tab = tab[np.array([t in teams for t in tab.Team], dtype=bool)]

    # This is for ordering players in the table by position
    pos2num = {'PG': 1, 'SG': 2, 'SF': 3, 'PF': 4, 'C': 5}
    tab['PosNum'] = [pos2num[x] for x in tab.Position]

    tab = tab.reset_index().set_index('TAG')
    
    ## Create Optimizer
    lo = lopt.LineupOptimizer(tab, nslots, salary_cap)

    # These constraints are specified on the websites
    # Roster Constraint
    lo.addPositionConstraint('PG', 'ge', 1)
    lo.addPositionConstraint('SG', 'ge', 1)
    lo.addPositionConstraint('SF', 'ge', 1)
    lo.addPositionConstraint('PF', 'ge', 1)
    lo.addPositionConstraint('C', 'ge', 1)
    lo.addPositionConstraint('PG/SG', 'le', 3)
    lo.addPositionConstraint('SF/PF', 'le', 3)
    lo.addPositionConstraint('PG/SG', 'ge', 2)
    lo.addPositionConstraint('SF/PF', 'ge', 2)
    lo.addPositionConstraint('C', 'le', 2)
    lo.addPositionConstraint('PG/SG/SF/PF/C', 'eq', nslots)
    # At least players from 2 teams (this is redundant to the next constraint)
#     for k, team in enumerate(np.unique(tab.Team)):
#         lo.addTeamLimitConstraint(team, 'le', 7)
    # At least players from 2 different games
    for game in games:
        teams = game.replace('@','/')
        lo.addTeamLimitConstraint(teams, 'le', 7)

    return lo

def get_optimizer_FD(PROJ, *args, **kwargs):
    games = np.unique(PROJ.Game).tolist()
    try:
        penalty = kwargs['penalty']
    except KeyError:
        penalty = 0
    
    ## FanDuel
    # {'PG': 2, 'SG': 2, 'SF': 2, 'PF': 2, 'C': 1}
    nslots = 9
    salary_cap = 60000
    tab = PROJ.copy()
    try:
        tab.Proj
    except:
        print 'Cannot find default column "Proj", defaulting to column "PROJ_FD"'
        tab['Proj'] = tab.PROJ_FD
    tab['Pos'] = tab.Position

    # Use only players in the selected games
    teams = [x for y in games for x in y.split('@')]
    games_dic = [x.split('@') for x in games]
    games_dic.extend([x.split('@')[::-1] for x in games])
    games_dic = dict(games_dic)
    tab = tab[np.array([t in teams for t in tab.Team], dtype=bool)]

    # This is for ordering players in the table by position
    pos2num = {'PG': 1, 'SG': 2, 'SF': 3, 'PF': 4, 'C': 5}
    tab['PosNum'] = [pos2num[x] for x in tab.Position]

    tab = tab.reset_index().set_index('TAG')
    
    ## Create Optimizer
    if penalty != 0:
        lo = lopt.CongruentLineupOptimizer(tab, nslots, penalty, salary_cap)
    else:
        lo = lopt.LineupOptimizer(tab, nslots, salary_cap)

    # These constraints are specified on the websites
    # Roster Constraint
    lo.addPositionConstraint('PG', 'eq', 2)
    lo.addPositionConstraint('SG', 'eq', 2)
    lo.addPositionConstraint('SF', 'eq', 2)
    lo.addPositionConstraint('PF', 'eq', 2)
    lo.addPositionConstraint('C', 'eq', 1)
    # At least players from 3 teams
    for k, team in enumerate(np.unique(tab.Team)):
        for team2 in np.unique(tab.Team)[k+1:]:
            lo.addTeamLimitConstraint('%s/%s' % (team,team2), 'le', 8)
        # Max 4 players from a single team
        lo.addTeamLimitConstraint(team, 'le', 4)

    return lo

def get_global_stats():
    global_stats = pd.read_csv('global_stats.csv', index_col=0)
    global_stats.GAME_DATE = [datetime.datetime.strptime(x, '%Y-%m-%d').date() for x in global_stats.GAME_DATE]
    global_stats = global_stats.sort(['GAME_DATE','Game_ID','TEAM'])
    return global_stats

def get_point_breakdown(lcf, global_stats, site='FD'):
    tmp = global_stats[(global_stats.Player_ID==get_pid(lcf)) & (global_stats.SEASON_ID==22015)]
    totals = tmp[['AST','PTS','BLK','REB','STL','TOV',site]].sum()
    print 'PTS: ', totals.PTS / totals.FD
    print 'AST: ', 1.5*totals.AST / (totals[site] + totals.TOV)
    print 'BLK: ', 2.0*totals.BLK / (totals[site] + totals.TOV)
    print 'REB: ', 1.2*totals.REB / (totals[site] + totals.TOV)
    print 'STL: ', 2.0*totals.STL / (totals[site] + totals.TOV)
    print 'TOV: ', 1.*totals.TOV / (totals[site] + totals.TOV)

fix_team_abbrev = {
    'GS' : 'GSW',
    'NO' : 'NOP',
    'SA' : 'SAS',
    'NY' : 'NYK',
    'PHO': 'PHX',
}

def fix_game_abbrev(game):
    a, h = game.upper().split('@')
    a = a.strip()
    h = h.strip()
    try:
        a = fix_team_abbrev[a]
    except:
        pass
    try:
        h = fix_team_abbrev[h]
    except:
        pass
    return '%s@%s' % (a, h)

def get_player_list(today):
    '''
    Grab the player lists from all of the sites and combine them into a dataframe.
    '''
    fname = 'pl_fd_nba_%s.csv' % today
    pl = pd.read_csv('PlayerLists/' + fname)
    try:
        del pl['Id']
    except:
        pass
    pl = pl.drop_duplicates()
    #tmp = [x.split('@') for x in pl.Game]
    #tmp = [(fix_team_abbrev[x] if x in fix_team_abbrev else x, fix_team_abbrev[y] if y in fix_team_abbrev else y) for x, y in tmp]
    #pl.Game = ['@'.join(x) for x in tmp]
    pl.Game = [fix_game_abbrev(x) for x in pl.Game]
    pl.Team = [fix_team_abbrev[x] if x in fix_team_abbrev else x for x in pl.Team]
    pl.Opponent = [fix_team_abbrev[x] if x in fix_team_abbrev else x for x in pl.Opponent]
    try:
        del pl['Unnamed: 12']
        del pl['Unnamed: 13']
    except:
        pass
    pl['LAST_COMMA_FIRST'] = ['%s, %s' % (l,f) for l,f in zip(pl['Last Name'],pl['First Name'])]
    pl['LAST_COMMA_FIRST'] = [x if x not in COMMA_NAME_CORRECTIONS else COMMA_NAME_CORRECTIONS[x] for x in pl.LAST_COMMA_FIRST]
    pl['Player_ID'] = [get_pid(x) for x in pl.LAST_COMMA_FIRST]
    pl['FD_Salary'] = [float(x) for x in pl.Salary]
    pl['FD_Position'] = pl['Position']
    del pl['Position']
    del pl['Salary']
    del pl['FPPG']
    del pl['Played']
    pl = pl.set_index('Player_ID')
    pl = pl.groupby(pl.index).first()
    
    try:
        fname = 'pl_dk_nba_%s.csv' % today
        dk = pd.read_csv('PlayerLists/' + fname, index_col=False)
        dk['First Name'] = [x.split(' ')[0] for x in dk.Name]
        dk['Last Name'] = [' '.join(x.split(' ')[1:]) for x in dk.Name]
        dk['LAST_COMMA_FIRST'] = ['%s, %s' % (l,f) for l,f in zip(dk['Last Name'],dk['First Name'])]
        dk['LAST_COMMA_FIRST'] = [x if x not in COMMA_NAME_CORRECTIONS else COMMA_NAME_CORRECTIONS[x] for x in dk.LAST_COMMA_FIRST]
        dk['Player_ID'] = [get_pid(x) for x in dk.LAST_COMMA_FIRST]
        dk = dk[~np.isnan(dk.Player_ID)]
        dk['DK_Salary'] = [float(x) for x in dk.Salary]
        try:
            dk['DK_Position'] = dk.Position
        except:
            dk['DK_Position'] = dk[dk.keys()[0]]
        dk = dk.set_index('Player_ID')
        dk = dk.groupby(dk.index).first()
    except IOError:
        print fname,'not found'
    
    try:
        fname = 'pl_yh_nba_%s.csv' % today
        yh = pd.read_csv('PlayerLists/' + fname)
        yh['LAST_COMMA_FIRST'] = ['%s, %s' % (l,f) for l,f in zip(yh['Last Name'],yh['First Name'])]
        yh['LAST_COMMA_FIRST'] = [x if x not in COMMA_NAME_CORRECTIONS else COMMA_NAME_CORRECTIONS[x] for x in yh.LAST_COMMA_FIRST]
        yh['Player_ID'] = [get_pid(x) for x in yh.LAST_COMMA_FIRST]
        yh = yh[~np.isnan(yh.Player_ID)]
        yh['YH_Salary'] = [float(x) for x in yh.Salary]
        yh['YH_Position'] = yh['Position']
        yh = yh.set_index('Player_ID')
        yh = yh.groupby(yh.index).first()
    except IOError:
        print fname,'not found'
    
    try:
        fname = 'pl_dp_nba_%s.csv' % today
        dp = pd.read_csv('PlayerLists/' + fname)
        dp['LAST_COMMA_FIRST'] = ['%s, %s' % (l,f) for l,f in zip(dp['Last Name'],dp['First Name'])]
        dp['LAST_COMMA_FIRST'] = [x if x not in COMMA_NAME_CORRECTIONS else COMMA_NAME_CORRECTIONS[x] for x in dp.LAST_COMMA_FIRST]
        dp['Player_ID'] = [get_pid(x) for x in dp.LAST_COMMA_FIRST]
        dp = dp[~np.isnan(dp.Player_ID)]
        dp['DP_Position'] = [x.upper() for x in dp['Position']]
        dp['DP_Salary'] = [float(x) for x in dp.Dppg]
        dp = dp.set_index('Player_ID')
        dp = dp.groupby(dp.index).first()
        dp = dp.drop_duplicates()
    except IOError:
        print fname,'not found'
    
    try:
        fname = 'pl_dd_nba_%s.csv' % today
        dd = pd.read_csv('PlayerLists/' + fname)
        dd['First Name'] = [x.split(' ')[0] for x in dd['Player Name']]
        dd['Last Name'] = [' '.join(x.split(' ')[1:]) for x in dd['Player Name']]
        dd['LAST_COMMA_FIRST'] = ['%s, %s' % (l,f) for l,f in zip(dd['Last Name'],dd['First Name'])]
        dd['LAST_COMMA_FIRST'] = [x if x not in COMMA_NAME_CORRECTIONS else COMMA_NAME_CORRECTIONS[x] for x in dd.LAST_COMMA_FIRST]
        dd['Player_ID'] = [get_pid(x) for x in dd.LAST_COMMA_FIRST]
        dd = dd[~np.isnan(dd.Player_ID)]
        dd['DD_Position'] = [x.upper() for x in dd['Position']]
        dd['DD_Salary'] = [float(x) for x in dd.Salary]
        print dd[np.isnan(dd.Player_ID)].LAST_COMMA_FIRST
        dd = dd[~np.isnan(dd.Player_ID)]
        tmp = pd.DataFrame()
        for pid in np.unique(dd.Player_ID):
            sub = dd[dd.Player_ID == int(pid)]
            pos = ','.join(sub.DD_Position.tolist())
            row = sub.iloc[0]
            row.DD_Position = pos
            tmp = tmp.append(row)
        dd = tmp
        dd = dd.set_index('Player_ID')
        dd = dd.groupby(dd.index).first()
    except IOError:
        print fname,'not found'
    
    try:
        fname = 'pl_fa_nba_%s.csv' % today
        fa = pd.read_csv('PlayerLists/' + fname, index_col=False)
        fa['LAST_COMMA_FIRST'] = ['%s, %s' % (l,f) for l,f in zip(fa['Last Name'],fa['First Name'])]
        fa['LAST_COMMA_FIRST'] = [x if x not in COMMA_NAME_CORRECTIONS else COMMA_NAME_CORRECTIONS[x] for x in fa.LAST_COMMA_FIRST]
        fa['Player_ID'] = [get_pid(x) for x in fa.LAST_COMMA_FIRST]
        fa['FA_Position'] = fa.Pos
        fa['FA_Salary'] = fa.Sal
        fa = fa.set_index('Player_ID')
        fa = fa.groupby(fa.index).first()
    except IOError:
        print fname, 'not found'
    
    try:
        fname = 'pl_ff_nba_%s.csv' % today
        ff = pd.read_csv('PlayerLists/' + fname)
        ff['First Name'] = [x.split(' ')[0] for x in ff['Name']]
        ff['Last Name'] = [' '.join(x.split(' ')[1:]) for x in ff['Name']]
        ff['LAST_COMMA_FIRST'] = ['%s, %s' % (l,f) for l,f in zip(ff['Last Name'],ff['First Name'])]
        ff['LAST_COMMA_FIRST'] = [x if x not in COMMA_NAME_CORRECTIONS else COMMA_NAME_CORRECTIONS[x] for x in ff.LAST_COMMA_FIRST]
        ff['Player_ID'] = [get_pid(x) for x in ff.LAST_COMMA_FIRST]
        ff = ff[~np.isnan(ff.Player_ID)]
        ff['FF_Position'] = [x.upper() for x in ff['Position']]
        ff['FF_Salary'] = [float(x) for x in ff.Salary]
        print ff[np.isnan(ff.Player_ID)].LAST_COMMA_FIRST
        ff = ff[~np.isnan(ff.Player_ID)]
        tmp = pd.DataFrame()
        for pid in np.unique(ff.Player_ID):
            sub = ff[ff.Player_ID == int(pid)]
            pos = ','.join(sub.FF_Position.tolist())
            row = sub.iloc[0]
            row.FF_Position = pos
            tmp = tmp.append(row)
        ff = tmp
        ff = ff.set_index('Player_ID')
        ff = ff.groupby(ff.index).first()
    except IOError:
        print fname,'not found'

    print np.unique(pl.Game)
    try:
        pl['DK_Salary'] = dk['DK_Salary']
        pl['DK_Position'] = dk['DK_Position']
    except:
        pass
    print np.unique(pl.Game)
    try:
        pl['YH_Salary'] = yh['YH_Salary']
        pl['YH_Position'] = yh['YH_Position']
        pl['Time'] = yh['Time']
    except:
        pass
    print np.unique(pl.Game)
    try:
        pl['DP_Salary'] = dp['DP_Salary']
        pl['DP_Position'] = dp['DP_Position']
    except Exception, exc:
        print exc.message
    print np.unique(pl.Game)
    try:
        pl['DD_Salary'] = dd['DD_Salary']
        pl['DD_Position'] = dd['DD_Position']
    except:
        pass
    try:
        pl['FA_Salary'] = fa['FA_Salary']
        pl['FA_Position'] = fa['FA_Position']
    except Exception, exc:
        print exc.message
    print np.unique(pl.Game)
    try:
        pl['FF_Salary'] = ff['FF_Salary']
        pl['FF_Position'] = ff['FF_Position']
    except Exception, exc:
        print exc.message
    print np.unique(pl.Game)
    return pl
