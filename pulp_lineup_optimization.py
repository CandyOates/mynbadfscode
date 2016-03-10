import pulp
import numpy as np

def fd_nba_minprob_con_factory(x, prob, vs, table):
    prob += pulp.lpSum([float(table.Position.iloc[n]=='PG')*vs[n] for n in range(len(vs))]) <= 2, "PG_Total_UB"
    prob += pulp.lpSum([float(table.Position.iloc[n]=='SG')*vs[n] for n in range(len(vs))]) <= 2, "SG_Total_UB"
    prob += pulp.lpSum([float(table.Position.iloc[n]=='SF')*vs[n] for n in range(len(vs))]) <= 2, "SF_Total_UB"
    prob += pulp.lpSum([float(table.Position.iloc[n]=='PF')*vs[n] for n in range(len(vs))]) <= 2, "PF_Total_UB"
    prob += pulp.lpSum([float(table.Position.iloc[n]=='C')*vs[n]  for n in range(len(vs))]) <= 1, "C_Total_UB"
    
    prob += pulp.lpSum([float(table.Position.iloc[n]=='PG')*vs[n] for n in range(len(vs))]) >= 0, "PG_Total_LB"
    prob += pulp.lpSum([float(table.Position.iloc[n]=='SG')*vs[n] for n in range(len(vs))]) >= 0, "SG_Total_LB"
    prob += pulp.lpSum([float(table.Position.iloc[n]=='SF')*vs[n] for n in range(len(vs))]) >= 0, "SF_Total_LB"
    prob += pulp.lpSum([float(table.Position.iloc[n]=='PF')*vs[n] for n in range(len(vs))]) >= 0, "PF_Total_LB"
    prob += pulp.lpSum([float(table.Position.iloc[n]=='C')*vs[n]  for n in range(len(vs))]) >= 0, "C_Total_LB"
    return prob

class RobustCongruentLineupOptimizer(object):
    def __init__(self, table, roster_slots, penalty, salary_cap=None, min_prob_con_factory=None):
        """
        Makes Lineup Optimization easy.
        max_x min_w {s'(x-w) - .5*penalty*||x-w||^2}
        sum(w) = roster_slots
        
        Input:
            table - pd.DataFrame
                Contains relevant data on each player. Must include the columns ['Pos','Salary','Team','OwnLB','OwnUB']. 
                If you want
                the resulting table sorted, include a column ['PosNum'] which assigns a numeric value to each position.
            roster_slots - int
                the total number of spaces in the roster
                For example, in NBA 1 through 5 for PG through C.
            penalty - float
                the weight assigned to the penalty term for weight misalignment. Preliminary tests suggest 10 is a 
                good value for the NBA. A penalty of 0 is equivalent to the MILP formulation solved by the 
                LineupOptimizer class.
            salary_cap - float
                The total salary limit of the team, if there is any. Default is None.
            min_prob_con_factory - function
                Takes the most recent candidate solution of the maximization problem as an array of floats or ints, 
                a pulp.LpProblem, a list of pulp.LpVariable, and a pd.DataFrame and returns a pulp.LpProblem with
                the appropriate constraints for the minimization problem for the specific contest type and site this class
                is being used for. If unspecified, None is defaulted and the only constraint in the minimization problem will
                be that the total of the weights sum to roster_slots and the weights stay between their bounds.
        """
        self.nslots = roster_slots
        self.table = table
        self.salary_cap = salary_cap
        self.pen = penalty
        if self.pen == 0:
            raise Exception('penalty must be nonzero.')
        self.min_prob_con_factory = min_prob_con_factory
        
        self.players = table.index.tolist()

        self.prob = pulp.LpProblem('Lineup Optimization', pulp.LpMaximize)
        self.player_vars = pulp.LpVariable.dicts("p", self.players, 0, 1, 'Binary')
        self.rev_player_vars = dict(zip([str(x) for x in self.player_vars.values()], \
                                        self.player_vars.keys()))
        self.z = pulp.LpVariable('z')
        
        mu = np.array(self.table.Proj) / (2.*self.pen)
        self.m = np.empty(len(mu))
        projs = self.table.Proj.loc[self.players].tolist()
        for n in range(len(self.players)):
            t = mu
            el = self.table.OwnLB.loc[self.players[n]]
            u = self.table.OwnUB.loc[self.players[n]]
            t[n] = el
            fl = t.dot(projs) - .5*self.pen*t.dot(t)
            t[n] = u
            fu = t.dot(projs) - .5*self.pen*t.dot(t)
            self.m[n] = (fu - fl) / (u - el)

        # Objective
        self.prob += pulp.lpSum([self.table.Proj[k]*self.player_vars[k] for k in self.players]) - self.z*self.pen, 'Aggregate Projected Score'

        # Salary Cap Constraint
        if self.salary_cap is not None:
            self.prob += pulp.lpSum([self.table.Salary[k]*self.player_vars[k] for k in self.players]) <= self.salary_cap, 'Salary Cap Constraint'

        self.const_tracker = {}
    
    def _addQuadraticConstraint(self, x, iteration):
        """
        Add a linear approximation to the quadratic constraint.
        
        x - binary array
            solution of the last MILP iteration
        iteration - int
            the iteration number of the last MILP iteration
        """
        con_name = 'QUADCON%d'%iteration
        y = np.array([x[n] - self.table.RobustOwn.loc[k] for n, k in enumerate(self.players)], dtype=float)
        self.prob += 2.*pulp.lpSum([y[n]*self.player_vars[k] for n, k in enumerate(self.players)]) - self.z <= \
            y.dot(y) + 2.*sum([y[n]*self.table.RobustOwn.loc[k] for n, k in enumerate(self.players)]), con_name
    
    def _removeQuadraticConstraints(self, ct):
        """
        Remove all linear approximation quadratic constraints.
        """
        for k in xrange(ct):
            con_name = 'QUADCON%d'%k
            self.removeConstraint(con_name)
    
    def removeConstraint(self, con_name):
        """
        Remove the constraint with name given by con_name.
        """
        try:
            del self.prob.constraints[con_name]
            return True
        except:
            return False
    
    def addPositionConstraint(self, pos, con_type, bound, con_name=None):
        """
        Constrain the number of position appearances.
        
        Inputs:
            pos - string
                position of player, if multiple, concatenated with '/'
            con_type - string
                {'le','eq','ge'}
            bound - float
                the rhs of the constraint
            con_name - string
                the name to assign the constraint. If not given, defaults to 'Column %s' % column
        
        Returns:
            True if successful, False otherwise
        """
        if con_name is None:
            con_name = 'Pos %s %s' % (con_type,pos)
            try:
                self.const_tracker[con_name] += 1
            except KeyError:
                self.const_tracker[con_name] = 0
            con_name = '%s%d' % (con_name , self.const_tracker[con_name])
        
        pos = pos.split('/')
        if con_type == 'eq':
            self.prob += pulp.lpSum([(self.table['Pos'].loc[k] in pos)*self.player_vars[k] for k in self.players]) == bound, con_name
            return True
        elif con_type == 'le':
            self.prob += pulp.lpSum([(self.table['Pos'].loc[k] in pos)*self.player_vars[k] for k in self.players]) <= bound, con_name
            return True
        elif con_type == 'ge':
            self.prob += pulp.lpSum([(self.table['Pos'].loc[k] in pos)*self.player_vars[k] for k in self.players]) >= bound, con_name
            return True
        raise(Exception('%s is not a valid input for con_type' % con_type))
    
    def _solve_minimization(self, x):
        minprob = pulp.LpProblem("min_prob", pulp.LpMinimize)
        varis = [pulp.LpVariable("y%d"%n, lowBound=x[n]-self.table.OwnUB.loc[self.players[n]], upBound=x[n]-self.table.OwnLB.loc[self.players[n]]) for n in range(len(self.players))]
        minprob += pulp.lpSum([self.m[n]*varis[n] for n in range(len(self.players))]), "LinearizedObjective"
        if self.min_prob_con_factory is not None:
            minprob = self.min_prob_con_factory(x, minprob, varis, self.table.loc[self.players])
        print minprob
        minprob.solve()
        
        if pulp.LpStatus[minprob.status] != 'Optimal':
            raise Exception('Minimization subproblem did not converge')
        self.table['RobustOwn'] = np.array([x[n]-float(v.value()) for n, v in enumerate(varis)], dtype=float)
    
    def _solve_maximization(self):
        self.prob.solve()
        x = [int(self.player_vars[k].varValue == 1 and str(self.player_vars[k])[0] == 'p') for k in self.players]
        xold = len(x) * [0]
        ct = 0
        while np.array(xold).dot(x) != self.nslots:
            xold = x
            x = [int(self.player_vars[k].varValue == 1 and str(self.player_vars[k])[0] == 'p') for k in self.players]
            self._addQuadraticConstraint(x, ct)
            ct += 1
            self.prob.solve()
        self._removeQuadraticConstraints(ct)
        x = [int(self.player_vars[k].varValue == 1 and str(self.player_vars[k])[0] == 'p') for k in self.players]
        return x
        
    def solve(self):
        """
        Solves the problem.
        
        Returns:
            status - the solver's status
            lineup - the inputed table subsetted to hold the players in the lineup. Sorts the lineup if a 'PosNum'
                     column was given.
        """
        self.prob.solve()
        x = [int(v.varValue == 1 and str(v)[0] == 'p') for v in self.prob.variables()]
        xold = len(x) * [0]
        xoldold = len(x) * [0]
        ct = 0
        while np.array(xold).dot(x) != self.nslots and np.array(xoldold).dot(x) != self.nslots:
            xoldold = xold
            xold = x
            self._solve_minimization(x)
            x = self._solve_maximization()
        
        I = []
        for v in self.prob.variables():
            if v.varValue == 1 and str(v)[0] == 'p':
                player = self.rev_player_vars[v.getName()]
                I.append(player)
        lu = self.table.loc[I]
        
        try:
            lu = lu.sort(['PosNum','Salary'], ascending=[True,False])
        except Exception, exc:
            pass
        
        if np.array(xoldold).dot(x) != self.nslots:
            return 'Oscillating', lu
        return pulp.LpStatus[self.prob.status], lu
    
    def addTableConstraint(self, column, con_type, bound, con_name=None):
        """
        Add a constraint with the dot product of a column in the inputed table.
        
        Inputs:
            column - string
                name of the column in the table to use
            con_type - string
                {'le','eq','ge'}
            bound - float
                the rhs of the constraint
            con_name - string
                the name to assign the constraint. If not given, defaults to 'Column %s' % column
        
        Returns:
            True if successful, False otherwise
        """
        if con_name is None:
            con_name = 'Column %s' % column
            try:
                self.const_tracker[con_name] += 1
            except KeyError:
                self.const_tracker[con_name] = 0
            con_name = '%s%d' % (con_name , self.const_tracker[con_name])
        
        if con_type == 'eq':
            self.prob += pulp.lpSum([self.table[column].loc[k]*self.player_vars[k] for k in self.players]) == bound, con_name
            return True
        elif con_type == 'le':
            self.prob += pulp.lpSum([self.table[column].loc[k]*self.player_vars[k] for k in self.players]) <= bound, con_name
            return True
        elif con_type == 'ge':
            self.prob += pulp.lpSum([self.table[column].loc[k]*self.player_vars[k] for k in self.players]) >= bound, con_name
            return True
        raise(Exception('%s is not a valid input for con_type' % con_type))
    
    def addTeamLimitConstraint(self, teams, con_type, bound):
        """
        Constrain the number of players from the combination of the teams. Multiple teams are entered
        delimited by a '/' as in 'CLE/DET'
        
        Inputs:
            teams - string
                teams to be constrained delimited by '/'
        
        see addColumnConstraint
        """
        teams = teams.split('/')
        con_name = 'Team %s Limit %s' % (teams, con_type)
        try:
            self.const_tracker[con_name] += 1
        except KeyError:
            self.const_tracker[con_name] = 0
        con_name = '%s%d' % (con_name , self.const_tracker[con_name])
        if con_type == 'eq':
            self.prob += pulp.lpSum([(self.table.loc[k]['Team'] in teams)*self.player_vars[k] for k in self.players]) == bound, con_name
            return True
        elif con_type == 'le':
            self.prob += pulp.lpSum([(self.table.loc[k]['Team'] in teams)*self.player_vars[k] for k in self.players]) <= bound, con_name
            return True
        elif con_type == 'ge':
            self.prob += pulp.lpSum([(self.table.loc[k]['Team'] in teams)*self.player_vars[k] for k in self.players]) >= bound, con_name
            return True
        raise(Exception('%s is not a valid input for con_type' % con_type))
    
    def addCustomConstraint(self, func, con_type, bound, con_name=None):
        """
        Add a custom constraint which results from passing the inputed table into func. This is a more
        general version of addColumnConstraints.
        
        Inputs:
            func - function
                Takes self.table as an input and returns a pd.Series with indices matching that of self.table
        
        see addColumnConstraints
        """
        series = func(self.table)
        
        if con_name is None:
            con_name = 'Custom Func %d'
            try:
                self.const_tracker[con_name] += 1
            except KeyError:
                self.const_tracker[con_name] = 0
            con_name = '%s%d' % (con_name , self.const_tracker[con_name])
        
        return self.addSeriesConstraint(series, con_type, bound, con_name)
    
    def addTableColumns(self, series, names):
        """
        Add the series to self.table with the corresponding column names.
        """
        for serie, name in zip(series, names):
            self.table[name] = serie
        return True
    
    def addSeriesConstraint(self, series, con_type, bound, con_name=None):
        """
        Add a custom constraint which results from passing the inputed table into func. This is a more
        general version of addColumnConstraints.
        
        Inputs:
            func - function
                Takes self.table as an input and returns a pd.Series with indices matching that of self.table
        
        see addColumnConstraints
        """
        if con_name is None:
            con_name = 'Custom Func %d'
            try:
                self.const_tracker[con_name] += 1
            except KeyError:
                self.const_tracker[con_name] = 0
            con_name = '%s%d' % (con_name , self.const_tracker[con_name])
        if con_type == 'eq':
            self.prob += pulp.lpSum([series.loc[k]*self.player_vars[k] for k in self.players]) == bound, con_name
            return True
        elif con_type == 'le':
            self.prob += pulp.lpSum([series.loc[k]*self.player_vars[k] for k in self.players]) <= bound, con_name
            return True
        elif con_type == 'ge':
            self.prob += pulp.lpSum([series.loc[k]*self.player_vars[k] for k in self.players]) >= bound, con_name
            return True
        raise(Exception('%s is not a valid input for con_type' % con_type))
    
    def updateObjective(self, series):
        """
        Replace the current objective function with the one specified by series.
        """
        self.prob.setObjective(pulp.lpSum([series[k]*self.player_vars[k] for k in self.players]) - self.pen*self.z)
    
    def disallowLineup(self, lineup):
        """
        Take a lineup DataFrame (subset of self.table) and don't allow this specific lineup.
        """
        players = lineup.index.tolist()
        con_name = 'BlockLineup %s' % str(players)
        self.prob += pulp.lpSum([(k in players)*self.player_vars[k] for k in self.players]) <= self.nslots-1, con_name
    
    def addPlayerConstraint(self, players, con_type, bound, con_name=None):
        """
        Make a constraint based on appearances of players.
        
        Inputs:
            players - string
                the names of players as found in the index of self.table delimited by '/'
        
        see addColumnConstraints
        """
        if con_name is None:
            con_name = 'Players %s %%d' % (players)
            try:
                self.const_tracker[con_name] += 1
            except KeyError:
                self.const_tracker[con_name] = 0
            con_name = '%s%d' % (con_name , self.const_tracker[con_name])
        players = players.split('/')
        if con_type == 'eq':
            self.prob += pulp.lpSum([self.player_vars[k] for k in players]) == bound, con_name
            return True
        elif con_type == 'le':
            self.prob += pulp.lpSum([self.player_vars[k] for k in players]) <= bound, con_name
            return True
        elif con_type == 'ge':
            self.prob += pulp.lpSum([self.player_vars[k] for k in players]) >= bound, con_name
            return True
        raise(Exception('%s is not a valid input for con_type' % con_type))

    

class CongruentLineupOptimizer(object):
    def __init__(self, table, roster_slots, penalty, salary_cap=None):
        """
        Makes Lineup Optimization easy.
        max s'(x-w) - .5*penalty*||x-w||^2
        
        Input:
            table - pd.DataFrame
                Contains relevant data on each player. Must include the columns ['Pos','Salary','Team','Own']. If you want
                the resulting table sorted, include a column ['PosNum'] which assigns a numeric value to each position.
            roster_slots - int
                the total number of spaces in the roster
                For example, in NBA 1 through 5 for PG through C.
            penalty - float
                the weight assigned to the penalty term for weight misalignment. Preliminary tests suggest 10 is a 
                good value for the NBA. A penalty of 0 is equivalent to the MILP formulation solved by the 
                LineupOptimizer class.
            salary_cap - float
                The total salary limit of the team, if there is any. Default is None.
        """
        self.nslots = roster_slots
        self.table = table
        self.salary_cap = salary_cap
        self.pen = penalty
        
        self.players = table.index.tolist()

        self.prob = pulp.LpProblem('Lineup Optimization', pulp.LpMaximize)
        self.player_vars = pulp.LpVariable.dicts("p", self.players, 0, 1, 'Binary')
        self.rev_player_vars = dict(zip([str(x) for x in self.player_vars.values()], \
                                        self.player_vars.keys()))
        self.z = pulp.LpVariable('z')

        # Objective
        self.prob += pulp.lpSum([self.table.Proj[k]*self.player_vars[k] for k in self.players]) - self.z*self.pen, 'Aggregate Projected Score'

        # Salary Cap Constraint
        if self.salary_cap is not None:
            self.prob += pulp.lpSum([self.table.Salary[k]*self.player_vars[k] for k in self.players]) <= self.salary_cap, 'Salary Cap Constraint'

        self.const_tracker = {}
    
    def _addQuadraticConstraint(self, x, iteration):
        """
        Add a linear approximation to the quadratic constraint.
        
        x - binary array
            solution of the last MILP iteration
        iteration - int
            the iteration number of the last MILP iteration
        """
        con_name = 'QUADCON%d'%iteration
        y = np.array([x[n] - self.table.Own.loc[k] for n, k in enumerate(self.players)], dtype=float)
        self.prob += 2.*pulp.lpSum([y[n]*self.player_vars[k] for n, k in enumerate(self.players)]) - self.z <= \
            y.dot(y) + 2.*sum([y[n]*self.table.Own.loc[k] for n, k in enumerate(self.players)]), con_name
    
    def _removeQuadraticConstraints(self, ct):
        """
        Remove all linear approximation quadratic constraints.
        """
        for k in xrange(ct):
            con_name = 'QUADCON%d'%k
            self.removeConstraint(con_name)
    
    def removeConstraint(self, con_name):
        """
        Remove the constraint with name given by con_name.
        """
        try:
            del self.prob.constraints[con_name]
            return True
        except:
            return False
    
    def addPositionConstraint(self, pos, con_type, bound, con_name=None):
        """
        Constrain the number of position appearances.
        
        Inputs:
            pos - string
                position of player, if multiple, concatenated with '/'
            con_type - string
                {'le','eq','ge'}
            bound - float
                the rhs of the constraint
            con_name - string
                the name to assign the constraint. If not given, defaults to 'Column %s' % column
        
        Returns:
            True if successful, False otherwise
        """
        if con_name is None:
            con_name = 'Pos %s %s' % (con_type,pos)
            try:
                self.const_tracker[con_name] += 1
            except KeyError:
                self.const_tracker[con_name] = 0
            con_name = '%s%d' % (con_name , self.const_tracker[con_name])
        
        pos = pos.split('/')
        if con_type == 'eq':
            self.prob += pulp.lpSum([(self.table['Pos'].loc[k] in pos)*self.player_vars[k] for k in self.players]) == bound, con_name
            return True
        elif con_type == 'le':
            self.prob += pulp.lpSum([(self.table['Pos'].loc[k] in pos)*self.player_vars[k] for k in self.players]) <= bound, con_name
            return True
        elif con_type == 'ge':
            self.prob += pulp.lpSum([(self.table['Pos'].loc[k] in pos)*self.player_vars[k] for k in self.players]) >= bound, con_name
            return True
        raise(Exception('%s is not a valid input for con_type' % con_type))
    
    def solve(self):
        """
        Solves the problem.
        
        Returns:
            status - the solver's status
            lineup - the inputed table subsetted to hold the players in the lineup. Sorts the lineup if a 'PosNum'
                     column was given.
        """
        self.prob.solve()
        x = [int(v.varValue == 1 and str(v)[0] == 'p') for v in self.prob.variables()]
        xold = len(x) * [0]
        ct = 0
        while np.array(xold).dot(x) != self.nslots:
            xold = x
            x = [int(v.varValue == 1 and str(v)[0] == 'p') for v in self.prob.variables()]
            self._addQuadraticConstraint(x, ct)
            ct += 1
            self.prob.solve()
        self._removeQuadraticConstraints(ct)
        
        I = []
        for v in self.prob.variables():
            if v.varValue == 1 and str(v)[0] == 'p':
                player = self.rev_player_vars[v.getName()]
                I.append(player)
        lu = self.table.loc[I]
        try:
            lu = lu.sort(['PosNum','Salary'], ascending=[True,False])
        except Exception, exc:
            pass
        return pulp.LpStatus[self.prob.status], lu
    
    def addTableConstraint(self, column, con_type, bound, con_name=None):
        """
        Add a constraint with the dot product of a column in the inputed table.
        
        Inputs:
            column - string
                name of the column in the table to use
            con_type - string
                {'le','eq','ge'}
            bound - float
                the rhs of the constraint
            con_name - string
                the name to assign the constraint. If not given, defaults to 'Column %s' % column
        
        Returns:
            True if successful, False otherwise
        """
        if con_name is None:
            con_name = 'Column %s' % column
            try:
                self.const_tracker[con_name] += 1
            except KeyError:
                self.const_tracker[con_name] = 0
            con_name = '%s%d' % (con_name , self.const_tracker[con_name])
        
        if con_type == 'eq':
            self.prob += pulp.lpSum([self.table[column].loc[k]*self.player_vars[k] for k in self.players]) == bound, con_name
            return True
        elif con_type == 'le':
            self.prob += pulp.lpSum([self.table[column].loc[k]*self.player_vars[k] for k in self.players]) <= bound, con_name
            return True
        elif con_type == 'ge':
            self.prob += pulp.lpSum([self.table[column].loc[k]*self.player_vars[k] for k in self.players]) >= bound, con_name
            return True
        raise(Exception('%s is not a valid input for con_type' % con_type))
    
    def addTeamLimitConstraint(self, teams, con_type, bound):
        """
        Constrain the number of players from the combination of the teams. Multiple teams are entered
        delimited by a '/' as in 'CLE/DET'
        
        Inputs:
            teams - string
                teams to be constrained delimited by '/'
        
        see addColumnConstraint
        """
        teams = teams.split('/')
        con_name = 'Team %s Limit %s' % (teams, con_type)
        try:
            self.const_tracker[con_name] += 1
        except KeyError:
            self.const_tracker[con_name] = 0
        con_name = '%s%d' % (con_name , self.const_tracker[con_name])
        if con_type == 'eq':
            self.prob += pulp.lpSum([(self.table.loc[k]['Team'] in teams)*self.player_vars[k] for k in self.players]) == bound, con_name
            return True
        elif con_type == 'le':
            self.prob += pulp.lpSum([(self.table.loc[k]['Team'] in teams)*self.player_vars[k] for k in self.players]) <= bound, con_name
            return True
        elif con_type == 'ge':
            self.prob += pulp.lpSum([(self.table.loc[k]['Team'] in teams)*self.player_vars[k] for k in self.players]) >= bound, con_name
            return True
        raise(Exception('%s is not a valid input for con_type' % con_type))
    
    def addCustomConstraint(self, func, con_type, bound, con_name=None):
        """
        Add a custom constraint which results from passing the inputed table into func. This is a more
        general version of addColumnConstraints.
        
        Inputs:
            func - function
                Takes self.table as an input and returns a pd.Series with indices matching that of self.table
        
        see addColumnConstraints
        """
        series = func(self.table)
        
        if con_name is None:
            con_name = 'Custom Func %d'
            try:
                self.const_tracker[con_name] += 1
            except KeyError:
                self.const_tracker[con_name] = 0
            con_name = '%s%d' % (con_name , self.const_tracker[con_name])
        
        return self.addSeriesConstraint(series, con_type, bound, con_name)
    
    def addTableColumns(self, series, names):
        """
        Add the series to self.table with the corresponding column names.
        """
        for serie, name in zip(series, names):
            self.table[name] = serie
        return True
    
    def addSeriesConstraint(self, series, con_type, bound, con_name=None):
        """
        Add a custom constraint which results from passing the inputed table into func. This is a more
        general version of addColumnConstraints.
        
        Inputs:
            func - function
                Takes self.table as an input and returns a pd.Series with indices matching that of self.table
        
        see addColumnConstraints
        """
        if con_name is None:
            con_name = 'Custom Func %d'
            try:
                self.const_tracker[con_name] += 1
            except KeyError:
                self.const_tracker[con_name] = 0
            con_name = '%s%d' % (con_name , self.const_tracker[con_name])
        if con_type == 'eq':
            self.prob += pulp.lpSum([series.loc[k]*self.player_vars[k] for k in self.players]) == bound, con_name
            return True
        elif con_type == 'le':
            self.prob += pulp.lpSum([series.loc[k]*self.player_vars[k] for k in self.players]) <= bound, con_name
            return True
        elif con_type == 'ge':
            self.prob += pulp.lpSum([series.loc[k]*self.player_vars[k] for k in self.players]) >= bound, con_name
            return True
        raise(Exception('%s is not a valid input for con_type' % con_type))
    
    def updateObjective(self, series):
        """
        Replace the current objective function with the one specified by series.
        """
        self.prob.setObjective(pulp.lpSum([series[k]*self.player_vars[k] for k in self.players]) - self.pen*self.z)
    
    def disallowLineup(self, lineup):
        """
        Take a lineup DataFrame (subset of self.table) and don't allow this specific lineup.
        """
        players = lineup.index.tolist()
        con_name = 'BlockLineup %s' % str(players)
        self.prob += pulp.lpSum([(k in players)*self.player_vars[k] for k in self.players]) <= self.nslots-1, con_name
    
    def addPlayerConstraint(self, players, con_type, bound, con_name=None):
        """
        Make a constraint based on appearances of players.
        
        Inputs:
            players - string
                the names of players as found in the index of self.table delimited by '/'
        
        see addColumnConstraints
        """
        if con_name is None:
            con_name = 'Players %s %%d' % (players)
            try:
                self.const_tracker[con_name] += 1
            except KeyError:
                self.const_tracker[con_name] = 0
            con_name = '%s%d' % (con_name , self.const_tracker[con_name])
        players = players.split('/')
        if con_type == 'eq':
            self.prob += pulp.lpSum([self.player_vars[k] for k in players]) == bound, con_name
            return True
        elif con_type == 'le':
            self.prob += pulp.lpSum([self.player_vars[k] for k in players]) <= bound, con_name
            return True
        elif con_type == 'ge':
            self.prob += pulp.lpSum([self.player_vars[k] for k in players]) >= bound, con_name
            return True
        raise(Exception('%s is not a valid input for con_type' % con_type))

#################################################

class LineupOptimizer(object):
    def __init__(self, table, roster_slots, salary_cap=None):
        """
        Makes Lineup Optimization easy.
        
        Input:
            table - pd.DataFrame
                Contains relevant data on each player. Must include the columns ['Pos','Salary','Team']. If you want
                the resulting table sorted, include a column ['PosNum'] which assigns a numeric value to each position.
            roster_slots - int
                the total number of spaces in the roster
                For example, in NBA 1 through 5 for PG through C.
            salary_cap - float
                The total salary limit of the team, if there is any. Default is None.
        """
        self.nslots = roster_slots
        self.table = table
        self.salary_cap = salary_cap
        
        self.players = table.index.tolist()

        self.prob = pulp.LpProblem('Lineup Optimization', pulp.LpMaximize)
        self.player_vars = pulp.LpVariable.dicts("p", self.players, 0, 1, 'Binary')
        self.rev_player_vars = dict(zip([str(x) for x in self.player_vars.values()], \
                                        self.player_vars.keys()))

        # Objective
        self.prob += pulp.lpSum([self.table.Proj[k]*self.player_vars[k] for k in self.players]), 'Aggregate Projected Score'

        # Salary Cap Constraint
        if self.salary_cap is not None:
            self.prob += pulp.lpSum([self.table.Salary[k]*self.player_vars[k] for k in self.players]) <= self.salary_cap, 'Salary Cap Constraint'

        self.const_tracker = {}
    
    def removeConstraint(self, con_name):
        """
        Remove the constraint with name given by con_name.
        """
        try:
            del self.prob.constraints[con_name]
            return True
        except:
            return False
    
    def addPositionConstraint(self, pos, con_type, bound, con_name=None):
        """
        Constrain the number of position appearances.
        
        Inputs:
            pos - string
                position of player, if multiple, concatenated with '/'
            con_type - string
                {'le','eq','ge'}
            bound - float
                the rhs of the constraint
            con_name - string
                the name to assign the constraint. If not given, defaults to 'Column %s' % column
        
        Returns:
            True if successful, False otherwise
        """
        if con_name is None:
            con_name = 'Pos %s %s' % (con_type,pos)
            try:
                self.const_tracker[con_name] += 1
            except KeyError:
                self.const_tracker[con_name] = 0
            con_name = '%s%d' % (con_name , self.const_tracker[con_name])
        
        pos = pos.split('/')
        if con_type == 'eq':
            self.prob += pulp.lpSum([(self.table['Pos'].loc[k] in pos)*self.player_vars[k] for k in self.players]) == bound, con_name
            return True
        elif con_type == 'le':
            self.prob += pulp.lpSum([(self.table['Pos'].loc[k] in pos)*self.player_vars[k] for k in self.players]) <= bound, con_name
            return True
        elif con_type == 'ge':
            self.prob += pulp.lpSum([(self.table['Pos'].loc[k] in pos)*self.player_vars[k] for k in self.players]) >= bound, con_name
            return True
        raise(Exception('%s is not a valid input for con_type' % con_type))
    
    def solve(self):
        """
        Solves the problem.
        
        Returns:
            status - the solver's status
            lineup - the inputed table subsetted to hold the players in the lineup. Sorts the lineup if a 'PosNum'
                     column was given.
        """
        self.prob.solve()
        
        I = []
        for v in self.prob.variables():
            if v.varValue == 1 and str(v)[0] == 'p':
                player = self.rev_player_vars[v.getName()]
                I.append(player)
        lu = self.table.loc[I]
        try:
            lu = lu.sort(['PosNum','Salary'], ascending=[True,False])
        except Exception, exc:
            pass
        return pulp.LpStatus[self.prob.status], lu
    
    def addTableConstraint(self, column, con_type, bound, con_name=None):
        """
        Add a constraint with the dot product of a column in the inputed table.
        
        Inputs:
            column - string
                name of the column in the table to use
            con_type - string
                {'le','eq','ge'}
            bound - float
                the rhs of the constraint
            con_name - string
                the name to assign the constraint. If not given, defaults to 'Column %s' % column
        
        Returns:
            True if successful, False otherwise
        """
        if con_name is None:
            con_name = 'Column %s' % column
            try:
                self.const_tracker[con_name] += 1
            except KeyError:
                self.const_tracker[con_name] = 0
            con_name = '%s%d' % (con_name , self.const_tracker[con_name])
        
        if con_type == 'eq':
            self.prob += pulp.lpSum([self.table[column].loc[k]*self.player_vars[k] for k in self.players]) == bound, con_name
            return True
        elif con_type == 'le':
            self.prob += pulp.lpSum([self.table[column].loc[k]*self.player_vars[k] for k in self.players]) <= bound, con_name
            return True
        elif con_type == 'ge':
            self.prob += pulp.lpSum([self.table[column].loc[k]*self.player_vars[k] for k in self.players]) >= bound, con_name
            return True
        raise(Exception('%s is not a valid input for con_type' % con_type))
    
    def addTeamLimitConstraint(self, teams, con_type, bound):
        """
        Constrain the number of players from the combination of the teams. Multiple teams are entered
        delimited by a '/' as in 'CLE/DET'
        
        Inputs:
            teams - string
                teams to be constrained delimited by '/'
        
        see addColumnConstraint
        """
        teams = teams.split('/')
        con_name = 'Team %s Limit %s' % (teams, con_type)
        try:
            self.const_tracker[con_name] += 1
        except KeyError:
            self.const_tracker[con_name] = 0
        con_name = '%s%d' % (con_name , self.const_tracker[con_name])
        if con_type == 'eq':
            self.prob += pulp.lpSum([(self.table.loc[k]['Team'] in teams)*self.player_vars[k] for k in self.players]) == bound, con_name
            return True
        elif con_type == 'le':
            self.prob += pulp.lpSum([(self.table.loc[k]['Team'] in teams)*self.player_vars[k] for k in self.players]) <= bound, con_name
            return True
        elif con_type == 'ge':
            self.prob += pulp.lpSum([(self.table.loc[k]['Team'] in teams)*self.player_vars[k] for k in self.players]) >= bound, con_name
            return True
        raise(Exception('%s is not a valid input for con_type' % con_type))
    
    def addCustomConstraint(self, func, con_type, bound, con_name=None):
        """
        Add a custom constraint which results from passing the inputed table into func. This is a more
        general version of addColumnConstraints.
        
        Inputs:
            func - function
                Takes self.table as an input and returns a pd.Series with indices matching that of self.table
        
        see addColumnConstraints
        """
        series = func(self.table)
        
        if con_name is None:
            con_name = 'Custom Func %d'
            try:
                self.const_tracker[con_name] += 1
            except KeyError:
                self.const_tracker[con_name] = 0
            con_name = '%s%d' % (con_name , self.const_tracker[con_name])
        
        return self.addSeriesConstraint(series, con_type, bound, con_name)
    
    def addTableColumns(self, series, names):
        """
        Add the series to self.table with the corresponding column names.
        """
        for serie, name in zip(series, names):
            self.table[name] = serie
        return True
    
    def addSeriesConstraint(self, series, con_type, bound, con_name=None):
        """
        Add a custom constraint which results from passing the inputed table into func. This is a more
        general version of addColumnConstraints.
        
        Inputs:
            func - function
                Takes self.table as an input and returns a pd.Series with indices matching that of self.table
        
        see addColumnConstraints
        """
        if con_name is None:
            con_name = 'Custom Func %d'
            try:
                self.const_tracker[con_name] += 1
            except KeyError:
                self.const_tracker[con_name] = 0
            con_name = '%s%d' % (con_name , self.const_tracker[con_name])
        if con_type == 'eq':
            self.prob += pulp.lpSum([series.loc[k]*self.player_vars[k] for k in self.players]) == bound, con_name
            return True
        elif con_type == 'le':
            self.prob += pulp.lpSum([series.loc[k]*self.player_vars[k] for k in self.players]) <= bound, con_name
            return True
        elif con_type == 'ge':
            self.prob += pulp.lpSum([series.loc[k]*self.player_vars[k] for k in self.players]) >= bound, con_name
            return True
        raise(Exception('%s is not a valid input for con_type' % con_type))
    
    def updateObjective(self, series):
        """
        Replace the current objective function with the one specified by series.
        """
        self.prob.setObjective(pulp.lpSum([series[k]*self.player_vars[k] for k in self.players]))
    
    def disallowLineup(self, lineup):
        """
        Take a lineup DataFrame (subset of self.table) and don't allow this specific lineup.
        """
        players = lineup.index.tolist()
        con_name = 'BlockLineup %s' % str(players)
        self.prob += pulp.lpSum([(k in players)*self.player_vars[k] for k in self.players]) <= self.nslots-1, con_name
    
    def addPlayerConstraint(self, players, con_type, bound, con_name=None):
        """
        Make a constraint based on appearances of players.
        
        Inputs:
            players - string
                the names of players as found in the index of self.table delimited by '/'
        
        see addColumnConstraints
        """
        if con_name is None:
            con_name = 'Players %s %%d' % (players)
            try:
                self.const_tracker[con_name] += 1
            except KeyError:
                self.const_tracker[con_name] = 0
            con_name = '%s%d' % (con_name , self.const_tracker[con_name])
        players = players.split('/')
        if con_type == 'eq':
            self.prob += pulp.lpSum([self.player_vars[k] for k in players]) == bound, con_name
            return True
        elif con_type == 'le':
            self.prob += pulp.lpSum([self.player_vars[k] for k in players]) <= bound, con_name
            return True
        elif con_type == 'ge':
            self.prob += pulp.lpSum([self.player_vars[k] for k in players]) >= bound, con_name
            return True
        raise(Exception('%s is not a valid input for con_type' % con_type))

####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################

class DNormLineupOptimizer(object):
    def __init__(self, table, roster_slots, Gamma, salary_cap=None):
        """
        Makes Robust Lineup Optimization easy using the D-Norm with parameter Gamma.
        
        U = {x | xj = xj^ - (xj^-xj_)wj for all j, wj in [0,1], sum(wj) <= Gamma}
        
        Input:
            table - pd.DataFrame
                Contains relevant data on each player. Must include the columns ['Proj','LB','UB','Pos','Salary','Team']. If you want
                the resulting table sorted, include a column ['PosNum'] which assigns a numeric value to each position.
                For example, in NBA 1 through 5 for PG through C.
            roster_slots - int
                the total number of spaces in the roster
            salary_cap - float
                The total salary limit of the team, if there is any. Default is None.
            Gamma - flaot
                The D-Norm cardinality constraint
        """
        self.nslots = roster_slots
        self.table = table
        self.salary_cap = salary_cap
        
        self.players = table.index.tolist()
        
        n = len(self.players)

        self.prob = pulp.LpProblem('Lineup Optimization', pulp.LpMaximize)
        self.player_vars = pulp.LpVariable.dicts("p", self.players, 0, 1, 'Binary')
        self.rev_player_vars = dict(zip([str(x) for x in self.player_vars.values()], \
                                        self.player_vars.keys()))
        
        y = [pulp.LpVariable('y%d'%d, lowBound=0, cat='Continuous') for d in range(4*n+1)]
        
        # Objective
        self.prob += pulp.lpSum([self.table.UB.loc[k]*self.player_vars[k] for k in self.players]) \
                   - pulp.lpSum(y[2*n:3*n]) - Gamma*y[-1], 'Objective'
        
        # Robust Dual Equality
        for k in xrange(n):
            self.prob += self.player_vars[self.players[k]] + y[k] - y[n+k] == 0, 'RobustEqX_%d' % k
        delta = self.table.UB - self.table.LB
        for k in xrange(n):
            dlta = delta.loc[self.players[k]]
            try:
                self.prob += dlta*y[k] - dlta*y[n+k] + y[2*n+k] - y[3*n+k] + y[-1] == 0, \
                    'RobustDualEq0_%d' % k
            except Exception, exc:
                print exc.message
                print dlta
                raise(exc)

        # Salary Cap Constraint
        if self.salary_cap is not None:
            self.prob += pulp.lpSum([self.table.Salary[k]*self.player_vars[k] for k in self.players]) <= self.salary_cap, 'Salary Cap Constraint'

        self.const_tracker = {}
    
    def addPositionConstraint(self, pos, con_type, bound, con_name=None):
        """
        Constrain the number of position appearances.
        
        Inputs:
            pos - string
                position of player, if multiple, concatenated with '/'
            con_type - string
                {'le','eq','ge'}
            bound - float
                the rhs of the constraint
            con_name - string
                the name to assign the constraint. If not given, defaults to 'Column %s' % column
        
        Returns:
            True if successful, False otherwise
        """
        if con_name is None:
            con_name = 'Pos %s %s' % (con_type,pos)
            try:
                self.const_tracker[con_name] += 1
            except KeyError:
                self.const_tracker[con_name] = 0
            con_name = '%s%d' % (con_name , self.const_tracker[con_name])
        
        pos = pos.split('/')
        if con_type == 'eq':
            self.prob += pulp.lpSum([(self.table['Pos'].loc[k] in pos)*self.player_vars[k] for k in self.players]) == bound, con_name
            return True
        elif con_type == 'le':
            self.prob += pulp.lpSum([(self.table['Pos'].loc[k] in pos)*self.player_vars[k] for k in self.players]) <= bound, con_name
            return True
        elif con_type == 'ge':
            self.prob += pulp.lpSum([(self.table['Pos'].loc[k] in pos)*self.player_vars[k] for k in self.players]) >= bound, con_name
            return True
        raise(Exception('%s is not a valid input for con_type' % con_type))
    
    def solve(self):
        """
        Solves the problem.
        
        Returns:
            status - the solver's status
            lineup - the inputed table subsetted to hold the players in the lineup. Sorts the lineup if a 'PosNum'
                     column was given.
        """
        self.prob.solve()
        
        I = []
        for k, v in enumerate(self.prob.variables()):
            if v.varValue == 1 and str(v)[0] == 'p':
                player = self.rev_player_vars[v.getName()]
                I.append(player)
        lu = self.table.loc[I]
        try:
            lu = lu.sort(['PosNum','Salary'], ascending=[True,False])
        except Exception, exc:
            pass
        return pulp.LpStatus[self.prob.status], lu
    
    def addTableConstraint(self, column, con_type, bound, con_name=None):
        """
        Add a constraint with the dot product of a column in the inputed table.
        
        Inputs:
            column - string
                name of the column in the table to use
            con_type - string
                {'le','eq','ge'}
            bound - float
                the rhs of the constraint
            con_name - string
                the name to assign the constraint. If not given, defaults to 'Column %s' % column
        
        Returns:
            True if successful, False otherwise
        """
        if con_name is None:
            con_name = 'Column %s' % column
            try:
                self.const_tracker[con_name] += 1
            except KeyError:
                self.const_tracker[con_name] = 0
            con_name = '%s%d' % (con_name , self.const_tracker[con_name])
        
        if con_type == 'eq':
            self.prob += pulp.lpSum([self.table[column].loc[k]*self.player_vars[k] for k in self.players]) == bound, con_name
            return True
        elif con_type == 'le':
            self.prob += pulp.lpSum([self.table[column].loc[k]*self.player_vars[k] for k in self.players]) <= bound, con_name
            return True
        elif con_type == 'ge':
            self.prob += pulp.lpSum([self.table[column].loc[k]*self.player_vars[k] for k in self.players]) >= bound, con_name
            return True
        raise(Exception('%s is not a valid input for con_type' % con_type))
    
    def addTeamLimitConstraint(self, teams, con_type, bound):
        """
        Constrain the number of players from the combination of the teams. Multiple teams are entered
        delimited by a '/' as in 'CLE/DET'
        
        Inputs:
            teams - string
                teams to be constrained delimited by '/'
        
        see addColumnConstraint
        """
        teams = teams.split('/')
        con_name = 'Team %s Limit %s' % (teams, con_type)
        try:
            self.const_tracker[con_name] += 1
        except KeyError:
            self.const_tracker[con_name] = 0
        con_name = '%s%d' % (con_name , self.const_tracker[con_name])
        if con_type == 'eq':
            self.prob += pulp.lpSum([(self.table.loc[k]['Team'] in teams)*self.player_vars[k] for k in self.players]) == bound, con_name
            return True
        elif con_type == 'le':
            self.prob += pulp.lpSum([(self.table.loc[k]['Team'] in teams)*self.player_vars[k] for k in self.players]) <= bound, con_name
            return True
        elif con_type == 'ge':
            self.prob += pulp.lpSum([(self.table.loc[k]['Team'] in teams)*self.player_vars[k] for k in self.players]) >= bound, con_name
            return True
        raise(Exception('%s is not a valid input for con_type' % con_type))
    
    def addCustomConstraint(self, func, con_type, bound, con_name=None):
        """
        Add a custom constraint which results from passing the inputed table into func. This is a more
        general version of addColumnConstraints.
        
        Inputs:
            func - function
                Takes self.table as an input and returns a pd.Series with indices matching that of self.table
        
        see addColumnConstraints
        """
        series = func(self.table)
        
        if con_name is None:
            con_name = 'Custom Func %d'
            try:
                self.const_tracker[con_name] += 1
            except KeyError:
                self.const_tracker[con_name] = 0
            con_name = '%s%d' % (con_name , self.const_tracker[con_name])
        
        return self.addSeriesConstraint(series, con_type, bound, con_name)
    
    def addTableColumns(self, series, names):
        """
        Add the series to self.table with the corresponding column names.
        """
        for serie, name in zip(series, names):
            self.table[name] = serie
        return True
    
    def addSeriesConstraint(self, series, con_type, bound, con_name=None):
        """
        Add a custom constraint which results from passing the inputed table into func. This is a more
        general version of addColumnConstraints.
        
        Inputs:
            func - function
                Takes self.table as an input and returns a pd.Series with indices matching that of self.table
        
        see addColumnConstraints
        """
        if con_name is None:
            con_name = 'Custom Func %d'
            try:
                self.const_tracker[con_name] += 1
            except KeyError:
                self.const_tracker[con_name] = 0
            con_name = '%s%d' % (con_name , self.const_tracker[con_name])
        if con_type == 'eq':
            self.prob += pulp.lpSum([series.loc[k]*self.player_vars[k] for k in self.players]) == bound, con_name
            return True
        elif con_type == 'le':
            self.prob += pulp.lpSum([series.loc[k]*self.player_vars[k] for k in self.players]) <= bound, con_name
            return True
        elif con_type == 'ge':
            self.prob += pulp.lpSum([series.loc[k]*self.player_vars[k] for k in self.players]) >= bound, con_name
            return True
        raise(Exception('%s is not a valid input for con_type' % con_type))
    
    def updateObjective(self, series):
        """
        Replace the current objective function with the one specified by series.
        """
        self.prob.setObjective(pulp.lpSum([series[k]*self.player_vars[k] for k in self.players]))
    
    def disallowLineup(self, lineup):
        """
        Take a lineup DataFrame (subset of self.table) and don't allow this specific lineup.
        """
        players = lineup.index.tolist()
        con_name = 'BlockLineup %s' % str(players)
        self.prob += pulp.lpSum([(k in players)*self.player_vars[k] for k in self.players]) <= self.nslots-1, con_name
    
    def addPlayerConstraint(self, players, con_type, bound, con_name=None):
        """
        Make a constraint based on appearances of players.
        
        Inputs:
            players - string
                the names of players as found in the index of self.table delimited by '/'
        
        see addColumnConstraints
        """
        if con_name is None:
            con_name = 'Players %s %%d' % (players)
            try:
                self.const_tracker[con_name] += 1
            except KeyError:
                self.const_tracker[con_name] = 0
            con_name = '%s%d' % (con_name , self.const_tracker[con_name])
        players = players.split('/')
        if con_type == 'eq':
            self.prob += pulp.lpSum([self.player_vars[k] for k in players]) == bound, con_name
            return True
        elif con_type == 'le':
            self.prob += pulp.lpSum([self.player_vars[k] for k in players]) <= bound, con_name
            return True
        elif con_type == 'ge':
            self.prob += pulp.lpSum([self.player_vars[k] for k in players]) >= bound, con_name
            return True
        raise(Exception('%s is not a valid input for con_type' % con_type))
        
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################

class MultiLineupOptimizer(object):
    def __init__(self, table, roster_slots, nlineups=1, salary_cap=None):
        """
        Makes Lineup Optimization easy. Set multiple lineups simultaneously. Maximizes the sum of
        the projected score of all of the lineups.
        
        Note for use:
            If you want the top N highest projected lineups, just use the LineupOptimizer in a for-loop
            which calls disallowLineup(lu) on each outputed optimal lineup.
            
            This is only beneficial when there are constraints on the number of appearances on all the
            players, otherwise it will just spit out the same lineup several times. For example, if we
            want 3 lineups, and we want each player to appear a maximum of two times, this is the
            appropriate optimizer to use.
        
        Input:
            table - pd.DataFrame
                Contains relevant data on each player. Must include the columns ['Pos','Salary','Team']. If you want
                the resulting table sorted, include a column ['PosNum'] which assigns a numeric value to each position.
            roster_slots - int
                the total number of spaces in the roster
                For example, in NBA 1 through 5 for PG through C.
            nlineups - int
                the number of lineups to set simultaneously
            salary_cap - float
                The total salary limit of the team, if there is any. Default is None.
        """
        self.nslots = roster_slots
        self.table = table
        self.salary_cap = salary_cap
        self.nlineups = nlineups
        
        self.players = table.index.tolist()

        self.prob = pulp.LpProblem('Lineup Optimization', pulp.LpMaximize)
        self.player_vars = []
        for k in xrange(self.nlineups):
            self.player_vars.append(pulp.LpVariable.dicts("p%d" % k, self.players, 0, 1, 'Binary'))
        self.rev_player_vars = dict(zip([str(x) for y in self.player_vars for x in y.values()], \
                                        [x for y in self.player_vars for x in y.keys()]))

        # Objective
        self.prob += pulp.lpSum([self.table.Proj[k]*self.player_vars[j][k] for k in self.players for j in range(self.nlineups)]), 'Aggregate Projected Score'

        # Salary Cap Constraint
        if self.salary_cap is not None:
            for j in xrange(self.nlineups):
                self.prob += pulp.lpSum([self.table.Salary[k]*self.player_vars[j][k] for k in self.players]) <= self.salary_cap, 'Salary Cap Constraint %d' % j

        self.const_tracker = {}
    
    def addPositionConstraint(self, pos, con_type, bound, lineup=0, con_name=None):
        """
        Constrain the number of position appearances.
        
        Inputs:
            pos - string
                position of player, if multiple, concatenated with '/'
            con_type - string
                {'le','eq','ge'}
            bound - float
                the rhs of the constraint
            lineup - int
                index of the lineup to set the constraint to.
            con_name - string
                the name to assign the constraint. If not given, defaults to 'Column %s' % column
        
        Returns:
            True if successful, False otherwise
        """
        if con_name is None:
            con_name = '%d Pos %s %s' % (lineup,con_type,pos)
            try:
                self.const_tracker[con_name] += 1
            except KeyError:
                self.const_tracker[con_name] = 0
            con_name = '%s%d' % (con_name , self.const_tracker[con_name])
        
        pos = pos.split('/')
        if con_type == 'eq':
            self.prob += pulp.lpSum([(self.table['Pos'].loc[k] in pos)*self.player_vars[lineup][k] for k in self.players]) == bound, con_name
            return True
        elif con_type == 'le':
            self.prob += pulp.lpSum([(self.table['Pos'].loc[k] in pos)*self.player_vars[lineup][k] for k in self.players]) <= bound, con_name
            return True
        elif con_type == 'ge':
            self.prob += pulp.lpSum([(self.table['Pos'].loc[k] in pos)*self.player_vars[lineup][k] for k in self.players]) >= bound, con_name
            return True
        raise(Exception('%s is not a valid input for con_type' % con_type))
    
    def solve(self):
        """
        Solves the problem.
        
        Returns:
            status - the solver's status
            lineup - the inputed table subsetted to hold the players in the lineup. Sorts the lineup if a 'PosNum'
                     column was given.
        """
        self.prob.solve()
        
        lus = []
        for k in xrange(self.nlineups):
            I = []
            for v in self.prob.variables()[k*len(self.players):(k+1)*len(self.players)]:
                if v.varValue == 1 and str(v)[0] == 'p':
                    player = self.rev_player_vars[v.getName()]
                    I.append(player)
            lu = self.table.loc[I]
            try:
                lu = lu.sort(['PosNum','Salary'], ascending=[True,False])
            except Exception, exc:
                pass
            lus.append(lu)
        return pulp.LpStatus[self.prob.status], lus
    
    def addTableConstraint(self, column, con_type, bound, lineup=0, con_name=None):
        """
        Add a constraint with the dot product of a column in the inputed table.
        
        Inputs:
            column - string
                name of the column in the table to use
            con_type - string
                {'le','eq','ge'}
            bound - float
                the rhs of the constraint
            lineup - int
                index of the lineup to apply the constraint to
            con_name - string
                the name to assign the constraint. If not given, defaults to 'Column %s' % column
        
        Returns:
            True if successful, False otherwise
        """
        if con_name is None:
            con_name = '%d Column %s' % (lineup,column)
            try:
                self.const_tracker[con_name] += 1
            except KeyError:
                self.const_tracker[con_name] = 0
            con_name = '%s%d' % (con_name , self.const_tracker[con_name])
        
        if con_type == 'eq':
            self.prob += pulp.lpSum([self.table[column].loc[k]*self.player_vars[lineup][k] for k in self.players]) == bound, con_name
            return True
        elif con_type == 'le':
            self.prob += pulp.lpSum([self.table[column].loc[k]*self.player_vars[lineup][k] for k in self.players]) <= bound, con_name
            return True
        elif con_type == 'ge':
            self.prob += pulp.lpSum([self.table[column].loc[k]*self.player_vars[lineup][k] for k in self.players]) >= bound, con_name
            return True
        raise(Exception('%s is not a valid input for con_type' % con_type))
    
    def addTeamLimitConstraint(self, teams, con_type, bound, lineup=0):
        """
        Constrain the number of players from the combination of the teams. Multiple teams are entered
        delimited by a '/' as in 'CLE/DET'
        
        Inputs:
            teams - string
                teams to be constrained delimited by '/'
        
        see addColumnConstraint
        """
        teams = teams.split('/')
        con_name = '%d Team %s Limit %s' % (lineup, teams, con_type)
        try:
            self.const_tracker[con_name] += 1
        except KeyError:
            self.const_tracker[con_name] = 0
        con_name = '%s%d' % (con_name , self.const_tracker[con_name])
        if con_type == 'eq':
            self.prob += pulp.lpSum([(self.table.loc[k]['Team'] in teams)*self.player_vars[lineup][k] for k in self.players]) == bound, con_name
            return True
        elif con_type == 'le':
            self.prob += pulp.lpSum([(self.table.loc[k]['Team'] in teams)*self.player_vars[lineup][k] for k in self.players]) <= bound, con_name
            return True
        elif con_type == 'ge':
            self.prob += pulp.lpSum([(self.table.loc[k]['Team'] in teams)*self.player_vars[lineup][k] for k in self.players]) >= bound, con_name
            return True
        raise(Exception('%s is not a valid input for con_type' % con_type))
    
    def addCustomConstraint(self, func, con_type, bound, lineup=0, con_name=None):
        """
        Add a custom constraint which results from passing the inputed table into func. This is a more
        general version of addColumnConstraints.
        
        Inputs:
            func - function
                Takes self.table as an input and returns a pd.Series with indices matching that of self.table
        
        see addColumnConstraints
        """
        series = func(self.table)
        
        if con_name is None:
            con_name = '%d Custom Func %%d' % lineup
            try:
                self.const_tracker[con_name] += 1
            except KeyError:
                self.const_tracker[con_name] = 0
            con_name = '%s%d' % (con_name , self.const_tracker[con_name])
        
        return self.addSeriesConstraint(series, con_type, bound, lineup, con_name)
    
    def addTableColumns(self, series, names):
        """
        Add the series to self.table with the corresponding column names.
        """
        for serie, name in zip(series, names):
            self.table[name] = serie
        return True
    
    def addSeriesConstraint(self, series, con_type, bound, lineup=0, con_name=None):
        """
        Add a custom constraint which results from passing the inputed table into func. This is a more
        general version of addColumnConstraints.
        
        Inputs:
            func - function
                Takes self.table as an input and returns a pd.Series with indices matching that of self.table
        
        see addColumnConstraints
        """
        if con_name is None:
            con_name = '%d Custom Func %%d' % lineup
            try:
                self.const_tracker[con_name] += 1
            except KeyError:
                self.const_tracker[con_name] = 0
            con_name = '%s%d' % (con_name , self.const_tracker[con_name])
        if con_type == 'eq':
            self.prob += pulp.lpSum([series.loc[k]*self.player_vars[lineup][k] for k in self.players]) == bound, con_name
            return True
        elif con_type == 'le':
            self.prob += pulp.lpSum([series.loc[k]*self.player_vars[lineup][k] for k in self.players]) <= bound, con_name
            return True
        elif con_type == 'ge':
            self.prob += pulp.lpSum([series.loc[k]*self.player_vars[lineup][k] for k in self.players]) >= bound, con_name
            return True
        raise(Exception('%s is not a valid input for con_type' % con_type))
    
    def updateObjective(self, series):
        """
        Replace the current objective function with the one specified by series.
        """
        self.prob.setObjective(pulp.lpSum([series[k]*self.player_vars[j][k] for k in self.players for j in range(self.nlineups)]))
    
    def disallowLineup(self, lineup):
        """
        Take a lineup DataFrame (subset of self.table) and don't allow this specific lineup.
        """
        players = lineup.index.tolist()
        con_name = '%%d BlockLineup %s' % str(players)
        for j in xrange(self.nlineups):
            self.prob += pulp.lpSum([(k in players)*self.player_vars[k] for k in self.players]) <= self.nslots-1, con_name % j
    
    def addPlayerConstraint(self, players, con_type, bound, lineup=0, con_name=None):
        """
        Make a constraint based on appearances of players.
        
        Inputs:
            players - string
                the names of players as found in the index of self.table delimited by '/'
            lineup - int
                if lineup >= 0, then it is the index of the lineup to which the constraint should
                be added. if lineup==-1, then the aggregate of all lineups will be constrained.
                e.g. addPlayerConstraint('LeBron James/Anthony Davis', 'eq', 3, lineup=-1,
                con_name='LJ/AD eq 3') will make it so that in total of all the lineups, 
                LeBron James + Anthony Davis will occur exactly 3 times.
        
        see addColumnConstraints
        """
        if con_name is None:
            con_name = '%d Players %s %%d' % (lineup, players)
            try:
                self.const_tracker[con_name] += 1
            except KeyError:
                self.const_tracker[con_name] = 0
            con_name = '%s%d' % (con_name , self.const_tracker[con_name])
        players = players.split('/')
        
        if lineup >= 0:
            if con_type == 'eq':
                self.prob += pulp.lpSum([self.player_vars[lineup][k] for k in players]) == bound,\
                    con_name
                return True
            elif con_type == 'le':
                self.prob += pulp.lpSum([self.player_vars[lineup][k] for k in players]) <= bound,\
                    con_name
                return True
            elif con_type == 'ge':
                self.prob += pulp.lpSum([self.player_vars[lineup][k] for k in players]) >= bound,\
                    con_name
                return True
        elif lineup == -1:
            if con_type == 'eq':
                self.prob += pulp.lpSum([self.player_vars[j][k] for k in players 
                                         for j in range(self.nlineups)]) == bound, con_name
                return True
            elif con_type == 'le':
                self.prob += pulp.lpSum([self.player_vars[j][k] for k in players 
                                         for j in range(self.nlineups)]) <= bound, con_name
                return True
            elif con_type == 'ge':
                self.prob += pulp.lpSum([self.player_vars[j][k] for k in players 
                                         for j in range(self.nlineups)]) >= bound, con_name
                return True
        raise(Exception('%s is not a valid input for con_type' % con_type))
    
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################

class RobustMultiLineupOptimizer(object):
    def __init__(self, table, roster_slots, Gamma, nlineups=1, salary_cap=None):
        """
        Makes Lineup Optimization easy. Set multiple lineups simultaneously. Maximizes the sum of
        the projected score of all of the lineups. Uses the D-Norm from RobustLineupOptimizer
        
        Note for use:
            If you want the top N highest projected lineups, just use the LineupOptimizer in a for-loop
            which calls disallowLineup(lu) on each outputed optimal lineup.
            
            This is only beneficial when there are constraints on the number of appearances on all the
            players, otherwise it will just spit out the same lineup several times. For example, if we
            want 3 lineups, and we want each player to appear a maximum of two times, this is the
            appropriate optimizer to use.
        
        Input:
            table - pd.DataFrame
                Contains relevant data on each player. Must include the columns ['Pos','Salary','Team']. If you want
                the resulting table sorted, include a column ['PosNum'] which assigns a numeric value to each position.
            roster_slots - int
                the total number of spaces in the roster
                For example, in NBA 1 through 5 for PG through C.
            Gamma - float
                Robustness parameter. See RobustLineupOptimizer for more information.
            nlineups - int
                the number of lineups to set simultaneously
            salary_cap - float
                The total salary limit of the team, if there is any. Default is None.
        """
        self.nslots = roster_slots
        self.table = table
        self.salary_cap = salary_cap
        self.nlineups = nlineups
        
        self.players = table.index.tolist()
        n = len(self.players)

        self.prob = pulp.LpProblem('Lineup Optimization', pulp.LpMaximize)
        self.player_vars = []
        for k in xrange(self.nlineups):
            self.player_vars.append(pulp.LpVariable.dicts("p%d" % k, self.players, 0, 1, 'Binary'))
        self.rev_player_vars = dict(zip([str(x) for y in self.player_vars for x in y.values()], \
                                        [x for y in self.player_vars for x in y.keys()]))

        y = [pulp.LpVariable('y%d'%d, lowBound=0, cat='Continuous') for d in range(4*n+1)]
        
        # Objective
        self.prob += pulp.lpSum([self.table.UB.loc[k]*self.player_vars[j][k] for k in self.players for j in range(self.nlineups)]) \
                   - pulp.lpSum(y[2*n:3*n]) - Gamma*y[-1], 'Objective'
        
        # Robust Dual Equality
        for k in xrange(n):
            self.prob += pulp.lpSum([self.player_vars[j][self.players[k]] 
                           for j in range(self.nlineups)]) + y[k] - y[n+k] == 0, 'RobustEqX_%d' % k
        delta = self.table.UB - self.table.LB
        for k in xrange(n):
            dlta = delta.loc[self.players[k]]
            try:
                self.prob += dlta*y[k] - dlta*y[n+k] + y[2*n+k] - y[3*n+k] + y[-1] == 0, \
                    'RobustDualEq0_%d' % k
            except Exception, exc:
                print exc.message
                print dlta
                raise(exc)

        # Salary Cap Constraint
        if self.salary_cap is not None:
            for j in xrange(self.nlineups):
                self.prob += pulp.lpSum([self.table.Salary[k]*self.player_vars[j][k] for k in self.players]) <= self.salary_cap, 'Salary Cap Constraint %d' % j

        self.const_tracker = {}
    
    def addPositionConstraint(self, pos, con_type, bound, lineup=0, con_name=None):
        """
        Constrain the number of position appearances.
        
        Inputs:
            pos - string
                position of player, if multiple, concatenated with '/'
            con_type - string
                {'le','eq','ge'}
            bound - float
                the rhs of the constraint
            lineup - int
                index of the lineup to set the constraint to.
            con_name - string
                the name to assign the constraint. If not given, defaults to 'Column %s' % column
        
        Returns:
            True if successful, False otherwise
        """
        if con_name is None:
            con_name = '%d Pos %s %s' % (lineup,con_type,pos)
            try:
                self.const_tracker[con_name] += 1
            except KeyError:
                self.const_tracker[con_name] = 0
            con_name = '%s%d' % (con_name , self.const_tracker[con_name])
        
        pos = pos.split('/')
        if con_type == 'eq':
            self.prob += pulp.lpSum([(self.table['Pos'].loc[k] in pos)*self.player_vars[lineup][k] for k in self.players]) == bound, con_name
            return True
        elif con_type == 'le':
            self.prob += pulp.lpSum([(self.table['Pos'].loc[k] in pos)*self.player_vars[lineup][k] for k in self.players]) <= bound, con_name
            return True
        elif con_type == 'ge':
            self.prob += pulp.lpSum([(self.table['Pos'].loc[k] in pos)*self.player_vars[lineup][k] for k in self.players]) >= bound, con_name
            return True
        raise(Exception('%s is not a valid input for con_type' % con_type))
    
    def solve(self):
        """
        Solves the problem.
        
        Returns:
            status - the solver's status
            lineup - the inputed table subsetted to hold the players in the lineup. Sorts the lineup if a 'PosNum'
                     column was given.
        """
        self.prob.solve()
        
        lus = []
        for k in xrange(self.nlineups):
            I = []
            for v in self.prob.variables()[k*len(self.players):(k+1)*len(self.players)]:
                if v.varValue == 1 and str(v)[0] == 'p':
                    player = self.rev_player_vars[v.getName()]
                    I.append(player)
            lu = self.table.loc[I]
            try:
                lu = lu.sort(['PosNum','Salary'], ascending=[True,False])
            except Exception, exc:
                pass
            lus.append(lu)
        return pulp.LpStatus[self.prob.status], lus
    
    def addTableConstraint(self, column, con_type, bound, lineup=0, con_name=None):
        """
        Add a constraint with the dot product of a column in the inputed table.
        
        Inputs:
            column - string
                name of the column in the table to use
            con_type - string
                {'le','eq','ge'}
            bound - float
                the rhs of the constraint
            lineup - int
                index of the lineup to apply the constraint to
            con_name - string
                the name to assign the constraint. If not given, defaults to 'Column %s' % column
        
        Returns:
            True if successful, False otherwise
        """
        if con_name is None:
            con_name = '%d Column %s' % (lineup,column)
            try:
                self.const_tracker[con_name] += 1
            except KeyError:
                self.const_tracker[con_name] = 0
            con_name = '%s%d' % (con_name , self.const_tracker[con_name])
        
        if con_type == 'eq':
            self.prob += pulp.lpSum([self.table[column].loc[k]*self.player_vars[lineup][k] for k in self.players]) == bound, con_name
            return True
        elif con_type == 'le':
            self.prob += pulp.lpSum([self.table[column].loc[k]*self.player_vars[lineup][k] for k in self.players]) <= bound, con_name
            return True
        elif con_type == 'ge':
            self.prob += pulp.lpSum([self.table[column].loc[k]*self.player_vars[lineup][k] for k in self.players]) >= bound, con_name
            return True
        raise(Exception('%s is not a valid input for con_type' % con_type))
    
    def addTeamLimitConstraint(self, teams, con_type, bound, lineup=0):
        """
        Constrain the number of players from the combination of the teams. Multiple teams are entered
        delimited by a '/' as in 'CLE/DET'
        
        Inputs:
            teams - string
                teams to be constrained delimited by '/'
        
        see addColumnConstraint
        """
        teams = teams.split('/')
        con_name = '%d Team %s Limit %s' % (lineup, teams, con_type)
        try:
            self.const_tracker[con_name] += 1
        except KeyError:
            self.const_tracker[con_name] = 0
        con_name = '%s%d' % (con_name , self.const_tracker[con_name])
        if con_type == 'eq':
            self.prob += pulp.lpSum([(self.table.loc[k]['Team'] in teams)*self.player_vars[lineup][k] for k in self.players]) == bound, con_name
            return True
        elif con_type == 'le':
            self.prob += pulp.lpSum([(self.table.loc[k]['Team'] in teams)*self.player_vars[lineup][k] for k in self.players]) <= bound, con_name
            return True
        elif con_type == 'ge':
            self.prob += pulp.lpSum([(self.table.loc[k]['Team'] in teams)*self.player_vars[lineup][k] for k in self.players]) >= bound, con_name
            return True
        raise(Exception('%s is not a valid input for con_type' % con_type))
    
    def addCustomConstraint(self, func, con_type, bound, lineup=0, con_name=None):
        """
        Add a custom constraint which results from passing the inputed table into func. This is a more
        general version of addColumnConstraints.
        
        Inputs:
            func - function
                Takes self.table as an input and returns a pd.Series with indices matching that of self.table
        
        see addColumnConstraints
        """
        series = func(self.table)
        
        if con_name is None:
            con_name = '%d Custom Func %%d' % lineup
            try:
                self.const_tracker[con_name] += 1
            except KeyError:
                self.const_tracker[con_name] = 0
            con_name = '%s%d' % (con_name , self.const_tracker[con_name])
        
        return self.addSeriesConstraint(series, con_type, bound, lineup, con_name)
    
    def addTableColumns(self, series, names):
        """
        Add the series to self.table with the corresponding column names.
        """
        for serie, name in zip(series, names):
            self.table[name] = serie
        return True
    
    def addSeriesConstraint(self, series, con_type, bound, lineup=0, con_name=None):
        """
        Add a custom constraint which results from passing the inputed table into func. This is a more
        general version of addColumnConstraints.
        
        Inputs:
            func - function
                Takes self.table as an input and returns a pd.Series with indices matching that of self.table
        
        see addColumnConstraints
        """
        if con_name is None:
            con_name = '%d Custom Func %%d' % lineup
            try:
                self.const_tracker[con_name] += 1
            except KeyError:
                self.const_tracker[con_name] = 0
            con_name = '%s%d' % (con_name , self.const_tracker[con_name])
        if con_type == 'eq':
            self.prob += pulp.lpSum([series.loc[k]*self.player_vars[lineup][k] for k in self.players]) == bound, con_name
            return True
        elif con_type == 'le':
            self.prob += pulp.lpSum([series.loc[k]*self.player_vars[lineup][k] for k in self.players]) <= bound, con_name
            return True
        elif con_type == 'ge':
            self.prob += pulp.lpSum([series.loc[k]*self.player_vars[lineup][k] for k in self.players]) >= bound, con_name
            return True
        raise(Exception('%s is not a valid input for con_type' % con_type))
    
    def updateObjective(self, series):
        """
        Replace the current objective function with the one specified by series.
        """
        self.prob.setObjective(pulp.lpSum([series[k]*self.player_vars[j][k] for k in self.players for j in range(self.nlineups)]))
    
    def disallowLineup(self, lineup):
        """
        Take a lineup DataFrame (subset of self.table) and don't allow this specific lineup.
        """
        players = lineup.index.tolist()
        con_name = '%%d BlockLineup %s' % str(players)
        for j in xrange(self.nlineups):
            self.prob += pulp.lpSum([(k in players)*self.player_vars[k] for k in self.players]) <= self.nslots-1, con_name % j
    
    def addPlayerConstraint(self, players, con_type, bound, lineup=0, con_name=None):
        """
        Make a constraint based on appearances of players.
        
        Inputs:
            players - string
                the names of players as found in the index of self.table delimited by '/'
            lineup - int
                if lineup >= 0, then it is the index of the lineup to which the constraint should
                be added. if lineup==-1, then the aggregate of all lineups will be constrained.
                e.g. addPlayerConstraint('LeBron James/Anthony Davis', 'eq', 3, lineup=-1,
                con_name='LJ/AD eq 3') will make it so that in total of all the lineups, 
                LeBron James + Anthony Davis will occur exactly 3 times.
        
        see addColumnConstraints
        """
        if con_name is None:
            con_name = '%d Players %s %%d' % (lineup, players)
            try:
                self.const_tracker[con_name] += 1
            except KeyError:
                self.const_tracker[con_name] = 0
            con_name = '%s%d' % (con_name , self.const_tracker[con_name])
        players = players.split('/')
        
        if lineup >= 0:
            if con_type == 'eq':
                self.prob += pulp.lpSum([self.player_vars[lineup][k] for k in players]) == bound,\
                    con_name
                return True
            elif con_type == 'le':
                self.prob += pulp.lpSum([self.player_vars[lineup][k] for k in players]) <= bound,\
                    con_name
                return True
            elif con_type == 'ge':
                self.prob += pulp.lpSum([self.player_vars[lineup][k] for k in players]) >= bound,\
                    con_name
                return True
        elif lineup == -1:
            if con_type == 'eq':
                self.prob += pulp.lpSum([self.player_vars[j][k] for k in players 
                                         for j in range(self.nlineups)]) == bound, con_name
                return True
            elif con_type == 'le':
                self.prob += pulp.lpSum([self.player_vars[j][k] for k in players 
                                         for j in range(self.nlineups)]) <= bound, con_name
                return True
            elif con_type == 'ge':
                self.prob += pulp.lpSum([self.player_vars[j][k] for k in players 
                                         for j in range(self.nlineups)]) >= bound, con_name
                return True
        raise(Exception('%s is not a valid input for con_type' % con_type))
