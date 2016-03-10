import sys
from os import listdir
import re
import datetime

def run():
	today = datetime.datetime.now().date()

	fnames = []
	patterns = [
		'FanDuel-NBA-.+\.csv',
		'DKSalaries.csv',
		'Yahoo_DF_.+\.csv',
		'draftpot.+\.csv',
		'FantasyAces.+\.csv',
		'snapshot_NBA.+\.csv',
		'contest-salaries\.csv',
	]

	for pattern in patterns:
		fnames.extend(filter(lambda x: re.search(pattern, x) is not None, listdir('.')))
	
	fname_map = {}
	for fn in fnames:
		if 'FanDuel' in fn:
			site = 'fd'
		elif 'DK' in fn:
			site = 'dk'
		elif 'Yahoo' in fn:
			site = 'yh'
		elif 'draftpot' in fn:
			site = 'dp'
		elif 'FantasyAces' in fn:
			site = 'fa'
		elif 'snapshot' in fn:
			site = 'dd'
		elif 'contest-salaries.csv' == fn:
			site = 'ff'
		else:
			raise Exception('Invalid File: %s'%fn)
		fname_map[fn] = 'pl_%s_nba_%s.csv' % (site, today)
	
	for fn in fnames:
		print 'Editing %s' % fn,
		with open(fn, 'r') as reader:
			with open(fname_map[fn], 'w') as writer:
				for line_num, line in enumerate(reader):
					outline = line.replace('"','')
					if fname_map[fn][3:5] == 'fa' and line_num > 0:
						spl = outline.split(',')
						outline = ','.join(spl[:-1]) + spl[-1]
					writer.write(outline)
		print '> Done %s' % fname_map[fn]

if __name__ == '__main__':
	run()
