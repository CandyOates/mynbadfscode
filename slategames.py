import argparse
import numpy as np
import pandas as pd
import datetime

ABBR = {
	'SA': 'SAS',
	'GS': 'GSW',
	'PHO': 'PHX',
	'NO': 'NOP',
	'NY': 'NYK',
}

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	infile = 'PlayerLists/pl_fd_nba_%s.csv' % datetime.datetime.now().date()
	parser.add_argument('-i','--infile', type=str, help='fanduel player list file', default=infile)
	parser.add_argument('-l','--slate', type=str, help='name of the slate', default='main')
	parser.add_argument('-s','--site', type=str, help='other sites to include')

	args = parser.parse_args()

	df = pd.read_csv(args.infile)
	games = np.unique(df.Game)

	li = ['FD','DK','YH','DP','DD','FA','FF']

	if args.site is not None:
		li = [args.site]
	with open('slate_games.csv', 'a') as w:
		for site in li:
			s = '%s,%s,%s,%s\n' % (datetime.datetime.now().date(), args.slate, '/'.join(games), site)
			for key, val in ABBR.iteritems():
				s = s.replace(key, val)
			w.write(s)
