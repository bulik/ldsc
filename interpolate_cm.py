from __future__ import division
import numpy as np
import argparse

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('--bim', default=None, type=str, 
		help='.bim file prefix (.bim is appended)')
	parser.add_argument('--map', default=None, type=str, 
		help='SHAPEIT map file')
	parser.add_argument('--out', default=None, type=str, 
		help='Output file prefix (.bim is appended)')
	args = parser.parse_args()
	
	if args.bim is None or args.map is None or args.out is None:
		raise ValueError('Must set all of --bim, --map, --out.')
	
	args.bim += '.bim'
	print 'Reading {B}'.format(B=args.bim)
	bim_array = np.loadtxt(args.bim)
	print 'Reading {M}'.format(M=rgs.map)
	map_array = np.loadtxt(args.map, skiprows=1, useocols=[0,2],
		dtype={('BP', float), ('CM', float)})
	if not np.all(np.diff(xp) > 0):
		print 'Map file not sorted by physical coordinate. Sorting.'
		map_array = map_array[map_array[:,0].argsort()]		
		
	min_cm = np.min(map_array[1])
	max_cm = np.max(map_array[1])
	print 'Interpolating centimorgan coordinates'
	bim_cm = np.interp(bim_array[:,2], map_array[:,0], map_array[:,1], left=min_cm,
		right=max_cm)
	bim_array[:,2] = bim_cm
	print 'Writing output to {O}'.foramt(O=args.out)
	bim_array.to_csv(file=args.bim+'.bim', header=False, index=False, sep='\t')