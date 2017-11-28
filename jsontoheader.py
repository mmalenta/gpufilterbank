import json
import sys
import numpy as np

schedstr=sys.argv[1]
telstr=sys.argv[2]
obsno=int(sys.argv[3])
telid=int(sys.argv[4])
schedfile=open(schedstr, 'r')
telfile=open(telstr, 'r')
schedule=json.load(schedfile)
schedule=schedule[0]['scheduleDetail']
telescopes=json.load(telfile)
locallo=telescopes['effectiveLO']

header=open('header.dat', 'w')
point=schedule[obsno]
telescope=telescopes['boards'][telid]
centre=telescope['centreFreq']
band=telescope['bandwidth']
topfreq=locallo + centre;
header.write('RAWFILE tastytest\n')
header.write('SOURCE ' + point['src'] + '\n')
header.write('AZ ' + str(point['startAz']) + '\n')
header.write('DEC 0.0\n')
header.write('FCENT ' + str(topfreq) + '\n')
header.write('FOFF ' + str(band) + '\n')
header.write('RA 0.0\n')
header.write('REFDM 0.0\n')
header.write('TSAMP 0.0\n')
header.write('TSTART 57000.0\n')
header.write('ZA ' + str(point['startEl']) + '\n')
header.write('DATATYPE 1\n')
header.write('BEAMNO 0\n')
header.write('MACHINEID 9\n')
header.write('NOBEAMS 0\n')
header.write('OUTBITS 32\n')
header.write('NOCHANS 0\n')
header.write('NOIFS 1\n')
header.write('TELESCOPEID 5\n')
header.close()
