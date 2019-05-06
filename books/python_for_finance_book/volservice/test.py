import sys, pdb
sys.path.append("/usr/local/lib/python2.7/dist-packages")
sys.path.append("/usr/local/lib/python3.4/dist-packages")
import numpy as np
import urllib.request

url = 'http://localhost:4000/'
urlpara = url + 'application?V0=%s&kappa=%s&theta=%s&sigma=%s&zeta=%s'
urlpara += '&T=%s&r=%s&K=%s'
urlval = urlpara % (25, 2.0, 20, 1.0, 0.0, 1.5, 0.02, 22.5)
print(urlval)
print(urllib.request.urlopen(urlval).read())