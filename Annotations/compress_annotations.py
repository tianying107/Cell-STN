import numpy as np
import pickle

# update the cell_type before run
cell_type = 'U2OS'

experiment = np.loadtxt("./{}/separate_annotations/experiment.txt".format(cell_type))
with open('./{}/separate_annotations/division.pkl'.format(cell_type), 'rb') as f:
    division = pickle.load(f)
with open('./{}/separate_annotations/death.pkl'.format(cell_type), 'rb') as f:
    death = pickle.load(f)
with open('./{}/separate_annotations/cols.pkl'.format(cell_type), 'rb') as f:
    cols = pickle.load(f)
with open('./{}/separate_annotations/rows.pkl'.format(cell_type), 'rb') as f:
    rows = pickle.load(f)
with open('./{}/separate_annotations/lineage.pkl'.format(cell_type), 'rb') as f:
    lineage = pickle.load(f)

annotations = {'cell_type': cell_type, 'division': division, 'death': death, 'rows': rows, 'cols': cols, 'lineage': lineage, 'experiment': np.array(experiment, dtype='uint16')}
print(annotations)
output = open('./{}/annotations.pkl'.format(cell_type), 'wb')
pickle.dump(annotations, output)
output.close()
exit()
