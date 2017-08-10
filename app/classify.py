import os
import pickle
import pandas as pd
import numpy as np

def c100_classify(image, model):
	n_guess = 5
	# Load label names to use in prediction results
	#graph = tf.get_default_graph()

	label_list_path = 'datasets/cifar-100-python/meta'
	keras_dir = os.path.expanduser(os.path.join('~', '.keras'))
	datadir_base = os.path.expanduser(keras_dir)
	if not os.access(datadir_base, os.W_OK):
	    datadir_base = os.path.join('/tmp', '.keras')
	label_list_path = os.path.join(datadir_base, label_list_path)

	with open(label_list_path, mode='rb') as f:
	    labels = pickle.load(f)

	prob = model.predict_proba(np.reshape(image,(1,32,32,3)), batch_size=1, verbose=0)
	pred = pd.DataFrame(data = np.reshape(prob,100), index=labels['fine_label_names'], \
                    columns={'probability'}).sort_values('probability', ascending=False)
	pred['name'] = pred.index
	#pred[:n_guess].to_csv('static/predictions.tsv',sep='\t', index=False)
	return pred[:5]



