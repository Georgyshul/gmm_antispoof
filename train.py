from gmm import train_gmm
from os.path import exists
import pickle

features = "mfcc"

ncomp = 64

model_file = 'gmm_LA_' + features + '.pkl'
model_file_final = 'gmm_' + features + '_asvspoof21_la.pkl'

path_to_data = "/home/georgy/Documents/deepfakes/"
train_folders = [path_to_data + "LA/ASVspoof2019_LA_train/flac/"]
train_keys = [path_to_data + 'LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt']

audio_ext = ".flac"

if not exists(model_file):
    gmm_bonafide = train_gmm(data_label="bonafide", features=features,
                             train_keys=train_keys, train_folders=train_folders,
                             audio_ext=audio_ext, model_file=model_file,
                             ncomp=ncomp, init_only=False)

    gmm_spoof = train_gmm(data_label="spoof", features=features,
                            train_keys=train_keys, train_folders=train_folders,
                            audio_ext=audio_ext, model_file=model_file,
                            ncomp=ncomp, init_only=False)

    gmm_dict = dict()
    gmm_dict['bonafide'] = gmm_bonafide._get_parameters()
    gmm_dict['spoof'] = gmm_spoof._get_parameters()
    with open(model_file, "wb") as tf:
        pickle.dump(gmm_dict, tf)

gmm_dict = dict()
with open(model_file + '_bonafide_init_partial.pkl', "rb") as tf:
    gmm_dict['bona'] = pickle.load(tf)

with open(model_file + '_spoof_init_partial.pkl', "rb") as tf:
    gmm_dict['spoof'] = pickle.load(tf)

with open(model_file_final, "wb") as f:
    pickle.dump(gmm_dict, f)