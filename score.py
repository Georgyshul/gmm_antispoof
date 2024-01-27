from gmm import scoring

features = 'mfcc'

# scores file to write
scores_file = 'scores-' + features + '-asvspoof21-LA.txt'

# configs
dict_file = 'gmm_' + features + '_asvspoof21_la.pkl'


db_folder = '/home/georgy/Documents/deepfakes/'  # put your database root path here
# eval_folder = db_folder + 'LA/ASVspoof2019_LA_eval/flac/'
# eval_ndx = db_folder + 'LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt'

eval_folder = db_folder + 'LA/ASVspoof2019_LA_eval/flac/'
eval_ndx = db_folder + 'LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt'

audio_ext = '.flac'

# run on ASVspoof 2021 evaluation set
scoring(scores_file=scores_file, dict_file=dict_file, features=features,
        eval_ndx=eval_ndx, eval_folder=eval_folder, audio_ext=audio_ext,
        features_cached=True, flag_debug=False)
