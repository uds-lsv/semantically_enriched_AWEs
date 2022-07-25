#!/usr/bin/env python
# coding: utf-8
import os
import sys
import yaml
import pprint
import collections

# to get time model was trained
from datetime import datetime
import pytz

# NOTE: import torch before pandas, otherwise segementation fault error occurs
# The couse of this problem is UNKNOWN, and not solved yet
import torch
import numpy as np
import pandas as pd
import pickle

import torch.nn as nn
import torch.optim as optim

from nn_speech_models import *
import train_utils

from CKA import *


# obtain yml config file from cmd line and print out content
if len(sys.argv) != 2:
	sys.exit("\nUsage: " + sys.argv[0] + " <config YAML file>\n")

config_file_path = sys.argv[1] # e.g., '/speech_cls/config_1.yml'
config_args = yaml.safe_load(open(config_file_path))
print('YML configuration file content:')
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(config_args)


# get time in CET timezone
current_time = datetime.now(pytz.timezone('Europe/Amsterdam'))
current_time_str = current_time.strftime("%d%m%Y_%H_%M_%S") # YYYYMMDD HH:mm:ss
#print(current_time_str)


# make a model str ID, this will be used to save model on desk
# config_args['model_str'] = '_'.join(str(_var) for _var in
#     [
#         current_time_str,
#         config_args['experiment_name'],
#         config_args['language_set'],
#         config_args['input_signal_params']['acoustic_features']
#     ]
# )


# make the dir str where the model will be stored
# if config_args['expand_filepaths_to_save_dir']:
#     config_args['model_state_file'] = os.path.join(
#         config_args['model_save_dir'], config_args['model_str']
#     )
#
#     print("Expanded filepaths: ")
#     print("\t{}".format(config_args['model_state_file']))
#
# # if dir does not exits on desk, make it
# train_utils.handle_dirs(config_args['model_save_dir'])


 # Check CUDA
if not torch.cuda.is_available():
    config_args['cuda'] = False

config_args['device'] = torch.device("cuda" if config_args['cuda'] else "cpu")

print("Using CUDA: {}".format(config_args['cuda']))


# Set seed for reproducibility
train_utils.set_seed_everywhere(config_args['seed'], config_args['cuda'])




##### HERE IT ALL STARTS ...
# dataset  & featurizer ...
speech_df = pd.read_csv(config_args['speech_metadata'],
    delimiter="\t", encoding='utf-8')


model_set = config_args['model_files'].split()


speech_df = speech_df[
    (speech_df.language.isin(label_set)) &
	(speech_df.fasttext==True) &
    (speech_df.num_ph>3) &
    (speech_df.frequency>1) &
    (speech_df.duration<1.10) # &
]


#speech_df = pd.concat([train_speech_df, valid_speech_df])

#print(speech_df.head())
print(len(speech_df))


#speech_df = speech_df.sample(n=2500, random_state=1)

# shuffle splits among words
#speech_df['split'] = np.random.permutation(speech_df.split)

word_vocab = set(speech_df['IPA'].values)

word_featurizer = WordFeaturizer(
    data_dir=config_args['data_dir'],
    acoustic_features= config_args['input_signal_params']['acoustic_features'],
    max_num_frames=config_args['input_signal_params']['max_num_frames'],
    spectral_dim=config_args['acoustic_encoder']['spectral_dim'],
    word_vocab=word_vocab
)

print('WordFeaturizer was initialized:', word_featurizer.char_vocab)

#  dataloader ...
with open(config_args['embedding_file'], 'rb') as f:
	embedding_lookup = pickle.load(f, encoding='utf-8')

embedding_lookup = {k.encode('utf-8'): v for k, v in  embedding_lookup.items()}
#
#
# print(len(embedding_lookup))
#
# orth_word_vocab = set(w.encode('utf-8') for w in set(speech_df['orth'].values))
#
#
# for w in embedding_lookup:
# 	print(w, end='\t') #.encode('utf-8')
# 	print()
# 	break
#
# for w in orth_word_vocab:
# 	print(w, end='\t') #.encode('utf-8')
# 	print()
# 	break
#
#
# target_embedding_lookup = {
# 	k:v for k, v in embedding_lookup.items() if k in orth_word_vocab
# }
#
# print(len(target_embedding_lookup))

#embedding_lookup = pd.read_pickle(config_args['embedding_file'])
word_dataset = SemanticWordDataset(speech_df, word_featurizer, embedding_lookup)

word2label = {word:idx for idx, word in enumerate(set(speech_df.orth.values))}

label2word = {idx:word for word, idx in word2label.items()}




# initialize acoustic encoder
if config_args['acoustic_encoder']['encoder_arch']=='CNN':
	# initialize a CNN encoder
	acoustic_encoder = AcousticEncoder(
		spectral_dim=config_args['acoustic_encoder']['spectral_dim'],
		max_num_frames=config_args['acoustic_encoder']['max_num_frames'],
		output_dim=config_args['acoustic_encoder']['output_dim'],
		num_channels=config_args['acoustic_encoder']['num_channels'],
		filter_sizes=config_args['acoustic_encoder']['filter_sizes'],
		stride_steps=config_args['acoustic_encoder']['stride_steps'],
		pooling_type=config_args['acoustic_encoder']['pooling_type'],
		unit_dropout_prob=config_args['acoustic_encoder']['unit_dropout_prob']
	)

elif config_args['acoustic_encoder']['encoder_arch']=='LSTM':
	# initialize an LSTM encoder
	acoustic_encoder = AcousticEncoderLSTM(
		spectral_dim=config_args['acoustic_encoder']['spectral_dim'],
		max_num_frames=config_args['acoustic_encoder']['max_num_frames'],
		output_dim=config_args['acoustic_encoder']['output_dim'],
		hidden_state_dim=config_args['acoustic_encoder']['hidden_state_dim'],
		n_layers=config_args['acoustic_encoder']['n_layers'],
		unit_dropout_prob=config_args['acoustic_encoder']['unit_dropout_prob']
	)

elif config_args['acoustic_encoder']['encoder_arch']=='BiLSTM':
	# initialize an LSTM encoder
	acoustic_encoder = AcousticEncoderBiLSTM(
		spectral_dim=config_args['acoustic_encoder']['spectral_dim'],
		max_num_frames=config_args['acoustic_encoder']['max_num_frames'],
		output_dim=config_args['acoustic_encoder']['output_dim'],
		hidden_state_dim=config_args['acoustic_encoder']['hidden_state_dim'],
		n_layers=config_args['acoustic_encoder']['n_layers'],
		unit_dropout_prob=config_args['acoustic_encoder']['unit_dropout_prob']
	)

elif config_args['acoustic_encoder']['encoder_arch']=='BiGRU':
	# initialize an LSTM encoder
	acoustic_encoder = AcousticEncoderBiGRU(
		spectral_dim=config_args['acoustic_encoder']['spectral_dim'],
		max_num_frames=config_args['acoustic_encoder']['max_num_frames'],
		output_dim=config_args['acoustic_encoder']['output_dim'],
		hidden_state_dim=config_args['acoustic_encoder']['hidden_state_dim'],
		n_layers=config_args['acoustic_encoder']['n_layers'],
		unit_dropout_prob=config_args['acoustic_encoder']['unit_dropout_prob']
	)

elif config_args['acoustic_encoder']['encoder_arch']=='GRU':
    # initialize an LSTM encoder
    acoustic_encoder = AcousticEncoderGRU(
	    spectral_dim=config_args['acoustic_encoder']['spectral_dim'],
	    max_num_frames=config_args['acoustic_encoder']['max_num_frames'],
	    output_dim=config_args['acoustic_encoder']['output_dim'],
	    hidden_state_dim=config_args['acoustic_encoder']['hidden_state_dim'],
	    n_layers=config_args['acoustic_encoder']['n_layers'],
	    unit_dropout_prob=config_args['acoustic_encoder']['unit_dropout_prob'],
	    frame_dropout=config_args['acoustic_encoder']['frame_dropout'],
	    signal_dropout_prob=config_args['acoustic_encoder']['signal_dropout_prob']
    )

else:
	raise NotImplementedError


phonetic_decoder = PhoneticDecoder(
	embedding_dim=config_args['phonetic_decoder']['embedding_dim'],
	hidden_state_dim=config_args['phonetic_decoder']['hidden_state_dim'],
	n_layers=config_args['phonetic_decoder']['n_layers'],
	vocab_size=config_args['phonetic_decoder']['vocab_size'][l]#len(word_featurizer.char_vocab)
)

print(config_args['device'])

for _model in model_set:
	# make the dir str where the model will be stored
	if config_args['expand_filepaths_to_save_dir']:
		config_args['model_state_file'] = os.path.join(
		config_args['model_save_dir'], config_args['pretrained_models'][_model]
	)

	print("Expanded filepaths: ")
	print("\t{}".format(config_args['model_state_file']))

	word_encoder = SemanticEncoder(acoustic_encoder,
		phonetic_decoder,
		config_args['acoustic_encoder']['encoder_arch'],
		config_args['device']
	)

	word_encoder.load_state_dict(torch.load(config_args['model_state_file']))
	# move model to GPU
	word_encoder = word_encoder.cuda()


	print(word_encoder)

	print('Evaluation started ...')

	batch_size = config_args['training_hyperparams']['batch_size']


	word_dataset.set_mode(config_args['eval_split'])
	word_encoder.eval()

		try:

			### VALIDATION ...

			batch_generator = generate_batches(word_dataset,
				batch_size=batch_size, device=config_args['device'],
				drop_last_batch=False,shuffle_batches=False
			)

			num_batches = word_dataset.get_num_batches(batch_size)


			for batch_index, batch_dict in enumerate(batch_generator):

				# forward pass, get embeddings
				phon_tar = batch_dict['symbolic_word']
				semantic_tar = batch_dict['semantic_embedding']

				# forward pass, get predictions and embeddings
				rnn1,  semantic_embs, rnn4 = word_encoder( #rnn2, rnn3,
					batch_dict['acoustic_word'],
					phon_tar,
					inference=True
				)

				#output_dim = phon_hat.shape[-1]
				#phon_tar = phon_tar.permute(1, 0)

				#print('output_dim', output_dim,  phon_hat.shape, phon_tar.shape)

				#phon_hat = phon_hat[1:].view(-1, output_dim)
				#phon_tar = phon_tar[1:].contiguous().view(-1)

				#val_phon_loss = cross_entropy_loss(phon_hat, phon_tar)
				#val_semantic_loss = MSE_loss(semantic_embs, semantic_tar)/config_args['training_hyperparams']['batch_size']


				# get word orthographic form for all word in batch
				words_in_batch = batch_dict['orth_sequence']


				# make word labels, this makes it possible to know whether or
				# not the representation belong to the same word in each view
				word_labels = [word2label[w] for w in words_in_batch]

				#rand_idx = torch.randint(0, ank_embeddings.shape[0], (ank_embeddings.shape[0],))

				# get vectors
				if batch_index == 0:
					#acoustic_vectors = acoustic_embs.cpu().detach().numpy()
					acoustic_vectors_1 = rnn1.cpu().detach().numpy()
					#acoustic_vectors_2 = rnn2.cpu().detach().numpy()
					# acoustic_vectors_3 = rnn3.cpu().detach().numpy()
					acoustic_vectors_4 = rnn4.cpu().detach().numpy()
					semantic_vectors = semantic_embs.cpu().detach().numpy()
					word_labels_list = word_labels
				else:
					acoustic_vectors_1 = np.concatenate(
						(acoustic_vectors_1, rnn1.cpu().detach().numpy()),
						axis=0
					)

					# acoustic_vectors_2 = np.concatenate(
					# 	(acoustic_vectors_2, rnn2.cpu().detach().numpy()),
					# 	axis=0
					# )

					# acoustic_vectors_3 = np.concatenate(
					# 	(acoustic_vectors_3, rnn3.cpu().detach().numpy()),
					# 	axis=0
					# )
					#
					acoustic_vectors_4 = np.concatenate(
						(acoustic_vectors_4, rnn4.cpu().detach().numpy()),
						axis=0
					)

					semantic_vectors = np.concatenate(
						(semantic_vectors, semantic_embs.cpu().detach().numpy()),
						axis=0
					)

					word_labels_list.extend(word_labels)

				print(f"{config_args['model_state_file']}"
					 f"[{batch_index + 1:>4}/{num_batches:>4}]   "
				)


			# lang2mat_1[l] = acoustic_vectors_1
			# lang2mat_2[l] = acoustic_vectors_4
			# # lang2mat_3[l] = acoustic_vectors_3
			# # lang2mat_4[l] = acoustic_vectors_4
			# lang2mat_5[l] = semantic_vectors

			#print(f"Size of validation set: {len(acoustic_vectors)}");
			#lang2mat_1[l] = semantic_vectors #acoustic_vectors #   #
			# at the end of one validation block, compute AP



			#print(acoustic_vectors[0])
			#print(acoustic_vectors[1])
			acoustic_AP = train_utils.average_precision(
			    acoustic_vectors_4,
			    acoustic_vectors_4,
			    word_labels_list,
			    word_labels_list,
			    label2word,
			    single_view=True
			)

			# semantic_AP = train_utils.average_precision(
			#     semantic_vectors,
			#     semantic_vectors,
			#     word_labels_list,
			#     word_labels_list,
			#     label2word,
			#     single_view=True
			# )


			# recall_at_1, recall_at_5, recall_at_10 = train_utils.compute_reacall_topk(
			#     target_embedding_vectors,
			#     semantic_vectors,
			#     target_word_labels,
			#     word_labels_list,
			#     label2word,
			# )

			mean_acoustic_AP = np.mean(acoustic_AP)
			#recall_at_10 = np.mean(recall_topK)


			print(
			    f"L: {len(acoustic_AP)} "
			    f"acoustic AP: {mean_acoustic_AP:<1.6f}"
			)
			#
			# print(
			# 	f"L: {len(acoustic_AP)} "
			# 	f"R @ 1: {recall_at_1:<1.6f}   "
			# 	f"R @ 5: {recall_at_5:<1.6f}   "
			#     f"R @10: {recall_at_10:<1.6f}   "
			#)


		except KeyboardInterrupt:
			print("Exiting loop")
