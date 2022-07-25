## Semantically-Enriched Acoustic Word Embeddings :tea:

This is the code base for the multi-task learning model for acoustic word embedding models, training experiments, and evaluation scripts for the experiments reported in our **INTERSPEECH 2022** paper 

:pencil: [Integrating Form and Meaning: A Multi-Task Learning Model for Acoustic Word Embeddings](Arxiv Link)

<!-- To cite the paper

```
@inproceedings{Abdullah2021DoAW,
  title={Integrating Form and Meaning: A Multi-Task Learning Model for Acoustic Word Embeddings},
  author={Badr M. Abdullah and Bernd Möbius and Dietrich Klakow},
  booktitle={Proc. Interspeech},
  year={2022}
}
``` -->

### Dependencies :dna:

python 3.8, pytorch 1.1, numpy, scipy, faiss, pickle, pandas, yaml


### Speech Data :speech_balloon: :left_speech_bubble:
The data in our study is drawn from the Multilingual GlobalPhone speech database for  Portuguese :brazil:, German :de: and  Polish :poland:. Because the data is distributed under a research license by XLingual LLC., we cannot re-distribute the raw speech data. However, if you have already access to the GlobalPhone speech database and you would like to access to our word-alignment annotations, train/test splits, and word-level IPA transcriptions, please contact the first author. 


### Working with the code :snake:
To run a experiment to train a multi-task learning model for AWEs, write down all (meat)data for experiments and hyperparameters and other info in the config file ```config_file_train_awe_gru_sem_pge.yml```

Then ...

```
>>> cd AWEs_phon_sim
>>> python nn_train_sem_pge_embeddings.py config_files/config_file_train_awe_gru_sem_pge.yml
```

The code is fairly documented and the vectorization logic, as well as the code for the models, should be useful for other speech technology tasks. If you use our code and encounter problems, please create an issue or contact the first author. 


If you use our code in a work that leads to a publication, please cite our paper as 

```
@inproceedings{Abdullah2021DoAW,
  title={Integrating Form and Meaning: A Multi-Task Learning Model for Acoustic Word Embeddings},
  author={Badr M. Abdullah and and Bernd Möbius and Dietrich Klakow},
  booktitle={Proc. Interspeech},
  year={2022}
}
```