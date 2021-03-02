# Tuning best layer of a pre-trained English model on WMT16 dataset

### Downloading the dataset
This downloads the WMT16 dataset and extracts it into a new folder called `wmt16`. If the folder `wmt16` exists, it will skip the process.
```sh
bash download_data.sh
```

### Tuning the models
Here is an example of tuning three models in a row:
```sh
python tune_layers.py -m bert-base-uncased roberta-base albert-base-v2
```
The results would be appended to `best_layers_log.txt`.
The last three lines of `best_layers_log.txt` would be
```
'bert-base-uncased': 9, # 0.692518813886652
'roberta-base': 10, # 0.7062886932674598
'albert-base-v2': 9, # 0.6682362357086912
```
which shows the model name, the best number of layers, and the pearson correlation with human judgement.
These can be copied and pasted into `model2layers` in `bert_score/utils.py`.