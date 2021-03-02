bash download_text_data.sh
python get_rescale_baseline.py --lang en -b 16 -m \
    microsoft/deberta-large \
    microsoft/deberta-large-mnli \
