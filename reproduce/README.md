# Reproducing Experimental Results in Our Paper

## WMT18 Segment-level Results
```sh
bash download_wmt18.sh
python get_wmt18_seg_results.py -b 16 --model roberta-large
```


## WMT17 System-level Results
```sh
bash download_wmt17.sh
python get_wmt17_sys_results.py -b 16 --model roberta-large
``