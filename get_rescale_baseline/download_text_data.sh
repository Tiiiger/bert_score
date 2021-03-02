mkdir -p data
cd data
if ! [ -f news.2017.en.shuffled.deduped ]; then
    wget http://data.statmt.org/wmt18/translation-task/news.2017.en.shuffled.deduped.gz
    gzip -d news.2017.en.shuffled.deduped.gz
fi

echo "finish downloading data"