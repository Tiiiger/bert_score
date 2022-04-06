if ! [ -d wmt16 ]; then
    mkdir wmt16
    gz_file=wmt16-metrics-results.tar.gz
    if ! [ -f $gz_file ]; then
        wget https://www.scss.tcd.ie/~ygraham/wmt16-metrics-results.tar.gz
    fi
    tar -xzf $gz_file -C wmt16
    rm -f $gz_file
    echo "Finished downloading and extracting the dataset"
else
    echo "Folder 'wmt16' exists already."
fi
