mkdir -p wmt18
cd wmt18
if ! [ -f wmt18-metrics-task-package ]; then
    wget http://ufallab.ms.mff.cuni.cz/\~bojar/wmt18-metrics-task-package.tgz
    tar -axvf wmt18-metrics-task-package.tgz

    wget http://ufallab.ms.mff.cuni.cz/~bojar/wmt18/wmt18-metrics-task-nohybrids.tgz
    tar -axvf wmt18-metrics-task-nohybrids.tgz

    mv wmt18-metrics-task-nohybrids wmt18-metrics-task-package/input
    mkdir -p wmt18
    mv wmt18-metrics-task-package wmt18

    rm -f *.tgz
fi