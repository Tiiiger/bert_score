if ! [ -d wmt17 ]; then
wget http://ufallab.ms.mff.cuni.cz/~bojar/wmt17-metrics-task-package.tgz
mkdir wmt17
tar -xzf wmt17-metrics-task-package.tgz -C wmt17
cd wmt17/input
# tar -xzf wmt17-metrics-task-no-hybrids.tgz -C no_hybrid
tar -xzf wmt17-metrics-task.tgz
fi