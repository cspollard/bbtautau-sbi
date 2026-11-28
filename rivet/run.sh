
source setup.sh

for f in HH-rw higgs # top ZH DYbb
do
  source runPythiaRivet.sh $f &
done
wait
