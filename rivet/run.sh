
rivet --pwd \
  -a tautaujets:csvname=$1.csv \
  -o $1.yoda \
  $1/Events/run_01/tag_1_pythia8_events.hepmc.gz
