
rivet --pwd \
  -a tautaujets:csvname=${1/hepmc.gz/csv} \
  -o ${1/hepmc.gz/yoda} \
  $1
