
PREFIX=$1/Events/run_01
WEIGHTED="false"

if [[ "$1" == "HH-rw" ]]
then
  WEIGHTED="true"
fi


cat <<EOF > $1.cmnd
Main:analyses = tautaujets:csvname->$1.csv:weighted->${WEIGHTED}
Beams:frameType = 4
Beams:LHEF = $PREFIX/unweighted_events.lhe.gz
EOF

./main93 -n 9999999 -l -c rivet.cmnd -c2 $1.cmnd -o $1
