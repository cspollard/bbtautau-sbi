
PREFIX=$1/Events/run_01

cat <<EOF > $1.cmnd
Main:analyses = tautaujets:csvname->$1.csv
Beams:frameType = 4
Beams:LHEF = $PREFIX/unweighted_events.lhe.gz
EOF

./main93 -n 9999999 -l -c rivet.cmnd -c2 $1.cmnd -o $1