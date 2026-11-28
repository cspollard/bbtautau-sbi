
PREFIX=$1/Events/run_01
if [[ -d "$1/Events/run_01_decayed_1" ]]
then
  PREFIX=$1/Events/run_01_decayed_1
fi
WEIGHTED="false"

if [[ "$1" == "HH-rw" ]]
then
  WEIGHTED="true"
fi


if [[ ! -f "$PREFIX/unweighted_events.lhe" && -f "$PREFIX/unweighted_events.lhe.gz" ]]
then
  gunzip -kf "$PREFIX/unweighted_events.lhe.gz"
fi

cat <<EOF > $1.cmnd
Rivet:fileName = $1.yoda
Rivet:analyses = {tautaujets:csvname=$1.csv:weighted=${WEIGHTED}}
Beams:frameType = 4
Beams:LHEF = $PREFIX/unweighted_events.lhe
EOF

$PYTHIAPATH/examples/main144 -n 9999999 -l -c rivet.cmnd -c $1.cmnd -o $1
