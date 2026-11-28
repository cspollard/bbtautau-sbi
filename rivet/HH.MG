set nb_core 4
set acknowledged_v3.1_syntax True
import model loop_sm_hh
generate g g > h h QED<=2 [noborn=QCD]
output HH-rw
shell cp HH-rw_reweight_card.dat HH-rw/Cards/reweight_card.dat
shell cp HH-rw_madspin_card.dat HH-rw/Cards/madspin_card.dat
launch HH-rw
madspin=ON
reweight=ON
set nevents 500000
set ebeam1 7000
set ebeam2 7000
set BSMINPUTS 1 1.0
