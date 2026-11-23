set nb_core 4
import model loop_sm
generate g g > h h QED<=2 [noborn=QCD]
output HH_bbtautau
launch HH_bbtautau
madspin=ON
set nevents 500000
set ebeam1 7000
set ebeam2 7000
decay h > b b~
decay h > ta+ ta-
