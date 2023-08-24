for f in top higgs ZH DYbb HH
  do $MGPATH/bin/mg5_aMC -f $f.MG 2>&1 | tee $f.logMG
done
