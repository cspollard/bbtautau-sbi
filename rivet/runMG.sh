for f in top higgs HH # ZH DYbb
  do $MGPATH/bin/mg5_aMC -f $f.MG 2>&1 | tee $f.logMG
done
