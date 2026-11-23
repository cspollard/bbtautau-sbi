for f in higgs HH # top ZH DYbb
  do $MGPATH/bin/mg5_aMC -f $f.MG 2>&1 | tee $f.logMG
done
