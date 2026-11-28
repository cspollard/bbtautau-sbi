PATH="$PWD/../inference/venv/bin:$PATH"
for f in higgs HH # top ZH DYbb
  do python3 $MGPATH/bin/mg5_aMC -f $f.MG 2>&1 | tee $f.logMG
done
