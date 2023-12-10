// -*- C++ -*-
#include "Rivet/Analysis.hh"
#include "Rivet/AnalysisHandler.hh"
#include "Rivet/Projections/FinalState.hh"
#include "Rivet/Projections/PromptFinalState.hh"
#include "Rivet/Projections/VetoedFinalState.hh"
#include "Rivet/Projections/IdentifiedFinalState.hh"
#include "Rivet/Projections/DressedLeptons.hh"
#include "Rivet/Projections/FastJets.hh"


namespace Rivet {


  /// @brief Add a short analysis description here
  class tautaujets : public Analysis {
  public:

    /// Constructor
    RIVET_DEFAULT_ANALYSIS_CTOR(tautaujets);


    /// @name Analysis methods
    /// @{

    /// Book histograms and initialise projections before the run
    void init() {

      // Initialise and register projections

      // handler().skipMultiWeights(true);

      // The basic final-state projection:
      // all final-state particles within
      // the given eta acceptance
      const VisibleFinalState vfs(Cuts::abseta < 5);

      PromptFinalState prompt(vfs);
      prompt.acceptTauDecays(true);

      IdentifiedFinalState photons(prompt);
      photons.acceptIdPair(PID::PHOTON);

      IdentifiedFinalState electrons(prompt);
      electrons.acceptIdPair(PID::ELECTRON);

      IdentifiedFinalState muons(prompt);
      muons.acceptIdPair(PID::MUON);

      Cut leptoncuts = Cuts::abseta < 2.5 && Cuts::pT > 25*GeV;

      DressedLeptons dressedelectrons(photons, electrons, 0.1, leptoncuts);
      declare(dressedelectrons, "dressedelectrons");

      DressedLeptons dressedmuons(photons, muons, 0.1, leptoncuts);
      declare(dressedmuons, "dressedmuons");

      VetoedFinalState jetfs(vfs);
      jetfs.addVetoOnThisFinalState(dressedelectrons);
      jetfs.addVetoOnThisFinalState(dressedmuons);


      FastJets jets
        ( jetfs
        , FastJets::ANTIKT
        , 0.4
        , JetAlg::Muons::NONE, JetAlg::Invisibles::NONE
        );

      declare(jets, "jets");


      csv = ofstream();
      const string& csvname = getOption("csvname");
      weighted = getOption("weighted") == "true";
      cout << "writing to csv file " << csvname << endl;

      csv.open(csvname);

      eventnumber = 0;

      csv
        << "eventnumber"
        << ","
        << "btagged"
        << ","
        << "ctagged"
        << ","
        << "tautagged"
        << ","
        << "px"
        << ","
        << "py"
        << ","
        << "pz"
        << ","
        << "weight_nom";

      if (weighted)
        csv
          << ","
          << "weight_kl0"
          << ","
          << "weight_kl2";

      csv << endl;

      cout
        << "WARNING: THIS ASSUMES THAT THE FIRST TWO WEIGHTS CORRESPOND TO kL = 0 and kL = 2!!!"
        << endl;

      return;
    }


    /// Perform the per-event analysis
    void analyze(const Event& event) {


      // Retrieve dressed leptons, sorted by pT
      const Jets& jets = 
        apply<FastJets>(event, "jets")
          .jetsByPt(Cuts::abseta < 2.5 && Cuts::pT > 20*GeV);

      vector<bool> btags, ctags, tautags;

      for (const auto& j : jets) {
        bool btag = j.bTagged();
        bool ctag = j.cTagged();
        bool tautag = !btag && j.tauTagged();

        btags.push_back(btag);
        ctags.push_back(ctag);
        tautags.push_back(tautag);
      }

      const GenEvent* genevent = event.genEvent();
      const std::valarray<double>& wgts = HepMCUtils::weights(*genevent);

      for (int i = 0; i < jets.size(); i++) {
        const Jet& j = jets.at(i);
        bool btag = btags.at(i);
        bool ctag = ctags.at(i);
        bool tautag = tautags.at(i);
        double wnom = wgts[0];

        csv
          << eventnumber
          << ","
          << int(btag)
          << ","
          << int(ctag)
          << ","
          << int(tautag)
          << ","
          << j.px()
          << ","
          << j.py()
          << ","
          << j.pz()
          << ","
          << wnom;

        if (weighted) {
          float w0 = wgts[1], w2 = wgts[2];
          csv
            << ","
            << w0
            << ","
            << w2;
        }

        csv << endl;

      }

      // it's alright to only increment event number after the veto.
      eventnumber++;

      return;
    }


    /// Normalise histograms etc., after the run
    void finalize() { csv.close(); }


    /// @}

    private:
      bool weighted;
      ofstream csv;
      int eventnumber;

  };


  RIVET_DECLARE_PLUGIN(tautaujets);

}
