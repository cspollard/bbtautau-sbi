// -*- C++ -*-
#include "Rivet/Analysis.hh"
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
      cout << "writing to csv file " << csvname << endl;

      csv.open(csvname);

      eventnumber = 0;

      csv
        << "eventnumber"
        << ","
        << "btagged"
        << ","
        << "tautagged"
        << ","
        << "px"
        << ","
        << "py"
        << ","
        << "pz"
        << endl;

      return;
    }


    /// Perform the per-event analysis
    void analyze(const Event& event) {

      cout << _weightNames() << endl;

      // Retrieve dressed leptons, sorted by pT
      const Jets& jets = 
        apply<FastJets>(event, "jets")
          .jetsByPt(Cuts::abseta < 2.5 && Cuts::pT > 20*GeV);

      if (jets.size() < 4) vetoEvent;

      vector<bool> btags, tautags;
      int nbtag = 0, ntautag = 0;

      for (const auto& j : jets) {
        bool btag = j.bTagged();
        bool tautag = !btag && j.tauTagged();

        btags.push_back(btag);
        tautags.push_back(tautag);

        nbtag += btag;
        ntautag += tautag;
      }

      if (nbtag < 2 || ntautag < 2) vetoEvent;

      for (int i = 0; i < jets.size(); i++) {
        const Jet& j = jets.at(i);
        bool btag = btags.at(i);
        bool tautag = tautags.at(i);

        csv
          << eventnumber
          << ","
          << int(btag)
          << ","
          << int(tautag)
          << ","
          << j.px()
          << ","
          << j.py()
          << ","
          << j.pz()
          << endl;

      }

      // it's alright to only increment event number after the veto.
      eventnumber++;

      return;
    }


    /// Normalise histograms etc., after the run
    void finalize() { csv.close(); }


    /// @}

    private:
      ofstream csv;
      int eventnumber;

  };


  RIVET_DECLARE_PLUGIN(tautaujets);

}
