from coffea.nanoevents import NanoEventsFactory
from coffea.nanoevents.schemas import TreeMakerSchema

import numpy as np
import awkward
import dask_awkward
import correctionlib
import correctionlib.schemav2 as cs
from coffea.lookup_tools.correctionlib_wrapper import correctionlib_wrapper

import hist
import hist.dask
import os, sys

from distributed import Client


def awkward_switch(arr):
  if isinstance(arr, dask_awkward.lib.core.Array):
    return dask_awkward
  else:
    return awkward


def run_deltar_matching(store,
                        target,
                        drname='deltaR',
                        radius=0.4,
                        unique=False,
                        sort=False):
  ak = awkward_switch(store)
  _, target = ak.unzip(ak.cartesian([store.eta, target], nested=True))
  target[drname] = store.delta_r(target)
  # if unique:  # Additional filtering
  #   t_index = ak.argmin(target[drname], axis=-2)
  #   s_index = ak.local_index(store.eta, axis=-1)
  #   _, t_index = ak.unzip(ak.cartesian([s_index, t_index], nested=True))
  #   target = target[s_index == t_index]

  # Cutting on the computed delta R
  target = target[target[drname] < radius]

  # Sorting according to the computed delta R
  if sort:
    idx = ak.argsort(target[drname], axis=-1)
    target = target[idx]
  return target


def get_dark_jet_mask(events, jets, radius, frac_thres=0.6):
  """
  Getting the True/False mask corresponding to whether the jet is generated from
  a dark quarks. This function is explicitly written because the default
  partonFlavor assignment in MINIAOD doesn't work well with the emerging jets
  samples.

  GenJets have a variable indicating the number of constituents with dark sector
  ancestors. This fraction is used to tag GEN jets as originating from a dark
  quark. The GenJets is then matched to the reco-level jet with the specified
  radius to tag reco-level jets to originate from a dark quark.

  A simple parsing of checking the events collection for any HV gen particles is
  done to avoid running this mask on SM events
  """
  ak = awkward_switch(events)
  return ak.zeros_like(jets.pt) == 1  # All false array
  if not ak.any(events.GenParticles.PdgId >= 4900001, axis=None):
    return ak.zeros_like(jets.pt) == 1  # All false array

  if 'nHVAncestors' not in events.GenJets.fields:
    return get_dark_jet_meson_mask(events, jets, radius, frac_thres)
  # Getting the closest GenJet in deltaR
  gen = run_deltar_matching(
      jets,
      events.GenJets if radius == 0.4 else events.GenJetsAK8,
      radius=radius,
      unique=True,
      sort=True)[:, :, :1]
  # checking the ancestor fraction:
  is_dark = (gen.nHVAncestors / gen.multiplicity) > frac_thres
  # Running an ak.sum to eliminate empty GenJet matching cases
  return ak.sum(is_dark, axis=-1) >= 1


def get_jet_gen_flavor(events, jets, radius):
  """
  Given an event and a jet collection, return a awkward array with the shape
  compatible with jet level variables representing the reduced jet gen-level
  flavor as defined using the JetFlavor enum.

  Flavor assignment is based on the algorithm defined here [1], with the
  additional definition that the dark flavor takes precedent.

  - If hadronFlavor is non-zero, then the flavor will always be that of the
    hadron flavor.
  - If hadronFlavor is zero, then we take this to be the partonFlavor.

  [1]
  https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideBTagMCTools#Hadron_parton_based_jet_flavour

  """
  ak = awkward_switch(events)
  if 'madHT' not in events.fields:  # setting to unknown for any data files
    return ak.where(ak.ones_like(jets.pt), JetFlavor.UNKNOWN, JetFlavor.UNKNOWN)

  # Assigning the default flavors first
  is_dark = get_dark_jet_mask(events, jets, radius)

  # standard hadronFlavor-precedent assignment method:
  is_b = ((np.abs(jets.hadronFlavor) == 5) |  #
          ((jets.hadronFlavor == 0) & (np.abs(jets.partonFlavor) == 5)))
  is_c = ((np.abs(jets.hadronFlavor) == 4) |  #
          ((jets.hadronFlavor == 0) & (np.abs(jets.partonFlavor) == 4)))
  is_l = ((np.abs(jets.hadronFlavor) == 0) &  #
          (1 <= np.abs(jets.partonFlavor)) & (np.abs(jets.partonFlavor) <= 3))
  is_g = (np.abs(jets.hadronFlavor) == 0) & (np.abs(jets.partonFlavor) == 21)

  flavor = ak.ones_like(jets.hadronFlavor) * -1
  flavor = ak.where(is_g, ak.ones_like(flavor) * 0, flavor)
  flavor = ak.where(is_l, ak.ones_like(flavor) * 1, flavor)
  flavor = ak.where(is_c, ak.ones_like(flavor) * 4, flavor)
  flavor = ak.where(is_b, ak.ones_like(flavor) * 5, flavor)
  # being assigned dark flavor overrides SM flavors.
  flavor = ak.where(is_dark, ak.ones_like(flavor) * 490000, flavor)

  return flavor


def run_jet_track_matching(events, jets, radius):
  ak = awkward_switch(events)
  # Basic jet kinematic selection
  jets = jets[(jets.pt > 100) &  #
              (np.abs(jets.eta) < 2.0) &  #
              (jets.ID > 0)]

  # Getting overlapping jets with PT > 50 GeV
  mask_jets = events.Jets
  mask_jets = mask_jets[mask_jets.pt > 0.5 * 100]
  mask_jets = run_deltar_matching(jets, mask_jets, radius=radius)
  mask_jets = mask_jets[ak.argsort(mask_jets.pt, ascending=False, axis=-1)]

  # Storing the overlapping jets, keeping only the highest PT jet.
  jets['neighborJets'] = mask_jets
  jets = jets[jets.neighborJets.origIndex[:, :, 0] == jets.origIndex]

  # Making track selection objects
  tracks = events.Tracks
  tracks = tracks[(tracks.pt > 1.) & ((tracks.quality & (1 << 2)) > 0)]

  # Generate jet-track association structure
  matched_tracks = run_deltar_matching(jets, tracks, radius=radius, unique=True)
  matched_tracks["ipz"] = matched_tracks.referencePoint.z - \
                          events.PrimaryVertices.z[:, 0]
  matched_tracks["DN"] = (matched_tracks.ipz**2 / 0.0001 +
                          matched_tracks.IP2DSigPV0**2)**0.5
  # Assigning tracks to jets
  jets["Tracks"] = matched_tracks[:]  # Making a copy
  jets["has_highpt_track"] = ak.any(jets.Tracks.pt >= jets.pt * 0.6, axis=-1)

  # Jet-Track selection
  jets = jets[ak.count(jets.Tracks.pt, axis=-1) > 0]
  jets = jets[~jets.has_highpt_track]

  # Make additional jet-level variables
  jets['flavor'] = get_jet_gen_flavor(events, jets, 0.4)
  jets["Weight"] = events.Weight  # creating a per-jet weight column
  jets['TRadius'] = 0.4 * ak.ones_like(
      jets.pt)  # Saving the association radius for future use.

  return jets


def ak_quantile_inner(x, q=0.5, **kwargs):
  ak = awkward_switch(x)
  sorted_x = ak.sort(x, axis=-1)
  index = ak.argsort(sorted_x, axis=-1)
  target_index = (ak.count(sorted_x, axis=-1) - 1) * q
  weight = target_index - np.floor(target_index)
  lower_val = sorted_x[index == np.floor(target_index)]
  upper_val = sorted_x[index == np.ceil(target_index)]

  ans = lower_val * weight + upper_val * (1 - weight)
  # ans should only have one entry in the inner most dimesion
  return ak.sum(ans, axis=-1)


def ak_divide_denzero(num, den, val=0):
  ak = awkward_switch(num)
  return ak.where(den != 0, num / den, ak.ones_like(num) * val)


def ak_weighted_avg(arr, weight, axis=-1, denzero=0):
  ak = awkward_switch(arr)
  return ak_divide_denzero(ak.sum(arr * weight, axis=axis),
                           ak.sum(weight, axis=axis),
                           val=denzero)


def refine_jet_tracks(events, jets, ipz_cut):
  # Ensuring that the jets is a subcollection of the events collection
  # assert len(events) == len(orig_jets)
  ak = awkward_switch(events)

  # Refined jet-track selection
  jets["Tracks"] = jets.Tracks[np.abs(jets.Tracks.ipz) < np.abs(ipz_cut)]
  jets = jets[ak.count(jets.Tracks.x, axis=-1) > 0]

  # Counting the number of tracks
  jets['trackmult'] = ak.count(jets.Tracks.x, axis=-1)

  # Track IP2D parameters
  ip2d_abs = np.abs(jets.Tracks.IP2DPV0)
  jets["IP2DMean"] = ak.mean(ip2d_abs, axis=-1)
  jets["IP2DMedian"] = ak_quantile_inner(ip2d_abs, q=0.5)
  jets["IP2DQp75"] = ak_quantile_inner(ip2d_abs, q=0.75)
  jets["IP2DQp875"] = ak_quantile_inner(ip2d_abs, q=0.875)

  # Summing PT related parameters
  jets['tkPtSum'] = ak.sum(jets.Tracks.pt, axis=-1)

  # Making delta R related jet parameters
  jets['PtDeltaR'] = ak_weighted_avg(jets.Tracks.deltaR, jets.Tracks.pt)
  jets['IP2DDeltaR'] = ak_weighted_avg(jets.Tracks.deltaR, jets.Tracks.IP2DPV0)
  jets['deltaRIP2D'] = ak_weighted_avg(jets.Tracks.IP2DPV0, jets.Tracks.deltaR)

  # Calculating the modified n-subjettiness using the matched jets, because
  # ak.cartesian doesn't play well with doublely nested structures, we will need
  # to flatten the jet-level array.
  njets = ak.count(jets.pt, axis=-1)
  radius = ak.flatten(ak.ones_like(jets.Tracks.pt) * jets.TRadius, axis=1)
  tracks = ak.flatten(jets.Tracks, axis=1)
  subjets = ak.flatten(jets.neighborJets, axis=1)
  subjets = run_deltar_matching(tracks, subjets, radius=100, drname='drT')

  def tau(n):
    num = ak.sum(tracks.pt * ak.min(subjets.drT[:, :, :n], axis=-1), axis=-1)
    den = ak.sum(radius * tracks.pt, axis=-1)
    return ak.unflatten(num / den, counts=njets)

  jets['tau1'] = tau(1)
  jets['tau2'] = tau(2)
  jets['tau3'] = tau(3)

  return jets


def standard_vertex_selection(events):
  events = events[ak.count(events.PrimaryVertices.isGood, axis=-1) > 0]
  events = events[(events.PrimaryVertices.isGood[:, 0])
                  & (~events.PrimaryVertices.isFake[:, 0])
                  & (np.abs(events.PrimaryVertices.z[:, 0]) < 15)]
  events['PassSumPT2'] = (events.PrimaryVertices.sumTrackPt2[:, 0] == ak.max(
      events.PrimaryVertices.sumTrackPt2, axis=-1))
  return events


def pass_MET_filters(events):
  return ((events.PrimaryVertexFilter == 1) &  # Flag_goodVertex
          (events.globalSuperTightHalo2016Filter == 1) &  #
          (events.HBHENoiseFilter == 1) &  #
          (events.HBHEIsoNoiseFilter == 1) &  #
          (events.EcalDeadCellTriggerPrimitiveFilter == 1) &  #
          (events.BadPFMuonFilter == 1) &  #
          (events.BadPFMuonDzFilter == 1) &  #
          (events.eeBadScFilter == 1) &  #
          (events.ecalBadCalibFilter == 1)  #
          )


def __mc_lumi_weight(sample_name):
  """
  Returning the luminosity of the era string (useful for MC sample weighting) in
  units of pb-1. This is based on the calculations of the triggers used to
  produce the TreeMaker n-tuples [1], so the results will be different from the
  official luminosity as posted by the LUMI-POG where all triggers are taken
  into consideration.

  Here we are assuming that the GJets samples will be compared to the
  SinglePhoton/EGamma data stream, everything else will be assumed to be
  compared to the JetHT data stream.
  [1]
  https://twiki.cern.ch/twiki/bin/viewauth/CMS/RA2b13TeVProduction#Dataset_luminosities_2016_pb_1
  """

  __lookup = {
      ('Summer20UL16APV', 'GJets'): 19495.440,
      ('Summer20UL16APV', 'QCD'): 19497.914,
      ('Summer20UL16', 'GJets'): 16810.810,
      ('Summer20UL16', 'QCD'): 16810.813,
      ('Summer20UL17', 'GJets'): 41476.390,
      ('Summer20UL17', 'QCD'): 41471.589,
      ('Summer20UL18', 'GJets'): 59816.229,
      ('Summer20UL18', 'QCD'): 59817.406,
  }

  era, tag = sample_name.split('.')
  if (era, tag) not in __lookup:
    print(f"Unrecognized sample {sample_name}! returning 1")
  else:
    return __lookup[(era, tag)]


def modify_mc_events(events, dataset):
  # Pile up correctoins
  events['Weight'] = events.Weight * events.puWeight * __mc_lumi_weight(dataset)

  def reorder_jec(jets, jec):
    """
    Reordering JEC container such that jec.o will matched jets.origIndex. This is
    done by performing a pair wise comparison of the two arrays, assuming jets
    and jec has identical dimensions and unique matching.
    """
    _, jeco = ak.unzip(ak.cartesian([jets.origIndex, jec.o], nested=True))
    jeco = (jeco == events.Jets.origIndex)  # Pairwise matching
    new_args = ak.argmax(jeco, axis=-1)  # Getting index of 'True'
    return jec[new_args]  # Returning the ordered set.

  # Making a copy of the jets for a modification
  jets = events.Jets
  # JEC related
  jec_up = ((1 + events.Jets.jecUnc) * 1.0 / events.Jets.jerFactor)
  #reorder_jec(events.Jets, events.JetsJECup).j / events.Jets.jerFactor)
  jec_down = ((1 - events.Jets.jecUnc) * 1.0 / events.Jets.jerFactor)
  #reorder_jec(events.Jets, events.JetsJECdown).j /
  #events.Jets.jerFactor)
  jets['pt_jecup'] = jets.pt * jec_up
  jets['energy_jecup'] = jets.energy * jec_up
  jets['pt_jecdown'] = jets.pt * jec_down
  jets['energy_jecdown'] = jets.energy * jec_down
  events['HT_jecup'] = events.HT * ak_weighted_avg(jec_up, jets.pt)
  events['HT_jecdown'] = events.HT * ak_weighted_avg(jec_down, jets.pt)

  # JER related issues
  jer_up = events.Jets.jerFactorUp / events.Jets.jerFactor
  jer_down = events.Jets.jerFactorDown / events.Jets.jerFactor
  jets['pt_jerup'] = jets.pt * jer_up
  jets['energy_jerup'] = jets.energy * jer_up
  jets['pt_jerdown'] = jets.pt * jer_down
  jets['energy_jerdown'] = jets.energy * jer_down
  events['HT_jerup'] = events.HT * ak_weighted_avg(jer_up, jets.pt)
  events['HT_jerdown'] = events.HT * ak_weighted_avg(jer_down, jets.pt)

  # Pushing jets back into the events collection
  events['Jets'] = jets

  tracks = events.Tracks
  tracks['Weight'] = ak.ones_like(tracks.x)
  events['Tracks'] = tracks

  return events


class TriggerManager(object):
  def __init__(self, events):
    self.trigger_list = events.TriggerPass.layout.parameter('__doc__')
    self.trigger_list = self.trigger_list.split(',')

  def has_trigger(self, events, name):
    if name not in self.trigger_list:
      print(
          "Warning! trigger [{0}] is not stored in events' trigger list.".format(
              name))
      return events.HT >= 0
    else:
      return events.TriggerPass[:, self.trigger_list.index(name)] == 1


def dummy_sf(events, jets):
  return 1, 1


def lumi_unc(events, jets):
  return 1 + 0.025, 1 - 0.025


def pileup_unc(events, jets):
  return (events.puSysUp / events.puWeight), (events.puSysDown / events.puWeight)


def trigger_unc(events, jets):
  resmodel = cs.Correction(name="resmodel",
                           description="A jet energy resolution smearing model",
                           version=1,
                           inputs=[
                               cs.Variable(name="HT",
                                           type="real",
                                           description="JetHT"),
                           ],
                           output=cs.Variable(name="scale", type="real"),
                           data=cs.Binning(nodetype="binning",
                                           input="HT",
                                           edges=[1000, 1200, 1400, 1600, 2000],
                                           content=[1.0, 1.1, 1.2, 1.3],
                                           flow="clamp",
                                           ))
  ones = ak.ones_like(events.HT)
  return ones * 1.1, ones * 0.9
  # return ak.from_numpy(1 + sf_up / sf_cen), ak.from_numpy(1 - sf_lo / sf_cen)


def pdf_unc(events, jets):
  return 1.2, 0.8


def scale_unc(events, jets):
  return 1.001, 0.999


def dummy_unc(events, jets):
  return events, jets


_norm_smear_evaluator_ = cs.Correction(
    name="detrng",
    description="Deterministic random number generator.",
    version=0,
    inputs=[
        cs.Variable(name="pt",
                    type="real",
                    description="input pt (entropy source)"),
        cs.Variable(name="eta",
                    type="real",
                    description="input pseudorapdity (entropy source)"),
        cs.Variable(name="phi",
                    type="real",
                    description="input phi (entropy source)"),
    ],
    output=cs.Variable(name="rng", type="real"),
    data=cs.HashPRNG(nodetype="hashprng",
                     inputs=["pt", "eta", "phi"],
                     distribution="stdnormal")).to_evaluator()


class DetRNG(correctionlib_wrapper):
  def __init__(self):
    super().__init__(_norm_smear_evaluator_)


norm_rng = DetRNG()  # creating a default object for high level function


def randomize_norm(col, loc=0, scale=1.0):
  return norm_rng(col.pt, col.eta, col.phi) * scale + loc


def track_unc(events, jets):
  ipz_smear = 0.0001
  ip2d_smear = 0.000002

  tracks = jets.Tracks[:]  # We actually need to make a copy here
  tracks['ipz'] = randomize_norm(tracks, loc=tracks.ipz, scale=ipz_smear)
  tracks['IP2DPV0'] = randomize_norm(tracks,
                                     loc=tracks.IP2DPV0,
                                     scale=ip2d_smear)
  tracks['DN'] = (tracks.ipz**2 / 0.0001 + tracks.IP2DSigPV0**2)**0.5
  jets = jets[:]  # Actually need to make a copy here
  jets['Tracks'] = tracks
  return events, jets


def vertex_prompt_fraction(events, jets):
  ipz = np.abs(jets.Tracks.ipz[:])
  nprompt = ak.sum(ak.count(ipz[ipz < 0.01], axis=-1), axis=-1)
  ntotal = ak.sum(ak.count(ipz, axis=-1), axis=-1)
  ntotal = ak.where(ntotal == 0, ak.ones_like(ntotal), ntotal)
  return nprompt / ntotal


def JEC_unc_up(events, jets):
  jets['pt'] = jets.pt_jecup
  jets['energy'] = jets.energy_jecup
  events['HT'] = events.HT_jecup
  # Resorting the jets
  jets = jets[ak.argsort(jets.pt, axis=1, ascending=False)]
  return events, jets


def JEC_unc_lo(events, jets):
  jets['pt'] = jets.pt_jecdown
  jets['energy'] = jets.energy_jecdown
  events['HT'] = events.HT_jecdown
  jets = jets[ak.argsort(jets.pt, axis=1, ascending=False)]
  return events, jets


def JER_unc_up(events, jets):
  jets['pt'] = jets.pt_jerup
  jets['energy'] = jets.energy_jerup
  events['HT'] = events.HT_jerup
  jets = jets[ak.argsort(jets.pt, axis=1, ascending=False)]
  return events, jets


def JER_unc_lo(events, jets):
  jets['pt'] = jets.pt_jerdown
  jets['energy'] = jets.energy_jerdown
  events['HT'] = events.HT_jerdown
  jets = jets[ak.argsort(jets.pt, axis=1, ascending=False)]
  return events, jets


dict_B = {
    'central': dummy_unc,  # Needed
    'track_up': track_unc,
    'track_down': track_unc,
    'jec_up': JEC_unc_up,
    'jec_down': JEC_unc_lo,
    'jer_up': JER_unc_up,
    'jer_down': JER_unc_lo,
}
dict_A = {
    'lumi': lumi_unc,
    'pileup': pileup_unc,
    'trigger': trigger_unc,
    'pdf': pdf_unc,
    'scale': scale_unc,
    # 'tagger': eff_unc,
}


def run_sig_tag(events, jets, radius, ipz):
  if radius == 0.4:
    ip2d_cut = 10**-1.6 if ipz == 0.5 else 10**-1.4
    prompt_mask = (jets.Tracks.DN < 4)
    prompt_pt = ak.sum(ak.where(prompt_mask, jets.Tracks.pt,
                                ak.zeros_like(jets.Tracks.pt)),
                       axis=-1)
    sum_pt = ak.sum(jets.Tracks.pt, axis=-1)
    a3d = prompt_pt / sum_pt
    jet_tag = (jets.IP2DMedian > ip2d_cut) & (a3d < 0.25)
  else:
    pass_girth = jets.PtDeltaR > 0.05
    pass_tau21 = (jets.tau2 / jets.tau1) > 0.50

    ip2d_mask = (jets.Tracks.IP2DPV0 > 10**-2.2)
    pass_ip2d = ak.sum(ip2d_mask, axis=-1)

    mult_cut = 10 if ipz == 0.5 else 12

    jet_tag = (pass_girth) & (pass_ip2d > mult_cut) & (pass_tau21)

  sig = events[ak.sum(jet_tag, axis=-1) >= 2]  # Tagging selection
  sig = sig[sig.HT > 1500]
  sig = sig[(sig.GoodJets.pt[:, 0] >= 150) & (sig.GoodJets.pt[:, 1] >= 150) &
            (sig.GoodJets.pt[:, 2] >= 120) & (sig.GoodJets.pt[:, 3] >= 100)]
  return sig


if __name__ == '__main__':
  import dask

  client = Client()
  
  output = {
      'HT':
      hist.dask.Hist(hist.axis.StrCategory([], growth=True, name='dataset'),
                     hist.axis.StrCategory([], growth=True, name='tagname'),
                     hist.axis.Regular(20, 1000, 5000, name='HT'),
                     hist.axis.StrCategory([], growth=True, name='sys'),
                     storage=hist.storage.Weight())
  }

  def _fill_histogram(events, weight, sys, rad, ipz):
    output['HT'].fill(dataset='mydataset',
                      tagname=f'{rad:.1f}@{ipz:.1f}',
                      HT=events.HT,
                      sys=sys,
                      weight=weight)

  mevents = NanoEventsFactory.from_root(
      'file:' + os.path.abspath('./0_RA2AnalysisTree.root'), #'./treemaker_test_events100.root'),
      treepath='TreeMaker2/PreSelection',
      chunks_per_file=5,
      permit_dask=True,
      schemaclass=TreeMakerSchema).events()

  ak = awkward_switch(mevents)
  print(ak.__file__)

  trgmgr = TriggerManager(mevents)
  modify_mc_events(mevents, 'Summer20UL18.QCD')

  mevents = standard_vertex_selection(mevents)
  mevents = mevents[pass_MET_filters(mevents)]
  mevents = mevents[ak.sum(mevents.Jets.pt > 90, axis=-1) >= 4]

  for radius in [0.4, 0.8]:
    revents = mevents[:]
    revents['BaseJets'] = run_jet_track_matching(revents,
                                                 revents.Jets,
                                                 radius=radius)
    revents['PVTrackFraction'] = vertex_prompt_fraction(revents,
                                                        revents.BaseJets)

    for ipz in [0.5, 1.0]:

      for key_b, unc_b in dict_B.items():
        events, basejets = unc_b(revents, revents.BaseJets)
        # More event selections
        events['GoodJets'] = refine_jet_tracks(events, basejets, ipz)
        events['GoodJets'] = events.GoodJets[:, 0:4]
        events = events[events.PVTrackFraction > 0.1]
        events = events[ak.count(events.GoodJets.pt, axis=-1) >= 4]

        # Running tagging the signal region selections.
        signal = run_sig_tag(events, events.GoodJets, radius, ipz)

        if key_b != 'central':
          _fill_histogram(signal, signal.Weight, key_b, radius, ipz)
        else:
          _fill_histogram(signal, signal.Weight, 'central', radius, ipz)

          for key_a, unc_a in dict_A.items():
            w_up, w_down = unc_a(signal, signal.GoodJets)
            _fill_histogram(signal, signal.Weight * w_up, key_a + '_up', radius,
                            ipz)
            _fill_histogram(signal, signal.Weight * w_down, key_a + '_down',
                            radius, ipz)

  import time
  tic = time.monotonic()
  #opt = dask.optimize(output)[0]
  print("time opt:", time.monotonic() - tic)
            
  # Dummy loop for systematics for now
  #output['HT'].visualize(output='dask_graph_systematic_opt.pdf',
  #                       optimize_graph=True)

  #print(dask.compute(output, scheduler="sync"))
  tic = time.monotonic()
  #print(dask_awkward.necessary_columns(output))
  print("time col:", time.monotonic() - tic)
  #print(output['HT'].dask)
  # print(output['HT'].visualize(output='dask_graph_systematic.pdf'))
