# Player-centric Causal History Stage A

This experiment freezes the shared and main Partial-L2 paths and trains only
the private Player branch. `PH-HistK5` means a five-event personal sequence
plus cumulative causal current-match statistics. `PH-Shuffled-Team` uses a
fixed within-team derangement, while `PH-Null` removes values, masks, lengths,
and seen-state information.

The validation decision is locked before full-test loading. This experiment
does not feed Player history or predictions into Event or Position heads.

