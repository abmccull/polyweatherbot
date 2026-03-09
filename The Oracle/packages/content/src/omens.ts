import type { OmenDef } from "./schema";

export const omenDefs: OmenDef[] = [
  {
    id: "ornithomancy",
    priestRole: "Augur",
    family: "military",
    templates: [
      "Three eagles crossed the valley at dawn; one turned back in silence.",
      "A hawk wheeled above the altar and vanished toward the sea.",
      "The birds broke formation when the sun touched the ridge.",
      "A raven settled on the laurel, cried once, and would not face the east.",
      "Two kites fought above the processional road until both dropped from sight.",
      "A lone falcon kept ahead of the envoy's litter and never once looked down."
    ]
  },
  {
    id: "pyromancy",
    priestRole: "Flame Keeper",
    family: "spiritual",
    templates: [
      "The flame rose blue at its root, then bent toward the west.",
      "Smoke gathered low before lifting cleanly through the vent.",
      "Sparks leapt twice from the brazier and died in the dust.",
      "The ash held a red seam long after the wood should have failed.",
      "The coals split in a circle and left a dark heart at the center.",
      "The brazier hissed as if rain had touched it, though the roof stones were dry."
    ]
  },
  {
    id: "hydromancy",
    priestRole: "Spring Warden",
    family: "economic",
    templates: [
      "The spring clouded at noon and cleared by the second hour.",
      "Ripples ran upstream against the breeze.",
      "The basin held still water until a single ring widened outward.",
      "A silver sheen crossed the basin and vanished before the hymn was done.",
      "The water struck the stone with a hollow tone before running sweet again.",
      "A thread of silt curled like a road across the basin and then dissolved."
    ]
  },
  {
    id: "extispicy",
    priestRole: "Sacrificial Priest",
    family: "ritual",
    templates: [
      "The liver showed a bright lobe and a broken edge on the same side.",
      "The gall bladder lay shrunken, but the veins were full and dark.",
      "The heart marked a clean notch where the knife should never have passed.",
      "The sacrificial smoke clung low while the entrails gleamed unnaturally pale.",
      "The lobe nearest the gate of fate was thick, yet the channel beneath it was hollow.",
      "The victim's tongue darkened after death, though the flesh remained warm."
    ]
  },
  {
    id: "oneiromancy",
    priestRole: "Dream Priest",
    family: "spiritual",
    templates: [
      "The sleeper saw a bronze door open onto surf without shore.",
      "A crowned child walked through cedar smoke and would not speak.",
      "The dream returned three times, each time with fewer stars above it.",
      "A serpent coiled around the tripod and drank from an unseen spring.",
      "The envoy's face appeared in the dream, but wore another city's eyes.",
      "The Pythia woke with laurel leaves clenched in her hand and no memory of taking them."
    ]
  },
  {
    id: "astromancy",
    priestRole: "Astronomer",
    family: "military",
    templates: [
      "Mars burned low beside the moon before dawn.",
      "A red star held the ridge line longer than the charts allowed.",
      "The Dog Star blurred while the western horizon stayed hard and clear.",
      "A wandering light crossed the temple roof as the last watch ended.",
      "The constellation of the ship dipped before midnight and rose crooked.",
      "Saturn stood untroubled over the mountain while the eastern stars thinned."
    ]
  },
  {
    id: "chresmology",
    priestRole: "Scholar",
    family: "spiritual",
    templates: [
      "An older oracle verse matched the envoy's question only in its final line.",
      "The cedar tablets agreed on the omen but differed on the cost.",
      "A worm-eaten scroll preserved the warning and lost the blessing beside it.",
      "Three prior verses named the same river, yet not the same city.",
      "The oldest hymn spoke the answer obliquely, but repeated the number twice.",
      "A margin note from a dead priest contradicted the public verse."
    ]
  }
];
