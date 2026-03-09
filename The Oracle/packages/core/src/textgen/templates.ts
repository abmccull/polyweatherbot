/**
 * Deterministic template-based text generation for atmospheric game narration.
 * All randomness is seeded for reproducibility.
 */

export type TextTemplate = {
  id: string;
  category: string;
  patterns: string[];
  variables: Record<string, string[]>;
};

// --- Seeded RNG ---

function seededHash(seed: number): number {
  let value = seed + 0x6d2b79f5;
  value = Math.imul(value ^ (value >>> 15), value | 1);
  value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
  return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
}

function seededPick<T>(items: readonly T[], seed: number): T {
  return items[Math.floor(seededHash(seed) * items.length) % items.length]!;
}

// --- Templates ---

export const templates: TextTemplate[] = [
  // Prophecy delivery narration
  {
    id: "prophecy-delivery",
    category: "prophecy",
    patterns: [
      "The Pythia {breathes} upon the tripod, and {vapor} rises as she speaks to the envoy of {faction}. {atmosphere}",
      "{atmosphere} The sacred {vapor} parts, and the Pythia's voice {echoes} through the sanctum as {faction}'s envoy listens. {closing}",
      "Before the envoy of {faction}, the Pythia {breathes} deeply. {vapor} coils around the tripod. {atmosphere} {closing}"
    ],
    variables: {
      breathes: ["inhales the sacred fumes", "draws the pneuma deep", "surrenders to the vapours", "trembles upon the seat"],
      vapor: ["laurel smoke", "sacred mist", "ethylene haze", "the god's breath", "coiling vapour"],
      echoes: ["rings", "reverberates", "trembles", "resonates", "pierces the silence"],
      atmosphere: [
        "The torchlight flickers as if the gods themselves lean close.",
        "A tremor passes through the stone floor of the adyton.",
        "The eternal flame gutters and roars anew.",
        "Shadows dance upon the temple walls like the shades of kings long dead.",
        "The scent of laurel and incense thickens until the air itself seems divine."
      ],
      closing: [
        "What was spoken cannot be unspoken.",
        "The envoy departs in troubled silence.",
        "The words settle like ash after sacrifice.",
        "Apollo has spoken through his servant."
      ]
    }
  },

  // Advisor messages
  {
    id: "advisor-observation",
    category: "advisor",
    patterns: [
      "{title} {observes}: \"{observation}\"",
      "\"{observation}\" -- {title} {observes} from the portico.",
      "{title} {observes}, remarking that \"{observation}\""
    ],
    variables: {
      title: ["The High Priest", "The Hierophant", "The Temple Steward", "The Elder Augur"],
      observes: ["notes with concern", "remarks quietly", "observes with narrowed eyes", "speaks gravely"],
      observation: [
        "the pilgrims grow fewer with each passing moon",
        "the treasuries of the city-states swell with ambition",
        "the philosophers sharpen their arguments like bronze blades",
        "rival oracles whisper in courts that once belonged to us"
      ]
    }
  },

  // Event descriptions
  {
    id: "event-general",
    category: "event",
    patterns: [
      "{prelude} {action}. {consequence}",
      "{action}. {prelude} {consequence}",
      "{prelude} {consequence} {action}."
    ],
    variables: {
      prelude: [
        "Word arrives from the agora:",
        "Smoke rises on the horizon --",
        "Messengers bring tidings:",
        "The stars foretell change --",
        "Tremors in the political order:"
      ],
      action: [
        "a great fleet sets sail",
        "treaties are signed in blood and wine",
        "the harvest fails across the lowlands",
        "a king falls upon the battlefield"
      ],
      consequence: [
        "The balance of power shifts.",
        "Delphi must respond with wisdom.",
        "The oracle's word carries new weight.",
        "The faithful look to the sanctuary for guidance."
      ]
    }
  },

  // Faction actions
  {
    id: "faction-action",
    category: "faction",
    patterns: [
      "{faction} {action}. {reaction}",
      "The agents of {faction} {action}, and {reaction}",
      "{reaction} {faction} {action}."
    ],
    variables: {
      action: [
        "marshals its armies at the border",
        "sends a delegation bearing gifts of gold",
        "withdraws its trade caravans in protest",
        "demands a new consultation at the sanctuary",
        "forges an alliance with a distant power"
      ],
      reaction: [
        "The other city-states watch with unease.",
        "Delphi's halls buzz with speculation.",
        "The envoys murmur of shifting allegiances.",
        "The oracle's counsel is sought more urgently than ever."
      ]
    }
  },

  // Age transitions
  {
    id: "age-transition",
    category: "age",
    patterns: [
      "{herald} {description} {portent}",
      "{portent} {herald} {description}",
      "{description} {herald} {portent}"
    ],
    variables: {
      herald: [
        "A new age dawns over the Greek world.",
        "The wheel of the ages turns once more.",
        "Time itself bends at Delphi's threshold.",
        "The old era crumbles; a new one rises from its ashes."
      ],
      description: [
        "The very air seems to shift, carrying new scents from distant lands.",
        "Kings and generals who once seemed eternal fade from memory.",
        "Trade routes that sustained empires are remapped by ambitious powers.",
        "The gods themselves seem to lean closer, or perhaps withdraw."
      ],
      portent: [
        "The oracle must adapt or be forgotten.",
        "What was true yesterday may be heresy tomorrow.",
        "New questions demand new wisdom.",
        "The Pythia feels the tremor of change in her bones."
      ]
    }
  },

  // Legendary consultation flavor
  {
    id: "legendary-arrival",
    category: "legendary",
    patterns: [
      "{arrival} {impression} {stakes}",
      "{impression} {arrival} {stakes}",
      "{arrival} {stakes} {impression}"
    ],
    variables: {
      arrival: [
        "A figure of legend approaches the sanctuary gates.",
        "The sacred way trembles beneath the weight of history.",
        "Guards stiffen as a name ripples through the crowd.",
        "Even the Pythia pauses mid-breath as word reaches the adyton."
      ],
      impression: [
        "Gold and iron glint in equal measure upon the visitor.",
        "The air thickens with the weight of destiny.",
        "Servants scatter offerings before the approaching retinue.",
        "The visitor's eyes hold the ambition of empires."
      ],
      stakes: [
        "This consultation will be remembered for a thousand years.",
        "The fate of nations hangs upon the oracle's next words.",
        "History leans forward, waiting.",
        "Whatever is spoken here will echo through the ages."
      ]
    }
  },

  // Contradiction narration
  {
    id: "contradiction-detected",
    category: "prophecy",
    patterns: [
      "{alarm} {detail} {gravity}",
      "{detail} {alarm} {gravity}",
      "{alarm} {gravity} {detail}"
    ],
    variables: {
      alarm: [
        "The sacred records reveal a terrible inconsistency.",
        "Scholars in the library gasp as they cross-reference the prophecies.",
        "A tremor of doubt runs through the priesthood.",
        "The High Priest discovers a fracture in the divine narrative."
      ],
      detail: [
        "Two prophecies speak with forked tongue on the same matter.",
        "What was promised to one faction now contradicts what was told to another.",
        "The oracle's words, when laid side by side, war with each other.",
        "Apollo's message, it seems, has been garbled in transmission."
      ],
      gravity: [
        "Credibility hangs by a thread.",
        "If the factions discover this, Delphi's authority will crumble.",
        "The contradiction must be resolved before it becomes scandal.",
        "This is the kind of error that topples oracles."
      ]
    }
  },

  // Arc milestone narration
  {
    id: "arc-milestone",
    category: "prophecy",
    patterns: [
      "{progression} {detail} {assessment}",
      "{detail} {progression} {assessment}",
      "{assessment} {progression} {detail}"
    ],
    variables: {
      progression: [
        "Another stage of the prophecy unfolds as foretold.",
        "Events in the wider world align with the oracle's words.",
        "The arc of prophecy bends toward fulfillment.",
        "A milestone in the divine narrative is reached."
      ],
      detail: [
        "Reports from the field confirm what the Pythia spoke.",
        "Envoys arrive bearing news that validates the sacred reading.",
        "The stars themselves seem to nod in agreement.",
        "Even the skeptics fall silent as events confirm the oracle."
      ],
      assessment: [
        "The oracle's credibility strengthens.",
        "Delphi's reputation grows with each fulfilled word.",
        "The factions take renewed interest in consultation.",
        "History records another mark in the oracle's favour."
      ]
    }
  },

  // Revolution narration
  {
    id: "revolution",
    category: "world",
    patterns: [
      "{upheaval} {faction} {aftermath}",
      "{faction} {upheaval} {aftermath}",
      "{upheaval} {aftermath} {faction}"
    ],
    variables: {
      upheaval: [
        "The old order collapses in fire and fury.",
        "A revolution sweeps through the city like a winter storm.",
        "The mob has risen; thrones topple and blood stains the agora.",
        "What was stable for generations shatters in a single season."
      ],
      aftermath: [
        "New rulers look to the oracle for legitimacy.",
        "The change sends tremors across the Greek world.",
        "Trade routes are disrupted and alliances redrawn.",
        "The dust settles to reveal a new political landscape."
      ]
    }
  }
];

const templateById = new Map(templates.map((t) => [t.id, t]));

/**
 * Renders a template with the given context and seeded RNG.
 * Context values like {faction} are replaced directly; template variables
 * like {vapor} are selected deterministically from the variable pool.
 */
export function renderTemplate(
  templateId: string,
  context: Record<string, string>,
  seed: number
): string {
  const template = templateById.get(templateId);
  if (!template) return "";

  const pattern = seededPick(template.patterns, seed);
  let result = pattern;

  // Replace template variables with seeded selections
  const variableRegex = /\{(\w+)\}/g;
  let match: RegExpExecArray | null;
  let iterationSeed = seed + 100;

  // First pass: collect all variable slots
  const slots: Array<{ key: string; fullMatch: string }> = [];
  while ((match = variableRegex.exec(pattern)) !== null) {
    slots.push({ key: match[1]!, fullMatch: match[0]! });
  }

  // Second pass: replace
  for (const slot of slots) {
    iterationSeed += 37;
    // Context values take priority over template variables
    if (context[slot.key] !== undefined) {
      result = result.replace(slot.fullMatch, context[slot.key]!);
    } else if (template.variables[slot.key]) {
      const selected = seededPick(template.variables[slot.key]!, iterationSeed);
      result = result.replace(slot.fullMatch, selected);
    }
  }

  return result;
}
