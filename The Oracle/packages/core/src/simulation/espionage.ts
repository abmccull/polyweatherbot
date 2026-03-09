import type { GameState, PriestSecret, PriestSecretKind } from "../state/gameState";
import type { EspionageOperation, EspionageState, CounterIntelEvent } from "../state/espionage";
import { createInitialEspionageState } from "../state/espionage";

function hash(seed: number, salt: string): number {
  let value = seed ^ 0x9e3779b9;
  for (let index = 0; index < salt.length; index += 1) {
    value = Math.imul(value ^ salt.charCodeAt(index), 0x45d9f3b);
    value ^= value >>> 15;
  }
  return (value >>> 0) / 4294967296;
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

const SECRET_KINDS: PriestSecretKind[] = ["corruption", "heresy", "rivalry", "forbidden_knowledge"];

function resolveOperation(
  state: GameState,
  operation: EspionageOperation,
  espionage: EspionageState
): { operation: EspionageOperation; counterIntelEvent?: CounterIntelEvent } {
  const agent = espionage.agents.find((a) => a.id === operation.agentId);
  if (!agent) {
    return {
      operation: { ...operation, status: "failed", result: "Agent lost." }
    };
  }

  const rivalIntel = state.rivalOracles?.roster.find((r) => r.id === operation.targetId)?.intel ?? 30;
  const successRoll = hash(state.worldSeed + operation.startDay * 17, `${operation.id}:resolve`);
  const successThreshold = (agent.skill * 0.6 + espionage.networkStrength * 0.2 - rivalIntel * 0.3) / 100;
  const succeeded = successRoll < clamp(successThreshold + 0.3, 0.15, 0.85);

  const detectionRoll = hash(state.worldSeed + operation.startDay * 23, `${operation.id}:detect`);
  const detectionThreshold = (rivalIntel * 0.4 - agent.skill * 0.25 - espionage.networkStrength * 0.1) / 100;
  const detected = detectionRoll < clamp(detectionThreshold + 0.1, 0.05, 0.5);

  let result: string;
  switch (operation.kind) {
    case "intercept_prophecy":
      result = succeeded ? "Intercepted a rival prophecy, gaining strategic insight." : "The prophecy slipped through unnoticed.";
      break;
    case "plant_false_omen":
      result = succeeded ? "A false omen has been planted, sowing confusion among rivals." : "The false omen was seen through.";
      break;
    case "recruit_informant":
      result = succeeded ? "A new informant has been turned inside the target faction." : "The recruitment attempt was rebuffed.";
      break;
    case "sabotage_rival":
      result = succeeded ? "Rival operations disrupted through careful sabotage." : "Sabotage failed; the rival remains undisturbed.";
      break;
    case "protect_oracle":
      result = succeeded ? "Oracle defenses strengthened against infiltration." : "Protection efforts yielded no tangible benefit.";
      break;
    case "seed_philosopher":
      result = succeeded ? "A philosopher now carries ideas favorable to Delphi." : "The philosopher proved resistant to influence.";
      break;
    default:
      result = succeeded ? "Operation completed." : "Operation failed.";
  }

  let counterIntelEvent: CounterIntelEvent | undefined;
  if (detected) {
    counterIntelEvent = {
      id: `counter-intel-${state.clock.day}-${operation.id}`,
      day: state.clock.day,
      description: `Agent ${agent.name}'s ${operation.kind.replace(/_/g, " ")} operation was detected by counter-intelligence.`,
      severityDelta: succeeded ? 3 : 6
    };
  }

  return {
    operation: {
      ...operation,
      status: detected ? "detected" : succeeded ? "success" : "failed",
      result
    },
    counterIntelEvent
  };
}

function generatePriestSecrets(state: GameState): GameState {
  const day = state.clock.day;
  let priests = state.priests;
  let changed = false;

  for (let index = 0; index < priests.length; index++) {
    const priest = priests[index]!;
    const roll = hash(state.worldSeed + day * 31 + index * 53, `${priest.id}:secret`);
    // ~8% chance per month per priest
    if (roll > 0.08) {
      continue;
    }

    const kindRoll = hash(state.worldSeed + day * 41 + index * 67, `${priest.id}:secret-kind`);
    const kind = SECRET_KINDS[Math.floor(kindRoll * SECRET_KINDS.length)]!;
    const severity = Math.round(20 + hash(state.worldSeed + day * 59 + index * 71, `${priest.id}:severity`) * 60);
    const existingSecrets = priest.secrets ?? [];

    // Don't pile up too many secrets
    if (existingSecrets.length >= 3) {
      continue;
    }

    const newSecret: PriestSecret = {
      id: `secret-${priest.id}-${day}`,
      kind,
      severity
    };

    if (!changed) {
      priests = [...priests];
      changed = true;
    }
    priests[index] = {
      ...priest,
      secrets: [...existingSecrets, newSecret]
    };
  }

  return changed ? { ...state, priests } : state;
}

function evaluateSuccessionContest(state: GameState): GameState {
  const politics = state.priestPolitics;
  if (!politics) {
    return state;
  }

  // Trigger succession contest when dominant bloc shifts AND high pressure
  const highPressure = politics.overallPressure >= 65;
  const existingContest = politics.successionContest;

  if (existingContest?.active) {
    // Resolve contest after 15 days
    if (state.clock.day - existingContest.startDay >= 15) {
      const frontRunner = existingContest.frontRunnerId ?? existingContest.candidates[0];
      return {
        ...state,
        priestPolitics: {
          ...politics,
          successionContest: {
            ...existingContest,
            active: false
          },
          overallPressure: Math.max(0, politics.overallPressure - 10)
        },
        priests: state.priests.map((p, i) => ({
          ...p,
          successionRank: p.id === frontRunner ? 1 : i + 2
        }))
      };
    }
    return state;
  }

  // Start new contest if conditions are met
  if (highPressure && !existingContest) {
    const candidates = state.priests
      .filter((p) => {
        const profile = politics.priests[p.id];
        return profile && profile.influence >= 40;
      })
      .map((p) => p.id);

    if (candidates.length >= 2) {
      const profiles = candidates.map((id) => ({
        id,
        score: (politics.priests[id]?.influence ?? 0) + (politics.priests[id]?.loyalty ?? 0) * 0.3
      }));
      profiles.sort((a, b) => b.score - a.score);

      return {
        ...state,
        priestPolitics: {
          ...politics,
          successionContest: {
            active: true,
            candidates,
            frontRunnerId: profiles[0]?.id,
            startDay: state.clock.day
          }
        }
      };
    }
  }

  return state;
}

export function advanceEspionage(state: GameState): GameState {
  const espionage = state.espionage ?? createInitialEspionageState();
  const day = state.clock.day;

  // Progress and resolve completed operations
  let nextOperations = espionage.operations;
  const newCounterIntelEvents: CounterIntelEvent[] = [];
  let changed = false;

  for (let index = 0; index < nextOperations.length; index++) {
    const operation = nextOperations[index]!;
    if (operation.status !== "active") {
      continue;
    }

    if (day >= operation.startDay + operation.duration) {
      if (!changed) {
        nextOperations = [...nextOperations];
        changed = true;
      }
      const resolution = resolveOperation(state, operation, espionage);
      nextOperations[index] = resolution.operation;
      if (resolution.counterIntelEvent) {
        newCounterIntelEvents.push(resolution.counterIntelEvent);
      }
    }
  }

  // Counter-intelligence: rivals may compromise agents
  let nextAgents = espionage.agents;
  let agentsChanged = false;
  for (let index = 0; index < nextAgents.length; index++) {
    const agent = nextAgents[index]!;
    if (agent.compromised) {
      continue;
    }
    const rivalIntel = state.rivalOracles?.roster
      .filter((r) => r.patronFactionId === agent.targetFactionId)
      .reduce((max, r) => Math.max(max, r.intel), 0) ?? 0;
    const compromiseRoll = hash(state.worldSeed + day * 37 + index * 43, `${agent.id}:compromise`);
    const compromiseThreshold = (rivalIntel * 0.3 - agent.skill * 0.2 - agent.loyalty * 0.1) / 100;
    if (compromiseRoll < clamp(compromiseThreshold, 0.01, 0.12)) {
      if (!agentsChanged) {
        nextAgents = [...nextAgents];
        agentsChanged = true;
      }
      nextAgents[index] = { ...agent, compromised: true };
      newCounterIntelEvents.push({
        id: `counter-intel-compromised-${day}-${agent.id}`,
        day,
        description: `Agent ${agent.name} has been compromised while operating as a ${agent.cover}.`,
        severityDelta: 8
      });
    }
  }

  // Update network strength
  const activeAgentCount = nextAgents.filter((a) => !a.compromised).length;
  const successfulOps = nextOperations.filter((op) => op.status === "success").length;
  const detectedOps = nextOperations.filter((op) => op.status === "detected").length;
  const networkStrength = clamp(
    Math.round(
      10 + activeAgentCount * 8 + successfulOps * 3 - detectedOps * 5
      - newCounterIntelEvents.reduce((sum, e) => sum + e.severityDelta, 0) * 0.5
    ),
    0,
    100
  );

  const nextEspionage: EspionageState = {
    agents: nextAgents,
    operations: nextOperations.slice(-50), // Keep last 50 operations
    counterIntelEvents: [...newCounterIntelEvents, ...espionage.counterIntelEvents].slice(0, 20),
    networkStrength
  };

  let nextState: GameState = {
    ...state,
    espionage: nextEspionage
  };

  // Apply operation effects to rival oracles
  for (const op of nextOperations) {
    if (op.status !== "success") continue;
    // Only process newly resolved ops (those that resolved this tick)
    const wasActive = espionage.operations.find((o) => o.id === op.id);
    if (wasActive?.status !== "active") continue;

    if (op.kind === "sabotage_rival" && nextState.rivalOracles) {
      nextState = {
        ...nextState,
        rivalOracles: {
          ...nextState.rivalOracles,
          roster: nextState.rivalOracles.roster.map((r) =>
            r.id === op.targetId
              ? { ...r, pressure: Math.max(0, r.pressure - 8), intrigue: Math.max(0, r.intrigue - 5) }
              : r
          )
        }
      };
    }
    if (op.kind === "intercept_prophecy" && nextState.rivalOracles) {
      nextState = {
        ...nextState,
        rivalOracles: {
          ...nextState.rivalOracles,
          roster: nextState.rivalOracles.roster.map((r) =>
            r.id === op.targetId
              ? { ...r, intel: Math.min(100, r.intel + 10), visibility: Math.min(100, r.visibility + 6) }
              : r
          )
        }
      };
    }
    if (op.kind === "protect_oracle") {
      nextState = {
        ...nextState,
        pythia: {
          ...nextState.pythia,
          attunement: Math.min(100, nextState.pythia.attunement + 2),
          prestige: Math.min(100, nextState.pythia.prestige + 1)
        }
      };
    }
  }

  // Add feed items for counter-intel events
  if (newCounterIntelEvents.length > 0) {
    nextState = {
      ...nextState,
      eventFeed: [
        ...newCounterIntelEvents.slice(0, 2).map((event) => ({
          id: `event-${event.id}`,
          day,
          text: event.description
        })),
        ...nextState.eventFeed
      ].slice(0, 8)
    };
  }

  // Generate priest secrets (monthly)
  nextState = generatePriestSecrets(nextState);

  // Evaluate succession contest
  nextState = evaluateSuccessionContest(nextState);

  return nextState;
}
