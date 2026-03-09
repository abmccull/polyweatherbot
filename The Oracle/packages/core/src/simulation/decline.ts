import type { GameState, EventFeedItem } from "../state/gameState";
import type { LegacyState } from "../state/legacy";
import { createInitialLegacyState, computeLegacyScore, generateLegacyArtifact } from "../state/legacy";
import { getAbsoluteDay } from "./clock";

/** Decline severity threshold that triggers the terminal phase. */
const TERMINAL_THRESHOLD = 10;

/** Decline entry: average faction credibility below this. */
const CREDIBILITY_THRESHOLD = 25;

/** Decline entry: treasury gold below this. */
const GOLD_THRESHOLD = 30;

/** Low reputation tiers that qualify for decline. */
const LOW_TIERS = new Set(["obscure"]);

/** Resource capacity contraction rate per month during decline. */
const CONTRACTION_RATE = 0.05;

/**
 * Checks whether decline conditions are met.
 * Called at month boundaries.
 */
function shouldEnterDecline(state: GameState): boolean {
  const factions = Object.values(state.factions);
  const averageCredibility = factions.length > 0
    ? factions.reduce((sum, f) => sum + f.credibility, 0) / factions.length
    : 0;
  const goldLow = state.resources.gold.amount < GOLD_THRESHOLD;
  const reputationLow = LOW_TIERS.has(state.campaign.reputation.currentTier);

  return averageCredibility < CREDIBILITY_THRESHOLD && goldLow && reputationLow;
}

/**
 * Checks whether a recent major prophecy success occurred, which can
 * reduce decline severity or exit decline entirely.
 */
function checkComebackCondition(state: GameState): boolean {
  const absoluteDay = getAbsoluteDay(state.clock);
  const recentSuccess = state.consultation.history.some(
    (p) =>
      p.resolved &&
      (p.credibilityDelta ?? 0) >= 8 &&
      p.resolvedDay !== undefined &&
      absoluteDay - p.resolvedDay <= 30
  );
  return recentSuccess;
}

/**
 * Applies resource capacity contraction during decline.
 */
function contractResources(state: GameState): GameState {
  const resources = { ...state.resources };
  for (const [key, value] of Object.entries(resources) as Array<[keyof typeof resources, typeof resources[keyof typeof resources]]>) {
    resources[key] = {
      ...value,
      capacity: Math.max(10, Math.round(value.capacity * (1 - CONTRACTION_RATE))),
      amount: Math.min(value.amount, Math.max(10, Math.round(value.capacity * (1 - CONTRACTION_RATE))))
    };
  }
  return { ...state, resources };
}

/**
 * Advances decline mechanics. Should be called at month boundaries
 * (when state.clock.month changes).
 */
export function advanceDecline(state: GameState): GameState {
  const legacy = state.legacy ?? createInitialLegacyState();
  const absoluteDay = getAbsoluteDay(state.clock);
  const feedItems: EventFeedItem[] = [];

  // Phase: thriving — check if we should enter decline
  if (legacy.phase === "thriving") {
    if (shouldEnterDecline(state)) {
      const nextLegacy: LegacyState = {
        ...legacy,
        phase: "declining",
        declineStartDay: absoluteDay,
        declineSeverity: 1,
        legacyScore: computeLegacyScore(state)
      };
      feedItems.push({
        id: `event-decline-start-${absoluteDay}`,
        day: state.clock.day,
        text: "The oracle's influence wanes. Credibility, gold, and reputation have all fallen dangerously low. Decline has begun."
      });
      return {
        ...state,
        legacy: nextLegacy,
        eventFeed: [...feedItems, ...state.eventFeed].slice(0, 8)
      };
    }

    // Update score even when thriving
    return {
      ...state,
      legacy: {
        ...legacy,
        legacyScore: computeLegacyScore(state)
      }
    };
  }

  // Phase: declining — track severity and check for comeback or terminal
  if (legacy.phase === "declining") {
    // Check comeback condition
    if (checkComebackCondition(state)) {
      const newSeverity = Math.max(0, legacy.declineSeverity - 3);
      const comebackAttempts = legacy.comebackAttempts + 1;

      if (newSeverity <= 0) {
        // Exited decline
        feedItems.push({
          id: `event-comeback-${absoluteDay}`,
          day: state.clock.day,
          text: "A triumphant prophecy has reversed the oracle's decline. The sanctuary breathes again."
        });
        return {
          ...state,
          legacy: {
            ...legacy,
            phase: "thriving",
            declineSeverity: 0,
            comebackAttempts,
            legacyScore: computeLegacyScore(state)
          },
          eventFeed: [...feedItems, ...state.eventFeed].slice(0, 8)
        };
      }

      feedItems.push({
        id: `event-comeback-partial-${absoluteDay}`,
        day: state.clock.day,
        text: `A major prophecy success eases the decline (severity ${newSeverity}). There may yet be hope.`
      });
      return {
        ...contractResources(state),
        legacy: {
          ...legacy,
          declineSeverity: newSeverity,
          comebackAttempts,
          legacyScore: computeLegacyScore(state)
        },
        eventFeed: [...feedItems, ...state.eventFeed].slice(0, 8)
      };
    }

    // No comeback — severity increases
    const newSeverity = legacy.declineSeverity + 1;

    if (newSeverity >= TERMINAL_THRESHOLD) {
      // Enter terminal phase
      const artifact = generateLegacyArtifact(state);
      feedItems.push({
        id: `event-terminal-${absoluteDay}`,
        day: state.clock.day,
        text: "The oracle's decline is irreversible. The Pythia speaks her final words as the sanctuary prepares for silence."
      });
      return {
        ...contractResources(state),
        legacy: {
          ...legacy,
          phase: "terminal",
          declineSeverity: newSeverity,
          legacyScore: computeLegacyScore(state),
          legacyArtifact: artifact
        },
        eventFeed: [...feedItems, ...state.eventFeed].slice(0, 8)
      };
    }

    feedItems.push({
      id: `event-decline-progress-${absoluteDay}`,
      day: state.clock.day,
      text: `The decline deepens (severity ${newSeverity}/${TERMINAL_THRESHOLD}). Resources contract and pilgrims grow scarce.`
    });
    return {
      ...contractResources(state),
      legacy: {
        ...legacy,
        declineSeverity: newSeverity,
        legacyScore: computeLegacyScore(state)
      },
      eventFeed: [...feedItems, ...state.eventFeed].slice(0, 8)
    };
  }

  // Phase: terminal — check for endless mode
  if (state.endlessMode) {
    // In endless mode, skip terminal and revert to declining with reduced severity
    feedItems.push({
      id: `event-endless-renewal-${absoluteDay}`,
      day: state.clock.day,
      text: "The Oracle Renewal: the sanctuary draws on deep reserves of sacred memory. Decline is halted and the cycle begins anew."
    });
    return {
      ...state,
      legacy: {
        ...legacy,
        phase: "declining",
        declineSeverity: Math.max(1, legacy.declineSeverity - 5),
        legacyScore: computeLegacyScore(state),
        legacyArtifact: undefined
      },
      eventFeed: [...feedItems, ...state.eventFeed].slice(0, 8)
    };
  }

  return {
    ...state,
    legacy: {
      ...legacy,
      legacyScore: computeLegacyScore(state)
    }
  };
}
