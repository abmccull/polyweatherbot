import type { FactionId } from "./gameState";

export type EspionageAgentCover = "merchant" | "pilgrim" | "priest" | "scholar";

export type EspionageOperationKind =
  | "intercept_prophecy"
  | "plant_false_omen"
  | "recruit_informant"
  | "sabotage_rival"
  | "protect_oracle"
  | "seed_philosopher";

export type EspionageAgent = {
  id: string;
  name: string;
  cover: EspionageAgentCover;
  targetFactionId: FactionId;
  skill: number;
  loyalty: number;
  compromised: boolean;
  recruitedDay: number;
};

export type EspionageOperationStatus = "active" | "success" | "failed" | "detected";

export type EspionageOperation = {
  id: string;
  kind: EspionageOperationKind;
  agentId: string;
  targetId: string;
  startDay: number;
  duration: number;
  status: EspionageOperationStatus;
  result?: string;
};

export type CounterIntelEvent = {
  id: string;
  day: number;
  description: string;
  severityDelta: number;
};

export type EspionageState = {
  agents: EspionageAgent[];
  operations: EspionageOperation[];
  counterIntelEvents: CounterIntelEvent[];
  networkStrength: number;
};

export function createInitialEspionageState(): EspionageState {
  return {
    agents: [],
    operations: [],
    counterIntelEvents: [],
    networkStrength: 10
  };
}
