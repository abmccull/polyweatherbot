import { createContext, useContext } from "react";
import type { BurdenId, EspionageAgentCover, EspionageOperationKind, FactionId, LegendaryConsultationId, OriginId, PlacementTool } from "@the-oracle/core";

export type GameDispatchActions = {
  onSetTool: (tool: PlacementTool) => void;
  onSetSpeed: (speed: 0 | 1 | 2 | 3) => void;
  onAssignPriest: (priestId: string, buildingId: string) => void;
  onStartConsultation: () => void;
  onPurchaseTradeOffer: (offerId: string) => void;
  onRestPythia: () => void;
  onPurifyPythia: () => void;
  onRepairBuilding: (buildingId: string) => void;
  onOpenRunSetup: () => void;
  onSave: () => void;
  onLoad: () => void;
  setActiveOverlay: (overlay: string | null) => void;
  onIssuePriestDecree: (decreeType: "calm" | "reform" | "investigate") => void;
  onDismissPriest: (priestId: string) => void;
  onEndorseBloc: (blocId: string) => void;
  onBeginExcavation: (siteId: string) => void;
  onClaimRelic: (siteId: string, layerDepth: number) => void;
  onActivateSacredSite: (siteId: string) => void;
  onLaunchEspionageOperation: (operationKind: EspionageOperationKind, agentId: string, targetId: string) => void;
  onInvestigatePriest: (priestId: string) => void;
  onRecruitAgent: (cover: EspionageAgentCover, targetFactionId: FactionId) => void;
  onTriggerEndOfRun: () => void;
  onBeginLegendaryConsultation: (consultationId: LegendaryConsultationId) => void;
  onAdvanceLegendaryStage: (consultationId: LegendaryConsultationId) => void;
  onStartNewLineageRun: (originId: OriginId, seedText: string, burdens: BurdenId[], endlessMode: boolean) => void;
  onRecordLineageRun: () => void;
};

export const GameDispatchContext =
  createContext<GameDispatchActions | null>(null);

export function useGameDispatch(): GameDispatchActions {
  const value = useContext(GameDispatchContext);
  if (value === null) {
    throw new Error(
      "useGameDispatch must be used within a GameDispatchContext.Provider",
    );
  }
  return value;
}
