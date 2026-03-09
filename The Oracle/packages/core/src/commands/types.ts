import type { LegendaryConsultationId, OriginId } from "@the-oracle/content";

import type { BuildingDefId, Coord, FactionId, PlacementTool } from "../state/gameState";
import type { EspionageAgentCover, EspionageOperationKind } from "../state/espionage";
import type { BurdenId } from "../state/lineage";

export type PlaceRoadCommand = {
  type: "PlaceRoadCommand";
  tile: Coord;
};

export type PlaceBuildingCommand = {
  type: "PlaceBuildingCommand";
  defId: Exclude<BuildingDefId, "sacred_way">;
  tile: Coord;
};

export type AssignPriestCommand = {
  type: "AssignPriestCommand";
  priestId: string;
  buildingId: string;
};

export type SetGameSpeedCommand = {
  type: "SetGameSpeedCommand";
  speed: 0 | 1 | 2 | 3;
};

export type AdvanceTickCommand = {
  type: "AdvanceTickCommand";
  ticks: number;
};

export type StartConsultationCommand = {
  type: "StartConsultationCommand";
  consultationId?: string;
};

export type PlaceProphecyTileCommand = {
  type: "PlaceProphecyTileCommand";
  tileId: string;
};

export type RemoveProphecyTileCommand = {
  type: "RemoveProphecyTileCommand";
  tileId: string;
};

export type DeliverProphecyCommand = {
  type: "DeliverProphecyCommand";
};

export type SaveGameCommand = {
  type: "SaveGameCommand";
  slotId: string;
};

export type LoadGameCommand = {
  type: "LoadGameCommand";
  slotId: string;
};

export type PurchaseTradeOfferCommand = {
  type: "PurchaseTradeOfferCommand";
  offerId: string;
};

export type RestPythiaCommand = {
  type: "RestPythiaCommand";
};

export type PurifyPythiaCommand = {
  type: "PurifyPythiaCommand";
};

export type SetToolCommand = {
  type: "SetToolCommand";
  tool: PlacementTool;
};

export type SelectEntityCommand = {
  type: "SelectEntityCommand";
  entityId?: string;
  entityKind?: "building" | "walker";
};

export type HoverTileCommand = {
  type: "HoverTileCommand";
  tile?: Coord;
};

export type InjectScenarioCommand = {
  type: "InjectScenarioCommand";
  scenario: "foundation" | "low-incense" | "consultation-ready" | "logistics-lab" | "campaign-lab" | "world-map-lab";
};

export type RepairBuildingCommand = {
  type: "RepairBuildingCommand";
  buildingId: string;
};

export type StartNewRunCommand = {
  type: "StartNewRunCommand";
  seed?: number | string;
  originId?: OriginId;
};

export type IssuePriestDecreeCommand = {
  type: "IssuePriestDecreeCommand";
  decreeType: "calm" | "reform" | "investigate";
};

export type DismissPriestCommand = {
  type: "DismissPriestCommand";
  priestId: string;
};

export type EndorseBlocCommand = {
  type: "EndorseBlocCommand";
  blocId: string;
};

export type BeginExcavationCommand = {
  type: "BeginExcavationCommand";
  siteId: string;
};

export type ClaimRelicCommand = {
  type: "ClaimRelicCommand";
  siteId: string;
  layerDepth: number;
};

export type ActivateSacredSiteCommand = {
  type: "ActivateSacredSiteCommand";
  siteId: string;
};

export type LaunchEspionageOperationCommand = {
  type: "LaunchEspionageOperationCommand";
  operationKind: EspionageOperationKind;
  agentId: string;
  targetId: string;
};

export type InvestigatePriestCommand = {
  type: "InvestigatePriestCommand";
  priestId: string;
};

export type RecruitAgentCommand = {
  type: "RecruitAgentCommand";
  cover: EspionageAgentCover;
  targetFactionId: FactionId;
};

export type TriggerEndOfRunCommand = {
  type: "TriggerEndOfRunCommand";
};

export type BeginLegendaryConsultationCommand = {
  type: "BeginLegendaryConsultationCommand";
  consultationId: LegendaryConsultationId;
};

export type AdvanceLegendaryStageCommand = {
  type: "AdvanceLegendaryStageCommand";
  consultationId: LegendaryConsultationId;
};

export type StartNewLineageRunCommand = {
  type: "StartNewLineageRunCommand";
  originId: OriginId;
  seedText: string;
  burdens: BurdenId[];
  endlessMode: boolean;
};

export type RecordLineageRunCommand = {
  type: "RecordLineageRunCommand";
};

export type GameCommand =
  | PlaceRoadCommand
  | PlaceBuildingCommand
  | AssignPriestCommand
  | SetGameSpeedCommand
  | AdvanceTickCommand
  | StartConsultationCommand
  | PlaceProphecyTileCommand
  | RemoveProphecyTileCommand
  | DeliverProphecyCommand
  | SaveGameCommand
  | LoadGameCommand
  | PurchaseTradeOfferCommand
  | RestPythiaCommand
  | PurifyPythiaCommand
  | RepairBuildingCommand
  | SetToolCommand
  | SelectEntityCommand
  | HoverTileCommand
  | InjectScenarioCommand
  | StartNewRunCommand
  | IssuePriestDecreeCommand
  | DismissPriestCommand
  | EndorseBlocCommand
  | BeginExcavationCommand
  | ClaimRelicCommand
  | ActivateSacredSiteCommand
  | LaunchEspionageOperationCommand
  | InvestigatePriestCommand
  | RecruitAgentCommand
  | TriggerEndOfRunCommand
  | BeginLegendaryConsultationCommand
  | AdvanceLegendaryStageCommand
  | StartNewLineageRunCommand
  | RecordLineageRunCommand;
