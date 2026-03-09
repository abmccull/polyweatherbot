import Phaser from "phaser";

import { captureModeEnabled } from "../runtime";
import { PrecinctScene } from "./PrecinctScene";

export function createGame(parent: HTMLElement): Phaser.Game {
  const game = new Phaser.Game({
    type: captureModeEnabled ? Phaser.CANVAS : Phaser.AUTO,
    parent,
    width: window.innerWidth,
    height: window.innerHeight,
    backgroundColor: "#d9c79c",
    scene: [PrecinctScene],
    render: captureModeEnabled
      ? {
          antialias: true,
          preserveDrawingBuffer: true,
          transparent: false
        }
      : undefined,
    scale: {
      mode: Phaser.Scale.RESIZE,
      autoCenter: Phaser.Scale.CENTER_BOTH
    }
  });

  game.events.once(Phaser.Core.Events.READY, () => {
    game.canvas.id = "precinct-canvas";
    game.canvas.setAttribute("data-testid", "precinct-canvas");
    if (captureModeEnabled) {
      game.canvas.setAttribute("data-capture-mode", "1");
      window.__oracleDebug.captureFrame = async () => game.canvas.toDataURL("image/png");
    }
  });

  return game;
}
