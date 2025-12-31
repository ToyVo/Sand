/*
 * Drives the primary game loops.
 *
 * Copyright (C) 2020, Josh Don
 *
 * Project Sand is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Project Sand is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

// Import specific exports we need
import {
  width,
  height,
  DEFAULT_FPS,
  ZOMBIE_ANIMATION_SPEED,
} from "./canvasConfig.js";
import {
  BACKGROUND,
  elementActions,
  setGameVars,
  initElements,
} from "./elements.js";
import { executeAndTime } from "./util.js";
import { initCursors, updateUserStroke } from "./cursor.js";
import { particles, initParticles, updateParticles } from "./particles.js";
import { initSpigots, updateSpigots, setSpigotGameVars } from "./spigots.js";
import { initMenu, drawFPSLabel } from "./menu.js";
import {
  initSoftBody,
  softBodyAnimate,
  softBodyRender,
  softBodyDragStart,
} from "./softBody.js";

/* ================================ Globals ================================ */

/* Scaling due to device pixel ratio */
const onscreenPixelRatio: number = window.devicePixelRatio;
const onscreenScaledWidth: number = onscreenPixelRatio * width;
const onscreenScaledHeight: number = onscreenPixelRatio * height;

/* Onscreen canvas. Scaled based on pixel ratio. */
export const onscreenCanvas = document.getElementById(
  "mainCanvas"
) as HTMLCanvasElement;
if (!onscreenCanvas) throw new Error("mainCanvas not found");
onscreenCanvas.width = onscreenScaledWidth;
onscreenCanvas.height = onscreenScaledHeight;
onscreenCanvas.style.width = width + "px";
onscreenCanvas.style.height = height + "px";
const onscreenCtx = onscreenCanvas.getContext("2d", {
  alpha: false,
}) as CanvasRenderingContext2D;

/*
 * Offscreen game canvas. Drawn at in-game resolution, then
 * scaled and transferred to the onscreen canvas.
 */
const gameCanvas = document.createElement("canvas");
gameCanvas.width = width;
gameCanvas.height = height;
const gameCtx = gameCanvas.getContext("2d") as CanvasRenderingContext2D;
const gameImagedata = gameCtx.createImageData(width, height);
export const gameImagedata32 = new Uint32Array(gameImagedata.data.buffer);

/* Storage for game save state. */
const saveGameImagedata32 = new Uint32Array(gameImagedata32.length);
let gamestateSaved: boolean = false;

/* Cached for performance */
export const MAX_X_IDX: number = width - 1;
export const MAX_Y_IDX: number = height - 1;
export const MAX_IDX: number = width * height - 1;

/* Globals for tracking and maintaining FPS */
let fpsSetting: number; /* controlled via menu */
let msPerFrame: number;
let lastLoop: number = 0;
let frameDebt: number = 0;
let lastFPSLabelUpdate: number = 0;
const refreshTimes: number[] = [];

/* ========================================================================= */

export function init(): void {
  // Set game vars in elements.js and spigots.js to avoid circular dependency
  setGameVars({ gameImagedata32, MAX_X_IDX, MAX_Y_IDX, VOID_MODE_ENABLED: false });
  setSpigotGameVars({ gameImagedata32, MAX_X_IDX });

  const gameWrapper = document.getElementById("gameWrapper");
  if (gameWrapper) {
    gameWrapper.style.height = height + "px";
    gameWrapper.style.width = width + "px";
  }

  /* setting FPS must occur before initMenu() */
  setFPS(DEFAULT_FPS);

  initCursors();
  initElements();
  initParticles();
  initSpigots();
  initMenu();
  initSoftBody();

  /* Initialize imagedata */
  const len = gameImagedata32.length;
  for (let i = 0; i < len; i++) {
    gameImagedata32[i] = BACKGROUND;
    saveGameImagedata32[i] = BACKGROUND;
  }

  /* Nice crisp pixels, regardless of pixel ratio */
  (onscreenCtx as any).mozImageSmoothingEnabled = false;
  onscreenCtx.imageSmoothingEnabled = false;
  (onscreenCtx as any).webkitImageSmoothingEnabled = false;
  (onscreenCtx as any).msImageSmoothingEnabled = false;
  (onscreenCtx as any).oImageSmoothingEnabled = false;
}

export function setFPS(fps: number): void {
  fpsSetting = fps;
  if (fps > 0) msPerFrame = 1000.0 / fpsSetting;
  else drawFPSLabel(0);
}
function updateGame(): void {
  updateSpigots();
  updateParticles();

  let x: number, y: number;
  let i = MAX_IDX;
  /*
   * Since i starts at MAX_IDX, we need to guarantee that we will start
   * our traversal by going to the left.
   */
  const direction = MAX_Y_IDX & 1;

  /*
   * Iterate the canvas from the bottom to top, zigzagging
   * the rows left and right.
   * To optimize for speed, we duplicate the code for the
   * left->right and right->left cases, as this is our hottest
   * inner path. This sacrifices readability, and violates DRY,
   * but is necessary for game performance.
   */
  for (y = MAX_Y_IDX; y !== -1; y--) {
    const Y = y;
    if ((Y & 1) === direction) {
      for (x = MAX_X_IDX; x !== -1; x--) {
        const elem = gameImagedata32[i];
        if (elem === BACKGROUND) {
          i--;
          continue; /* optimize to skip background */
        }
        const elem_idx =
          ((elem & 0x30000) >>> 12) + ((elem & 0x300) >>> 6) + (elem & 0x3);
        elementActions[elem_idx](x, Y, i);
        i--;
      }
      i++;
    } else {
      for (x = 0; x !== width; x++) {
        const elem = gameImagedata32[i];
        if (elem === BACKGROUND) {
          i++;
          continue;
        }
        const elem_idx =
          ((elem & 0x30000) >>> 12) + ((elem & 0x300) >>> 6) + (elem & 0x3);
        elementActions[elem_idx](x, Y, i);
        i++;
      }
      i--;
    }
    i -= width;
  }

  perfRecordFrame();
  frameDebt--;
}

function draw(): void {
  gameCtx.putImageData(gameImagedata, 0, 0);

  /*
   * To make sure our game looks crisp, we need to handle
   * device pixel ratio. We do this by taking our offscreen
   * game canvas (at our ingame resolution), and then scaling
   * and transferring it to the displayed canvas.
   */
  gameCtx.scale(onscreenPixelRatio, onscreenPixelRatio);
  onscreenCtx.drawImage(
    gameCanvas,
    0,
    0,
    onscreenScaledWidth,
    onscreenScaledHeight
  );
}

function setGameCanvas(elem: number): void {
  const iterEnd = MAX_IDX + 1;
  for (let i = 0; i !== iterEnd; i++) {
    gameImagedata32[i] = elem;
  }
}

export function clearGameCanvas(): void {
  particles.inactivateAll();
  setGameCanvas(BACKGROUND);
}

/*
 * Saves the current canvas state. Note that we don't also save particle state.
 */
export function saveGameCanvas(): void {
  /*
   * Copy it manually, rather than use a slice, so that we can use a constant
   * global pointer.
   */
  const iterEnd = MAX_IDX + 1;
  for (let i = 0; i !== iterEnd; i++)
    saveGameImagedata32[i] = gameImagedata32[i];

  gamestateSaved = true;
}

export function loadGameCanvas(): void {
  if (!gamestateSaved) return;

  particles.inactivateAll();

  const iterEnd = MAX_IDX + 1;
  for (let i = 0; i !== iterEnd; i++)
    gameImagedata32[i] = saveGameImagedata32[i];
}

/* Signal that we've updated a game frame to our FPS counter */
function perfRecordFrame(): void {
  const now = performance.now();
  const oneSecondAgo = now - 1000;
  while (refreshTimes.length > 0 && refreshTimes[0] <= oneSecondAgo) {
    refreshTimes.shift();
  }
  refreshTimes.push(now);

  if (now - lastFPSLabelUpdate > 200) {
    drawFPSLabel(refreshTimes.length);
    lastFPSLabelUpdate = now;
  }
}

function mainLoop(now: number): void {
  window.requestAnimationFrame(mainLoop);

  /* Handle initial update */
  if (lastLoop === 0) {
    lastLoop = now;
    return;
  }

  const deltaMs = now - lastLoop;
  lastLoop = now;
  if (deltaMs < 0) {
    console.log("time has gone backwards");
    return;
  }

  if (fpsSetting > 0) frameDebt += deltaMs / msPerFrame;

  /*
   * Avoid accumulating too much frame debt, which can
   * occur, for example, from:
   * - animation loop being paused due to loss of browser
   *   tab focus
   * - excessive time needed for updateGame() due to
   *   complex update
   *
   * Naturally, this also limits our max theoretical FPS, but
   * our MAX_FPS is set lower than this limit anyway.
   */
  frameDebt = Math.min(frameDebt, 5);

  /*
   * Always update the user stroke, regardless of whether
   * we're updating the gamestate. This results in smooth
   * drawing regardless of the current set FPS.
   *
   * Stop drawing the stroke if we're dragging a soft body,
   * since we don't want both at once.
   */
  if (!softBodyDragStart) {
    updateUserStroke();
  }

  let framesUpdated = 0;
  if (frameDebt >= 1) {
    if (frameDebt == 1) {
      /* shortcut for the common case of a single-frame update */
      updateGame();
      framesUpdated++;
    } else {
      /* multi-frame update */

      /* first get approx time for a single update */
      const updateTimeMs = executeAndTime(updateGame);
      framesUpdated++;

      /*
       * Approx time for doing stroke, draw, etc.
       * This is very rough and could be improved.
       */
      const loopMiscTimeMs = 3.5;
      let timeRemaining = deltaMs - loopMiscTimeMs - updateTimeMs;
      while (timeRemaining > updateTimeMs && frameDebt >= 1) {
        updateGame();
        timeRemaining -= updateTimeMs;
        framesUpdated++;
      }
    }
  }

  if (framesUpdated) {
    softBodyAnimate(framesUpdated * ZOMBIE_ANIMATION_SPEED);
    softBodyRender();
  }

  draw();
}

window.onload = function (): void {
  init();
  mainLoop(0);
};

