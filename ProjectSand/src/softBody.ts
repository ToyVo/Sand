/*
 * Interaction with matter.js.
 *
 * Copyright (C) 2025, Josh Don
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

/*
 * This encapsulates all soft body animations (which are driven by matter.js),
 * such as zombies.js.
 *
 * To learn how matter.js works, I recommend reading the source code:
 * https://github.com/liabru/matter-js
 */

import Matter from "matter-js";
import { width, height } from "./canvasConfig.js";
import { zombies } from "./zombies.js";
import {
  ZOMBIE_STATE_NORMAL,
  ZOMBIE_STATE_WET,
  ZOMBIE_STATE_BURNING,
  ZOMBIE_STATE_FROZEN,
  drawZombies,
} from "./zombies.js";
import {
  ZOMBIE,
  ZOMBIE_WET,
  ZOMBIE_BURNING,
  ZOMBIE_FROZEN,
} from "./elements.js";

export const softBodyEngine = Matter.Engine.create(); /* The global matter.js engine */

/*
 * Offscreen canvas for drawing soft body animations. We draw on
 * this canvas and then transfer the data to the main
 * canvas.
 */
const softBodyCanvas = document.createElement("canvas");
softBodyCanvas.width = width;
softBodyCanvas.height = height;
export const softBodyCtx = softBodyCanvas.getContext("2d", {
  alpha: false,
  willReadFrequently: true,
}) as CanvasRenderingContext2D;

/*
 * Globals for handling mouse interaction (ie. dragging the
 * soft bodies).
 */
export let softBodyDragStart: number = 0; /* timestamp in milliseconds */
let softBodyFreeDrag: boolean = false; /* whether the dragged body should ignore collisions with canvas elements */
export function getSoftBodyFreeDrag(): boolean {
  return softBodyFreeDrag;
}
export function setSoftBodyFreeDrag(value: boolean): void {
  softBodyFreeDrag = value;
}
export let softBodyMouse: any; // Matter.Mouse type
export let softBodyMouseConstraint: any; // Matter.MouseConstraint type

/* ======================= game.js API ====================== */

/* Initialize all soft body elements */
export function initSoftBody(): void {
  softBodyEngine.gravity.scale = 0.0002; /* library default is 0.001 */

  /*
   * These are the 4 walls that bound the animation to the canvas frame
   */
  const wallDepth: number = 60; /* a fairly arbitrary value, to prevent clipping */
  const categoryIgnoreMouse = Matter.Body.nextCategory();
  const options = {
    isStatic: true,
    collisionFilter: { category: categoryIgnoreMouse },
  };
  const topWall = Matter.Bodies.rectangle(
    width / 2,
    -wallDepth / 2,
    width * 1.2,
    wallDepth,
    options
  );
  const bottomWall = Matter.Bodies.rectangle(
    width / 2,
    height + wallDepth / 2,
    width * 1.2,
    wallDepth,
    options
  );
  const leftWall = Matter.Bodies.rectangle(
    -wallDepth / 2,
    height / 2,
    wallDepth,
    height * 1.2,
    options
  );
  const rightWall = Matter.Bodies.rectangle(
    width + wallDepth / 2,
    height / 2,
    wallDepth,
    height * 1.2,
    options
  );

  Matter.Composite.add(softBodyEngine.world, [
    topWall,
    bottomWall,
    leftWall,
    rightWall,
  ]);

  /*
   * Add mouse handler to allow user to drag soft bodies around.
   */
  const mainCanvas = document.getElementById("mainCanvas") as HTMLCanvasElement;
  if (!mainCanvas) throw new Error("mainCanvas not found");
  softBodyMouse = Matter.Mouse.create(mainCanvas);
  softBodyMouseConstraint = Matter.MouseConstraint.create(softBodyEngine, {
    mouse: softBodyMouse,
    constraint: {
      stiffness: 0.2,
      render: {
        visible: false,
      },
    },
    /* We need to prevent the mouse from interacting with these invisible boundary walls */
    collisionFilter: {
      mask: ~categoryIgnoreMouse,
    },
  });
  /*
   * We're attaching the mouse to the main canvas, which is not the same
   * size as the internal game canvas. They differ in scaling by the pixel
   * ratio.
   */
  Matter.Mouse.setScale(softBodyMouse, {
    x: 1.0 / window.devicePixelRatio,
    y: 1.0 / window.devicePixelRatio,
  });
  Matter.Composite.add(softBodyEngine.world, [softBodyMouseConstraint]);

  Matter.Events.on(softBodyMouseConstraint, "startdrag", (_event: any) => {
    softBodyDragStart = Date.now();
    setSoftBodyFreeDrag(false);
  });
  /*
   * Note: this fires for all mouse up events, even when we weren't
   * previously dragging.
   */
  Matter.Events.on(softBodyMouseConstraint, "mouseup", (_event: any) => {
    softBodyDragStart = 0;
  });
}

/* Advance the animation of all soft bodies by the given amount of milliseconds */
export function softBodyAnimate(milliseconds: number): void {
  const now = Date.now();
  const numZombies = zombies.length;
  for (let i = 0; i < numZombies; i++) {
    zombies[i].animate(now, i, milliseconds);
  }

  Matter.Engine.update(softBodyEngine, milliseconds);
}

/* Render all soft bodies onto the main canvas */
export function softBodyRender(): void {
  const normalZombies: any[] = [];
  const wetZombies: any[] = [];
  const burningZombies: any[] = [];
  const frozenZombies: any[] = [];

  const numZombies = zombies.length;
  for (let i = 0; i < numZombies; i++) {
    const zombie = zombies[i];
    const state = zombie.state;
    if (state === ZOMBIE_STATE_NORMAL) {
      normalZombies.push(zombie);
    } else if (state === ZOMBIE_STATE_WET) {
      wetZombies.push(zombie);
    } else if (state === ZOMBIE_STATE_BURNING) {
      burningZombies.push(zombie);
    } else if (state === ZOMBIE_STATE_FROZEN) {
      frozenZombies.push(zombie);
    } else {
      throw new Error("unexpected state");
    }
  }

  if (normalZombies.length) {
    drawZombies(normalZombies, ZOMBIE);
  }
  if (wetZombies.length) {
    drawZombies(wetZombies, ZOMBIE_WET);
  }
  if (burningZombies.length) {
    drawZombies(burningZombies, ZOMBIE_BURNING);
  }
  if (frozenZombies.length) {
    drawZombies(frozenZombies, ZOMBIE_FROZEN);
  }
}

/* ======================= Soft Body APIs ====================== */

/* Draw the given matter.js body onto the provided `ctx`. */
export function drawBody(ctx: CanvasRenderingContext2D, body: any): void {
  const vertices = body.vertices;

  ctx.moveTo(vertices[0].x, vertices[0].y);

  for (let j = 1; j < vertices.length; j += 1) {
    ctx.lineTo(vertices[j].x, vertices[j].y);
  }

  ctx.lineTo(vertices[0].x, vertices[0].y);
}

