/*
 * Handling for the four primary game spigots.
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

import {
  SAND,
  WATER,
  SALT,
  OIL,
  GUNPOWDER,
  NITRO,
  NAPALM,
  CONCRETE,
  LAVA,
  CRYO,
  ACID,
  MYSTERY,
} from "./elements.js";
import { width, height } from "./canvasConfig.js";
import { random } from "./util.js";

// These will be set by game.js after initialization to avoid circular dependency
let gameImagedata32: Uint32Array;
let MAX_X_IDX: number;
export function setSpigotGameVars(vars: {
  gameImagedata32: Uint32Array;
  MAX_X_IDX: number;
}): void {
  gameImagedata32 = vars.gameImagedata32;
  MAX_X_IDX = vars.MAX_X_IDX;
}

/* Menu options for the spigots */
export const SPIGOT_ELEMENT_OPTIONS: number[] = [
  SAND,
  WATER,
  SALT,
  OIL,
  GUNPOWDER,
  NITRO,
  NAPALM,
  CONCRETE,
  LAVA,
  CRYO,
  ACID,
  MYSTERY,
];
export const SPIGOT_SIZE_OPTIONS: number[] = [0, 5, 10, 15, 20, 25];
export const DEFAULT_SPIGOT_SIZE_IDX: number = 1;

/* Type and size of each spigot. Controlled via the menu. */
export const SPIGOT_ELEMENTS: number[] = [SAND, WATER, SALT, OIL];
export const SPIGOT_SIZES: number[] = [];

export const SPIGOT_HEIGHT: number = 10;
export const MAX_SPIGOT_WIDTH: number = Math.max(...SPIGOT_SIZE_OPTIONS);
export const NUM_SPIGOTS: number = SPIGOT_ELEMENTS.length;
export const SPIGOT_SPACING: number = Math.round(
  (width - MAX_SPIGOT_WIDTH * NUM_SPIGOTS) / (NUM_SPIGOTS + 1) +
    MAX_SPIGOT_WIDTH
);
export const SPIGOTS_ENABLED: boolean =
  MAX_SPIGOT_WIDTH * NUM_SPIGOTS <= width && SPIGOT_HEIGHT <= height;

export function initSpigots(): void {
  const defaultSize = SPIGOT_SIZE_OPTIONS[DEFAULT_SPIGOT_SIZE_IDX];
  for (let i = 0; i !== NUM_SPIGOTS; i++) {
    SPIGOT_SIZES.push(defaultSize);
  }
}

export function updateSpigots(): void {
  if (!SPIGOTS_ENABLED) return;

  let i: number, w: number, h: number;
  for (i = 0; i !== NUM_SPIGOTS; i++) {
    const elem = SPIGOT_ELEMENTS[i];
    const spigotLeft = SPIGOT_SPACING * (i + 1) - MAX_SPIGOT_WIDTH;
    const spigotRight = spigotLeft + SPIGOT_SIZES[i];
    if (spigotLeft < 0) continue;
    if (spigotRight > MAX_X_IDX) break;
    let heightOffset = 0;
    for (h = 0; h !== SPIGOT_HEIGHT; h++) {
      for (w = spigotLeft; w !== spigotRight; w++) {
        if (random() < 10) gameImagedata32[w + heightOffset] = elem;
      }
      heightOffset += width;
    }
  }
}

