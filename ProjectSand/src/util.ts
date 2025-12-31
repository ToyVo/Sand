/*
 * Miscellaneous utility functions and constants.
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

/* Pre-compute to improve performance */
export const TWO_PI: number = 2 * Math.PI;
export const HALF_PI: number = Math.PI / 2;
export const QUARTER_PI: number = Math.PI / 4;
export const EIGHTH_PI: number = Math.PI / 8;
export const SIXTEENTH_PI: number = Math.PI / 16;
export const EIGHTEENTH_PI: number = Math.PI / 18;

const __num_rand_ints: number = 8192;
const __rand_ints: Uint8Array = new Uint8Array(__num_rand_ints);
let __next_rand: number = 0;
for (let i = 0; i < __num_rand_ints; i++) {
  __rand_ints[i] = Math.floor(Math.random() * 100);
}

/*
 * Returns a pre-generated random byte between 0-99.
 * This is especially important for hot-paths that
 * can't tolerate the time to call Math.random() directly
 * (or deal with floats).
 */
export function random(): number {
  const r = __rand_ints[__next_rand];

  __next_rand++;
  if (__next_rand === __num_rand_ints) __next_rand = 0;

  return r;
}

/* Returns a random int in range [low, high) */
export function randomIntInRange(low: number, high: number): number {
  return Math.floor(Math.random() * (high - low) + low);
}

export function clamp(val: number, min: number, max: number): number {
  return Math.max(min, Math.min(val, max));
}

export function executeAndTime(func: () => void): number {
  const start = performance.now();
  func();
  const end = performance.now();

  return end - start;
}

export function displayPerformance(func: () => void, funcName: string): void {
  const execTime = executeAndTime(func);

  console.log(funcName, ": ", execTime, "ms");
}

export function docOffsetLeft(elem: HTMLElement): number {
  let offsetLeft = 0;
  let current: HTMLElement | null = elem;
  do {
    if (current && !isNaN(current.offsetLeft)) {
      offsetLeft += current.offsetLeft;
    }
    current = current?.offsetParent as HTMLElement | null;
  } while (current);
  return offsetLeft;
}

export function docOffsetTop(elem: HTMLElement): number {
  let offsetTop = 0;
  let current: HTMLElement | null = elem;
  do {
    if (current && !isNaN(current.offsetTop)) {
      offsetTop += current.offsetTop;
    }
    current = current?.offsetParent as HTMLElement | null;
  } while (current);
  return offsetTop;
}

export function distance(x1: number, y1: number, x2: number, y2: number): number {
  const dx = x1 - x2;
  const dy = y1 - y2;

  return Math.sqrt(dx * dx + dy * dy);
}

/*
 * We could convert i to xy using division and modulus, but
 * this can be slow. In cases where we want to convert a coordinate
 * 'i' that is known to border another coordinate with known xy,
 * we can determine the xy of the coordinate by iterating all
 * bordering pixels.
 */
export function fastItoXYBorderingAdjacent(
  startX: number,
  startY: number,
  startI: number,
  goalI: number,
  width: number
): [number, number] {
  const bottom = startI + width;
  if (bottom === goalI) return [startX, startY + 1];
  else if (bottom - 1 === goalI) return [startX - 1, startY + 1];
  else if (bottom + 1 === goalI) return [startX + 1, startY + 1];

  if (startI - 1 === goalI) return [startX - 1, startY];
  else if (startI + 1 === goalI) return [startX + 1, startY];

  const top = startI - width;
  if (top === goalI) return [startX, startY - 1];
  else if (top - 1 === goalI) return [startX - 1, startY - 1];
  else if (top + 1 === goalI) return [startX + 1, startY - 1];

  throw new Error("Not passed a bordering coordinate");
}

/*
 * See comment on fastItoXYBorderingAdjacent.
 * This function does the same, but ignores corners.
 */
export function fastItoXYBordering(
  startX: number,
  startY: number,
  startI: number,
  goalI: number,
  width: number
): [number, number] {
  if (startI + width === goalI) return [startX, startY + 1];

  if (startI - 1 === goalI) return [startX - 1, startY];
  else if (startI + 1 === goalI) return [startX + 1, startY];

  if (startI - width === goalI) return [startX, startY - 1];

  throw new Error("Not passed a bordering coordinate");
}

