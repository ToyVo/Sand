/*
 * Handling for particle animations. Unlike elements, which are
 * represented by colored pixels on the canvas, particles are
 * backed by an actual object. This means that we can store state
 * for each particle, in order to do things such as complex movement.
 * However, this also means that particles are more expensive than
 * regular elements, so we use them sparingly (as limited by
 * MAX_NUM_PARTICLES).
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

/*
 * ADDING NEW PARTICLES:
 *
 * 1. Define its particle index below. Pick the next available integer.
 * 2. Add its init and action methods to the __particleInit and
 *    __particleActions arrays. Order here must match the particle indices.
 * 3. Implement the init and action methods for your particle. Importantly,
 *    your action method is responsible for inactivating your particle (ie.
 *    after a certain number of iterations, or when it is offCanvas()). Use
 *    particles.makeParticleInactive(particle) for this.
 * 4. Your init method sets your particle color; it must pick a
 *    PAINTABLE_PARTICLE_COLOR.
 */

import { MAX_NUM_PARTICLES, width, height } from "./canvasConfig.js";
import {
  TWO_PI,
  HALF_PI,
  QUARTER_PI,
  EIGHTH_PI,
  SIXTEENTH_PI,
  EIGHTEENTH_PI,
  random,
} from "./util.js";
import {
  WALL,
  FIRE,
  ROCK,
  WATER,
  SALT_WATER,
  ICE,
  CHILLED_ICE,
  CRYO,
  LAVA,
  PLANT,
  SPOUT,
  WELL,
  WAX,
  BRANCH,
  LEAF,
  BACKGROUND,
} from "./elements.js";
import { gameImagedata32, MAX_X_IDX, MAX_Y_IDX, MAX_IDX } from "./game.js";

/* Offscreen canvas for drawing particles */
const offscreenParticleCanvas = document.createElement("canvas");
const offscreenParticleCtx = offscreenParticleCanvas.getContext("2d", {
  alpha: false,
}) as CanvasRenderingContext2D;

/* These values index into __particleInit and __particleActions arrays */
export const UNKNOWN_PARTICLE: number = 0;
export const NITRO_PARTICLE: number = 1;
export const NAPALM_PARTICLE: number = 2;
export const C4_PARTICLE: number = 3;
export const LAVA_PARTICLE: number = 4;
export const MAGIC1_PARTICLE: number = 5; /* multi-pronged star */
export const MAGIC2_PARTICLE: number = 6; /* spiral */
export const METHANE_PARTICLE: number = 7;
export const TREE_PARTICLE: number = 8;
export const CHARGED_NITRO_PARTICLE: number = 9;
export const NUKE_PARTICLE: number = 10;

type ParticleInitFunction = (particle: Particle) => void;
type ParticleActionFunction = (particle: Particle) => void;

const __particleInit: ParticleInitFunction[] = [
  UNKNOWN_PARTICLE_INIT,
  NITRO_PARTICLE_INIT,
  NAPALM_PARTICLE_INIT,
  C4_PARTICLE_INIT,
  LAVA_PARTICLE_INIT,
  MAGIC1_PARTICLE_INIT,
  MAGIC2_PARTICLE_INIT,
  METHANE_PARTICLE_INIT,
  TREE_PARTICLE_INIT,
  CHARGED_NITRO_PARTICLE_INIT,
  NUKE_PARTICLE_INIT,
];
Object.freeze(__particleInit);

const __particleActions: ParticleActionFunction[] = [
  UNKNOWN_PARTICLE_ACTION,
  NITRO_PARTICLE_ACTION,
  NAPALM_PARTICLE_ACTION,
  C4_PARTICLE_ACTION,
  LAVA_PARTICLE_ACTION,
  MAGIC1_PARTICLE_ACTION,
  MAGIC2_PARTICLE_ACTION,
  METHANE_PARTICLE_ACTION,
  TREE_PARTICLE_ACTION,
  CHARGED_NITRO_PARTICLE_ACTION,
  NUKE_PARTICLE_ACTION,
];
Object.freeze(__particleActions);

function UNKNOWN_PARTICLE_INIT(_particle: Particle): void {}
function UNKNOWN_PARTICLE_ACTION(_particle: Particle): void {
  throw new Error("Unknown particle");
}

function NITRO_PARTICLE_INIT(particle: Particle): void {
  particle.setColor(FIRE);

  const velocity = 5 + Math.random() * 10;
  const angle = Math.random() * TWO_PI;
  particle.setVelocity(velocity, angle);

  particle.size = 2 + Math.random() * 7;
}

function NITRO_PARTICLE_ACTION(particle: Particle): void {
  offscreenParticleCtx.beginPath();
  offscreenParticleCtx.lineWidth = particle.size;
  offscreenParticleCtx.strokeStyle = particle.rgbaColor;
  offscreenParticleCtx.lineCap = "round";
  offscreenParticleCtx.moveTo(particle.x, particle.y);
  particle.x += particle.xVelocity;
  particle.y += particle.yVelocity;
  offscreenParticleCtx.lineTo(particle.x, particle.y);
  offscreenParticleCtx.stroke();

  const iterations = particle.actionIterations;
  if (iterations % 5 === 0) particle.size /= 1.3;
  if (iterations % 15 === 0) particle.yVelocity += 10 * (iterations / 5);

  if (particle.size < 1.75) particles.makeParticleInactive(particle);
  else if (particle.offCanvas()) particles.makeParticleInactive(particle);
}

function NAPALM_PARTICLE_INIT(particle: Particle): void {
  particle.setColor(FIRE);
  particle.size = Math.random() * 8 + 6;
  particle.xVelocity = Math.random() * 8 - 4;
  particle.yVelocity = -1 * (Math.random() * 4 + 4);
  (particle as any).maxIterations = Math.floor(Math.random() * 10) + 5;
}

function NAPALM_PARTICLE_ACTION(particle: Particle): void {
  particle.drawCircle(particle.size);

  particle.x += particle.xVelocity;
  particle.y += particle.yVelocity;
  particle.size *= 1 + Math.random() * 0.1;

  if (particle.actionIterations > (particle as any).maxIterations)
    particles.makeParticleInactive(particle);
}

function C4_PARTICLE_INIT(particle: Particle): void {
  particle.setColor(FIRE);
  const rand = Math.random() * 10000;
  if (rand < 9000) {
    particle.size = Math.random() * 10 + 3;
  } else if (rand < 9500) {
    particle.size = Math.random() * 32 + 3;
  } else if (rand < 9800) {
    particle.size = Math.random() * 64 + 3;
  } else {
    particle.size = Math.random() * 128 + 3;
  }
}

function C4_PARTICLE_ACTION(particle: Particle): void {
  particle.drawCircle(particle.size);

  if (particle.actionIterations % 3 === 0) {
    particle.size /= 3;
    if (particle.size <= 1) particles.makeParticleInactive(particle);
  }
}

function LAVA_PARTICLE_INIT(particle: Particle): void {
  particle.setColor(FIRE);
  /* Make it harder for the angle to be steep */
  let angle = QUARTER_PI + Math.random() * HALF_PI;
  if (random() < 75 && Math.abs(HALF_PI - angle) < EIGHTEENTH_PI)
    angle += EIGHTEENTH_PI * (angle > HALF_PI ? 1 : -1);

  particle.xVelocity = (1 + Math.random() * 3) * Math.cos(angle);
  particle.yVelocity = (-4 * Math.random() - 3) * Math.sin(angle);
  (particle as any).initYVelocity = particle.yVelocity;
  (particle as any).yAcceleration = 0.06;

  particle.size = 4 + Math.random() * 3;
  particle.y -= particle.size;
}

function LAVA_PARTICLE_ACTION(particle: Particle): void {
  offscreenParticleCtx.beginPath();
  offscreenParticleCtx.lineWidth = particle.size;
  offscreenParticleCtx.strokeStyle = particle.rgbaColor;
  offscreenParticleCtx.lineCap = "round";
  offscreenParticleCtx.moveTo(particle.x, particle.y);

  const iterations = particle.actionIterations;
  particle.x += particle.xVelocity;
  particle.y =
    particle.initY +
    (particle as any).initYVelocity * iterations +
    ((particle as any).yAcceleration * iterations * iterations) / 2;

  offscreenParticleCtx.lineTo(particle.x, particle.y);
  offscreenParticleCtx.stroke();

  /* Allow particle to exist "above" the canvas */
  if (particle.x < 0 || particle.x > MAX_X_IDX || particle.y > MAX_Y_IDX) {
    particles.makeParticleInactive(particle);
    return;
  }

  /* possible extinguish due to water, lava, rock, ice, or wall */
  if (random() < 25) {
    /* Need to update yVelocity before calling aboutToHit() */
    particle.yVelocity =
      (particle as any).initYVelocity +
      (particle as any).yAcceleration * iterations;
    const touchingColor = particle.aboutToHit();

    let replaceColor = -1;
    if (touchingColor === WATER || touchingColor === SALT_WATER) {
      if (random() < 58) replaceColor = ROCK;
    } else if (touchingColor === LAVA || touchingColor === ROCK) {
      if (random() < 75) replaceColor = LAVA;
    } else if (
      touchingColor === ICE ||
      touchingColor === CHILLED_ICE ||
      touchingColor === CRYO
    ) {
      if (random() < 70) replaceColor = ROCK;
    } else if (touchingColor === WALL) {
      if (random() < 25) replaceColor = LAVA;
    }

    if (replaceColor !== -1) {
      particle.setColor(replaceColor);
      particle.drawCircle(particle.size / 2);
      particles.makeParticleInactive(particle);
      return;
    }
  }
}

function MAGIC1_PARTICLE_INIT(particle: Particle): void {
  if (!particle.reinitialized) particle.setRandomColor(MAGIC_COLORS);

  let numSpokes = 5 + Math.round(Math.random() * 13);
  const spokes: Particle[] = [particle];
  let i: number;
  for (i = 1; i !== numSpokes; i++) {
    /*
     * Temporarily set type to UNKNOWN_PARTICLE so that we don't
     * recurse back into this function.
     */
    const newSpoke = particles.addActiveParticle(
      UNKNOWN_PARTICLE,
      particle.x,
      particle.y,
      particle.i
    );
    if (!newSpoke) break;

    /*
     * We're manually changing the particle type; ensure that our
     * particle counts don't get corrupted.
     */
    particles.particleCounts[newSpoke.type]--;
    particles.particleCounts[MAGIC1_PARTICLE]++;

    newSpoke.type = MAGIC1_PARTICLE;
    newSpoke.setColor(particle.color);
    spokes.push(newSpoke);
  }
  numSpokes = spokes.length;

  const angle = TWO_PI / numSpokes;
  const velocity = 7 + Math.random() * 3;
  const spokeSize = 4 + Math.random() * 4;

  let currAngle = 0;
  for (i = 0; i !== numSpokes; i++) {
    const spoke = spokes[i];
    spoke.setVelocity(velocity, currAngle);
    spoke.size = spokeSize;
    currAngle += angle;
  }
}

function MAGIC1_PARTICLE_ACTION(particle: Particle): void {
  offscreenParticleCtx.beginPath();
  offscreenParticleCtx.lineWidth = particle.size;
  offscreenParticleCtx.strokeStyle = particle.rgbaColor;
  offscreenParticleCtx.lineCap = "square";
  offscreenParticleCtx.moveTo(particle.x, particle.y);
  particle.x += particle.xVelocity;
  particle.y += particle.yVelocity;
  offscreenParticleCtx.lineTo(particle.x, particle.y);
  offscreenParticleCtx.stroke();

  if (particle.offCanvas()) particles.makeParticleInactive(particle);
}

function MAGIC2_PARTICLE_INIT(particle: Particle): void {
  particle.setRandomColor(MAGIC_COLORS);

  particle.size = 4 + Math.random() * 8;
  particle.x = Math.floor(width / 2);
  particle.y = Math.floor(height / 2);
  particle.initX = particle.x;
  particle.initY = particle.y;
  (particle as any).magic_2_max_radius =
    Math.sqrt(width * width + height * height) / 2 + particle.size;
  (particle as any).magic_2_theta = 0;
  (particle as any).magic_2_speed = 20;
  (particle as any).magic_2_radius_spacing = 25 + Math.random() * 55;
  (particle as any).magic_2_radius = (particle as any).magic_2_radius_spacing;
}

function MAGIC2_PARTICLE_ACTION(particle: Particle): void {
  offscreenParticleCtx.beginPath();
  offscreenParticleCtx.lineWidth = particle.size;
  offscreenParticleCtx.strokeStyle = particle.rgbaColor;
  offscreenParticleCtx.lineCap = "round";
  offscreenParticleCtx.moveTo(particle.x, particle.y);

  const newTheta =
    (particle as any).magic_2_theta +
    (particle as any).magic_2_speed / (particle as any).magic_2_radius;
  (particle as any).magic_2_theta = newTheta;
  const newRadius =
    ((particle as any).magic_2_theta / TWO_PI) *
    (particle as any).magic_2_radius_spacing;
  (particle as any).magic_2_radius = newRadius;

  particle.x =
    newRadius * Math.cos(newTheta) + particle.initX;
  particle.y =
    newRadius * Math.sin(newTheta) + particle.initY;

  offscreenParticleCtx.lineTo(particle.x, particle.y);
  offscreenParticleCtx.stroke();

  if (newRadius > (particle as any).magic_2_max_radius)
    particles.makeParticleInactive(particle);
}

function METHANE_PARTICLE_INIT(particle: Particle): void {
  particle.setColor(FIRE);
  particle.size = 10 + Math.random() * 10;
}

function METHANE_PARTICLE_ACTION(particle: Particle): void {
  const iterations = particle.actionIterations;

  particle.drawCircle(particle.size);

  if (iterations > 2) particles.makeParticleInactive(particle);
}

class TreeType {
  constructor() {
    throw new Error("Should never actually instantiate this.");
  }

  /** @nocollapse */
  static initTreeParticle(_p: Particle, _oldP: Particle | null): void {}

  /** @nocollapse */
  static branchAngles(_treeParticle: Particle): number[] {
    throw new Error("Branch angles not implemented.");
  }

  /** @nocollapse */
  static branchSpacingFactor(_treeParticle: Particle): number {
    throw new Error("Branch spacing factor not implemented.");
  }
}

/* Standard tree */
class Tree0 extends TreeType {
  /** @nocollapse */
  static branchAngles(treeParticle: Particle): number[] {
    const branchAngle = EIGHTH_PI + Math.random() * QUARTER_PI;
    return [treeParticle.angle + branchAngle, treeParticle.angle - branchAngle];
  }

  /** @nocollapse */
  static branchSpacingFactor(_treeParticle: Particle): number {
    return 0.9;
  }
}

/* Single branch */
class _Tree1 extends TreeType {
  /** @nocollapse */
  static initTreeParticle(p: Particle, oldP: Particle | null): void {
    const branchDirection = oldP
      ? (oldP as any).branchDirection
      : random() < 50
        ? 1
        : -1;
    (p as any).branchDirection = branchDirection;
  }

  /** @nocollapse */
  static branchAngles(treeParticle: Particle): number[] {
    const branchAngle =
      (EIGHTH_PI + Math.random() * EIGHTH_PI) *
      (treeParticle as any).branchDirection;
    return [treeParticle.angle + branchAngle, treeParticle.angle];
  }

  /** @nocollapse */
  static branchSpacingFactor(_treeParticle: Particle): number {
    return 0.7;
  }
}

/* Lots of shallow angle branching */
class Tree2 extends TreeType {
  /** @nocollapse */
  static branchAngles(treeParticle: Particle): number[] {
    const branchAngle = Math.random() * SIXTEENTH_PI + EIGHTH_PI;
    return [
      treeParticle.angle,
      treeParticle.angle + branchAngle,
      treeParticle.angle - branchAngle,
    ];
  }

  /** @nocollapse */
  static branchSpacingFactor(_treeParticle: Particle): number {
    return 0.6;
  }
}

const TREE_TYPES: typeof TreeType[] = [
  Tree0,
  /* A little too cluttered to include Tree1 */
  /* Tree1, */
  Tree2,
];
const NUM_TREE_TYPES: number = TREE_TYPES.length;

function TREE_PARTICLE_INIT(particle: Particle): void {
  particle.setColor(BRANCH);
  particle.size = random() < 50 ? 3 : 4;

  const velocity = 1 + Math.random() * 0.5;
  const angle = -1 * (HALF_PI + EIGHTH_PI - Math.random() * QUARTER_PI);
  particle.setVelocity(velocity, angle);
  (particle as any).generation = 1;
  (particle as any).branchSpacing = 15 + Math.round(Math.random() * 45);
  (particle as any).maxBranches = 1 + Math.round(Math.random() * 2);
  (particle as any).nextBranch = (particle as any).branchSpacing;
  (particle as any).branches = 0;

  /* make it more likely to be a standard tree */
  if (random() < 62) {
    (particle as any).treeType = 0;
  } else {
    (particle as any).treeType =
      1 + Math.floor(Math.random() * (NUM_TREE_TYPES - 1));
  }

  TREE_TYPES[(particle as any).treeType].initTreeParticle(particle, null);
}

function TREE_PARTICLE_ACTION(particle: Particle): void {
  offscreenParticleCtx.beginPath();
  offscreenParticleCtx.lineWidth = particle.size;
  offscreenParticleCtx.strokeStyle = particle.rgbaColor;
  offscreenParticleCtx.lineCap = "round";
  offscreenParticleCtx.moveTo(particle.x, particle.y);
  particle.x += particle.xVelocity;
  particle.y += particle.yVelocity;
  offscreenParticleCtx.lineTo(particle.x, particle.y);
  offscreenParticleCtx.stroke();

  /* Don't grow through WALL */
  if (particle.aboutToHit() === WALL) {
    particles.makeParticleInactive(particle);
    return;
  }

  const iterations = particle.actionIterations;

  if (iterations >= (particle as any).nextBranch) {
    (particle as any).branches++;

    if ((particle as any).maxBranches === 0) {
      particles.makeParticleInactive(particle);
      return;
    }

    const leafBranch =
      particle.color === LEAF ||
      (particle as any).branches === (particle as any).maxBranches;

    const treeInfo = TREE_TYPES[(particle as any).treeType];
    const branchAngles = treeInfo.branchAngles(particle);
    const numBranches = branchAngles.length;
    for (let i = 0; i < numBranches; i++) {
      const b = particles.addActiveParticle(
        TREE_PARTICLE,
        particle.x,
        particle.y,
        particle.i
      );
      if (!b) break;
      (b as any).generation = (particle as any).generation + 1;
      (b as any).maxBranches = Math.max(
        0,
        (particle as any).maxBranches - 1
      );
      (b as any).branchSpacing =
        (particle as any).branchSpacing *
        treeInfo.branchSpacingFactor(particle);
      (b as any).nextBranch = (b as any).branchSpacing;
      (b as any).angle = branchAngles[i];
      b.setVelocity(particle.velocity, (b as any).angle);
      b.size = Math.max(particle.size - 1, 2);
      (b as any).treeType = (particle as any).treeType;
      treeInfo.initTreeParticle(b, particle);

      if (leafBranch) b.setColor(LEAF);
    }

    if ((particle as any).branches >= (particle as any).maxBranches) {
      particles.makeParticleInactive(particle);
      return;
    }

    if ((particle as any).branchSpacing > 45)
      (particle as any).branchSpacing *= 0.8;
    (particle as any).nextBranch =
      iterations +
      (particle as any).branchSpacing * (Math.random() * 0.35 + 0.65);
  }
}

function CHARGED_NITRO_PARTICLE_INIT(particle: Particle): void {
  particle.setColor(FIRE);
  particle.size = 4;
  particle.xVelocity = 0;
  particle.yVelocity = -100;

  /* Search upwards for a WALL collision (but don't check every pixel) */
  (particle as any).minY = -1;
  const step = (3 + Math.round(Math.random() * 2)) * width;
  for (let idx = particle.i; idx > -1; idx -= step) {
    if (gameImagedata32[idx] === WALL) {
      (particle as any).minY = idx / width;
      break;
    }
  }
}

function CHARGED_NITRO_PARTICLE_ACTION(particle: Particle): void {
  offscreenParticleCtx.beginPath();
  offscreenParticleCtx.lineWidth = particle.size;
  offscreenParticleCtx.strokeStyle = particle.rgbaColor;
  offscreenParticleCtx.lineCap = "square";
  offscreenParticleCtx.moveTo(particle.initX, particle.initY);
  particle.x += particle.xVelocity;
  particle.y = Math.max(
    (particle as any).minY,
    particle.y + particle.yVelocity
  );
  offscreenParticleCtx.lineTo(particle.x, particle.y);
  offscreenParticleCtx.stroke();

  if (particle.y <= (particle as any).minY || particle.offCanvas()) {
    particles.makeParticleInactive(particle);
    return;
  }
}

function NUKE_PARTICLE_INIT(particle: Particle): void {
  particle.setColor(FIRE);
  const maxDimension = Math.max(width, height);
  particle.size = maxDimension / 4 + (Math.random() * maxDimension) / 8;
}

function NUKE_PARTICLE_ACTION(particle: Particle): void {
  particle.drawCircle(particle.size);

  if (particle.actionIterations > 4) particles.makeParticleInactive(particle);
}

class Particle {
  type: number;
  initX: number;
  initY: number;
  x: number;
  y: number;
  i: number;
  color: number;
  rgbaColor: string;
  velocity: number;
  angle: number;
  xVelocity: number;
  yVelocity: number;
  size: number;
  actionIterations: number;
  active: boolean;
  next: Particle | null;
  prev: Particle | null;
  reinitialized: boolean;

  static warned_unpaintable_color: boolean = false;

  constructor() {
    this.type = UNKNOWN_PARTICLE;
    this.initX = -1;
    this.initY = -1;
    this.x = -1;
    this.y = -1;
    this.i = -1;
    this.color = 0;
    this.rgbaColor = "rgba(0, 0, 0, 1)";
    this.velocity = 0;
    this.angle = 0;
    this.xVelocity = 0;
    this.yVelocity = 0;
    this.size = 0;
    this.actionIterations = 0;
    this.active = false;
    this.next = null;
    this.prev = null;
    this.reinitialized = false;
  }

  setColor(hexColor: number): void {
    if (!Particle.warned_unpaintable_color) {
      if (!(hexColor in PAINTABLE_PARTICLE_COLORS)) {
        console.log("Unpaintable particle color: " + hexColor);
        Particle.warned_unpaintable_color = true;
      }
    }

    this.color = hexColor;

    const r = hexColor & 0xff;
    const g = (hexColor & 0xff00) >>> 8;
    const b = (hexColor & 0xff0000) >>> 16;
    this.rgbaColor = "rgba(" + r + "," + g + "," + b + ", 1)";
  }

  setRandomColor(whitelist: number[]): void {
    const colorIdx = Math.floor(Math.random() * whitelist.length);
    this.setColor(whitelist[colorIdx]);
  }

  offCanvas(): boolean {
    const x = this.x;
    const y = this.y;
    return x < 0 || x > MAX_X_IDX || y < 0 || y > MAX_Y_IDX;
  }

  setVelocity(velocity: number, angle: number): void {
    this.velocity = velocity;
    this.angle = angle;
    this.xVelocity = velocity * Math.cos(angle);
    this.yVelocity = velocity * Math.sin(angle);
  }

  /*
   * For a spherical particle on a trajectory, figure out what element the
   * particle is about to hit (right at its tip).
   *
   * Expects caller has updated particle's x and y velocity.
   */
  aboutToHit(): number {
    const radius = this.size / 2;
    const theta = Math.atan2(this.yVelocity, this.xVelocity);
    const xPrime = this.x + Math.cos(theta) * radius;
    const yPrime = this.y + Math.sin(theta) * radius;
    const idx = Math.round(xPrime) + Math.round(yPrime) * width;

    if (idx < 0 || idx > MAX_IDX) return BACKGROUND;

    return gameImagedata32[idx];
  }

  drawCircle(radius: number): void {
    offscreenParticleCtx.beginPath();
    offscreenParticleCtx.lineWidth = 0;
    offscreenParticleCtx.fillStyle = this.rgbaColor;
    offscreenParticleCtx.arc(this.x, this.y, radius, 0, TWO_PI);
    offscreenParticleCtx.fill();
  }
}

/*
 * Two doubly-linked lists: one for active and one for
 * inactive particles
 */
class ParticleList {
  activeHead: Particle | null;
  activeSize: number;
  inactiveHead: Particle | null;
  inactiveSize: number;
  particleCounts: Uint32Array;

  constructor() {
    this.activeHead = null;
    this.activeSize = 0;
    this.inactiveHead = null;
    this.inactiveSize = 0;
    this.particleCounts = new Uint32Array(__particleInit.length);

    /* This probably isn't necessary, but I don't trust javascript */
    for (let i = 0; i < this.particleCounts.length; i++) {
      this.particleCounts[i] = 0;
    }
  }

  addActiveParticle(
    type: number,
    x: number,
    y: number,
    i: number
  ): Particle | null {
    if (this.inactiveSize === 0) return null;

    const particle = this.inactiveHead!;
    this.inactiveHead = this.inactiveHead!.next;
    if (this.inactiveHead) this.inactiveHead.prev = null;
    this.inactiveSize--;

    if (!this.activeHead) {
      particle.next = null;
      particle.prev = null;
      this.activeHead = particle;
    } else {
      this.activeHead.prev = particle;
      particle.next = this.activeHead;
      particle.prev = null;
      this.activeHead = particle;
    }
    this.activeSize++;

    particle.active = true;
    particle.reinitialized = false;
    particle.actionIterations = 0;
    particle.type = type;
    particle.initX = x;
    particle.initY = y;
    particle.x = x;
    particle.y = y;
    particle.i = i;
    this.particleCounts[type]++;
    __particleInit[type](particle);

    return particle;
  }

  makeParticleInactive(particle: Particle): void {
    particle.active = false;
    this.particleCounts[particle.type]--;
    particle.type = UNKNOWN_PARTICLE;
    if (particle.prev) {
      particle.prev.next = particle.next;
    }
    if (particle.next) {
      particle.next.prev = particle.prev;
    }
    if (particle === this.activeHead) {
      this.activeHead = particle.next;
    }
    this.activeSize--;

    if (!this.inactiveHead) {
      particle.next = null;
      particle.prev = null;
      this.inactiveHead = particle;
    } else {
      this.inactiveHead.prev = particle;
      particle.next = this.inactiveHead;
      particle.prev = null;
      this.inactiveHead = particle;
    }
    this.inactiveSize++;
  }

  inactivateAll(): void {
    let particle = this.activeHead;
    while (particle) {
      const next = particle.next;
      this.makeParticleInactive(particle);
      particle = next;
    }
  }

  reinitializeParticle(particle: Particle, newType: number): void {
    if (!particle.active) throw new Error("Can only be used with active particles");

    this.particleCounts[particle.type]--;
    this.particleCounts[newType]++;
    particle.type = newType;
    particle.reinitialized = true;
    __particleInit[newType](particle);
  }

  particleActive(particleType: number): boolean {
    return this.particleCounts[particleType] > 0;
  }
}

export const particles = new ParticleList();

/*
 * When we copy the particle strokes to the main canvas, some
 * of the colors will not match any elements (due to anti-aliasing
 * of the stroke). We need a fast way to know if a given color is
 * a valid color for painting. Hence, this dictionary of colors that can
 * be copied from the particle canvas to the main canvas.
 */
const PAINTABLE_PARTICLE_COLORS: Record<number, null> = {};

const MAGIC_COLORS: number[] = [];

export function initParticles(): void {
  if (__particleInit.length !== __particleActions.length)
    throw new Error("Particle arrays must be same length");

  let numParticlesToCreate = MAX_NUM_PARTICLES;
  let prevParticle: Particle;

  /* Setup the head */
  particles.inactiveHead = new Particle();
  particles.inactiveSize++;
  prevParticle = particles.inactiveHead;
  numParticlesToCreate--;

  /*
   * We pre-allocate all of the particles, rather than create them on demand.
   * This avoids latency spikes due to garbage collection reap. It does require
   * that we use two linked lists to keep track of them all, but that's fine.
   */
  while (numParticlesToCreate > 0) {
    const particle = new Particle();

    particle.prev = prevParticle;
    prevParticle.next = particle;
    particles.inactiveSize++;

    prevParticle = particle;

    numParticlesToCreate--;
  }

  offscreenParticleCanvas.width = width;
  offscreenParticleCanvas.height = height;

  PAINTABLE_PARTICLE_COLORS[FIRE] = null;
  PAINTABLE_PARTICLE_COLORS[WALL] = null;
  PAINTABLE_PARTICLE_COLORS[ROCK] = null;
  PAINTABLE_PARTICLE_COLORS[LAVA] = null;
  PAINTABLE_PARTICLE_COLORS[PLANT] = null;
  PAINTABLE_PARTICLE_COLORS[SPOUT] = null;
  PAINTABLE_PARTICLE_COLORS[WELL] = null;
  PAINTABLE_PARTICLE_COLORS[WAX] = null;
  PAINTABLE_PARTICLE_COLORS[ICE] = null;
  PAINTABLE_PARTICLE_COLORS[BRANCH] = null;
  PAINTABLE_PARTICLE_COLORS[LEAF] = null;
  Object.freeze(PAINTABLE_PARTICLE_COLORS);

  /* All of these are also in PAINTABLE_PARTICLE_COLORS */
  MAGIC_COLORS.push(WALL);
  MAGIC_COLORS.push(PLANT);
  MAGIC_COLORS.push(SPOUT);
  MAGIC_COLORS.push(WELL);
  MAGIC_COLORS.push(WAX);
  MAGIC_COLORS.push(ICE);
  Object.freeze(MAGIC_COLORS);
}

export function updateParticles(): void {
  if (!particles.activeHead) return;

  const canvasWidth = offscreenParticleCanvas.width;
  const canvasHeight = offscreenParticleCanvas.height;

  /* reset the particle canvas */
  offscreenParticleCtx.beginPath();
  offscreenParticleCtx.fillStyle = "rgba(0, 0, 0, 1)";
  offscreenParticleCtx.rect(0, 0, canvasWidth, canvasHeight);
  offscreenParticleCtx.fill();

  /* perform particle actions */
  let particle: Particle | null = particles.activeHead;
  while (particle) {
    /* grab next before doing action, as next could change */
    const next: Particle | null = particle.next;
    particle.actionIterations++;
    __particleActions[particle.type](particle);
    particle = next;
  }

  /* move particle draw state to main canvas */
  const particleImageData = offscreenParticleCtx.getImageData(
    0,
    0,
    canvasWidth,
    canvasHeight
  );
  const particleImageData32 = new Uint32Array(particleImageData.data.buffer);
  let x: number, y: number;
  let __yOffset = 0;
  const aliasingSearchDistance = 3;
  for (y = 0; y !== canvasHeight; y++) {
    const yOffset = __yOffset; /* optimization: make const copy */
    for (x = 0; x !== canvasWidth; x++) {
      const i = x + yOffset;
      const particleColor = particleImageData32[i];

      if (particleColor === 0xff000000) continue;

      /*
       * ImageData will container other colors due to anti-aliasing.
       * However, we can only copy over valid colors to the main canvas.
       *
       * If the color appears to be invalid, it is likely right along the
       * edge of a valid color. In this case, we can search nearby pixels
       * for such a color.
       *
       * The motivation for this is that when many overlapping shapes are
       * drawn on the canvas (ie. the various particles), the aliased border
       * of each sub-object created gaps of invalid colors.
       */
      if (particleColor in PAINTABLE_PARTICLE_COLORS) {
        gameImagedata32[i] = particleColor;
        continue;
      } else {
        let searchColor: number;
        if (x - aliasingSearchDistance >= 0) {
          searchColor = particleImageData32[i - aliasingSearchDistance];
          if (searchColor in PAINTABLE_PARTICLE_COLORS) {
            gameImagedata32[i] = searchColor;
            continue;
          }
        }
        if (x + aliasingSearchDistance <= MAX_X_IDX) {
          searchColor = particleImageData32[i + aliasingSearchDistance];
          if (searchColor in PAINTABLE_PARTICLE_COLORS) {
            gameImagedata32[i] = searchColor;
            continue;
          }
        }
        if (y - aliasingSearchDistance >= 0) {
          searchColor =
            particleImageData32[i - aliasingSearchDistance * width];
          if (searchColor in PAINTABLE_PARTICLE_COLORS) {
            gameImagedata32[i] = searchColor;
            continue;
          }
        }
        if (y + aliasingSearchDistance <= MAX_Y_IDX) {
          searchColor =
            particleImageData32[i + aliasingSearchDistance * width];
          if (searchColor in PAINTABLE_PARTICLE_COLORS) {
            gameImagedata32[i] = searchColor;
            continue;
          }
        }
      }
    }
    __yOffset += canvasWidth;
  }
}
