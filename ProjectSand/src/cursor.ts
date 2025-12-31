/*
 * Deals with the user drawing on the canvas.
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

import Matter from "matter-js";
import { width, height } from "./canvasConfig.js";
import { TWO_PI, docOffsetLeft, docOffsetTop, distance } from "./util.js";
import { BACKGROUND, ZOMBIE, WALL } from "./elements.js";
import { PEN_SIZES, DEFAULT_PEN_IDX } from "./menu.js";
import {
  MAX_X_IDX,
  MAX_Y_IDX,
  onscreenCanvas,
  gameImagedata32,
} from "./game.js";
import { softBodyMouse } from "./softBody.js";

/*
 * Cursor options. Controlled via the menu.
 */
let PENSIZE: number;
let SELECTED_ELEM: number;
let OVERWRITE_ENABLED: boolean;

// Export getters and setters since ES modules don't allow reassigning imports
export function getPENSIZE(): number {
  return PENSIZE;
}
export function setPENSIZE(value: number): void {
  PENSIZE = value;
}
export function getSELECTED_ELEM(): number {
  return SELECTED_ELEM;
}
export function setSELECTED_ELEM(value: number): void {
  SELECTED_ELEM = value;
}
export function getOVERWRITE_ENABLED(): boolean {
  return OVERWRITE_ENABLED;
}
export function setOVERWRITE_ENABLED(value: boolean): void {
  OVERWRITE_ENABLED = value;
}

/*
 * Offscreen canvas for drawing user stroke. We draw on
 * this canvas and then transfer the data to the main
 * canvas. We can't use the built-in drawing methods directly
 * on the main game canvas, since it produces off-shade colors
 * in order to perform anti-aliasing.
 */
const userstrokeCanvas = document.createElement("canvas");
userstrokeCanvas.width = width;
userstrokeCanvas.height = height;
const userstrokeCtx = userstrokeCanvas.getContext("2d", {
  alpha: false,
  willReadFrequently: true,
}) as CanvasRenderingContext2D;

const CURSORS: Cursor[] = [];

/* Generic cursor */
class Cursor {
  x: number;
  y: number;
  prevX: number;
  prevY: number;
  initX: number;
  initY: number;
  documentX: number;
  documentY: number;
  isDown: boolean;
  inCanvas: boolean;
  canvas: HTMLCanvasElement;

  constructor(canvas: HTMLCanvasElement) {
    /* x, y, prevX, and prevY are coordinates *inside* the canvas */
    this.x = 0;
    this.y = 0;
    this.prevX = 0;
    this.prevY = 0;
    this.initX = 0;
    this.initY = 0;

    /*
     * documentX and documentY are coordinates relative to the canvas, but
     * be outside the canvas (ie. negative)
     */
    this.documentX = 0;
    this.documentY = 0;

    this.isDown = false;
    this.inCanvas = false;
    this.canvas = canvas;
  }

  notifyCursorUp(): void {
    /*
     * If drawStroke() adjusted the mouse offset, we need to move it back.
     */
    if (softBodyMouse.offset.x != 0) {
      Matter.Mouse.setOffset(softBodyMouse, { x: 0, y: 0 });
    }
  }

  canvasCursorDown(x: number, y: number): void {
    this.isDown = true;
    this.inCanvas = true;

    this.prevX = x;
    this.prevY = y;
    this.x = x;
    this.y = y;
    this.initX = x;
    this.initY = y;
  }

  canvasCursorMove(getPos: () => [number, number]): void {
    if (!this.isDown) return;

    const pos = getPos();

    this.x = pos[0];
    this.y = pos[1];
  }

  canvasCursorEnter(
    getInnerCoords: (self: Cursor) => [number, number],
    getOuterCoords: (self: Cursor) => [number, number]
  ): void {
    this.inCanvas = true;

    if (!this.isDown) return;

    const innerCoords = getInnerCoords(this);
    const outerCoords = getOuterCoords(this);

    Cursor.interpolateCursorBorderPosition(innerCoords, outerCoords);

    this.prevX = outerCoords[0];
    this.prevY = outerCoords[1];
    this.x = innerCoords[0];
    this.y = innerCoords[1];
  }

  canvasCursorLeave(
    getOuterCoords: (self: Cursor) => [number, number]
  ): void {
    this.inCanvas = false;

    if (!this.isDown) return;

    const outerCoords = getOuterCoords(this);
    Cursor.interpolateCursorBorderPosition(
      [this.prevX, this.prevY],
      outerCoords
    );

    this.x = outerCoords[0];
    this.y = outerCoords[1];
  }

  documentCursorMove(getPos: () => [number, number]): void {
    if (!this.isDown) return;
    if (this.inCanvas) return;

    const pos = getPos();
    this.documentX = pos[0];
    this.documentY = pos[1];
  }

  documentCursorUp(): void {
    this.isDown = false;
    this.notifyCursorUp();
  }

  documentCursorDown(
    e: MouseEvent,
    getPos: (self: Cursor) => [number, number]
  ): void {
    if (e.target == onscreenCanvas) return;
    if (this.isDown) return;

    this.isDown = true;
    this.inCanvas = false;

    /*
     * prevent drawStroke() from mistakenly drawing another segment if the
     * cursor was previously in the canvas
     */
    this.prevX = this.x;
    this.prevY = this.y;

    this.initX = this.x;
    this.initY = this.y;

    const pos = getPos(this);
    this.documentX = pos[0];
    this.documentY = pos[1];
  }

  documentVisibilityChange(_e: Event): void {}

  /*
   * Given that the cursor moved from coordinates outside the canvas
   * to coordinates inside the canvas, interpolate the coordinate that
   * the cursor passed through on the border of the canvas.
   *
   * Modifies and returns the result in outercoords.
   *
   * Note that outercoords is relative to the canvas, not the document.
   */
  static interpolateCursorBorderPosition(
    innercoords: [number, number],
    outercoords: [number, number]
  ): [number, number] {
    /* Get line parameters */
    let dy = innercoords[1] - outercoords[1];
    let dx = innercoords[0] - outercoords[0];
    if (dy === 0) dy = 0.001;
    if (dx === 0) dx = 0.001;
    const slope = dy / dx;
    const y_intercept = innercoords[1] - slope * innercoords[0];

    if (outercoords[0] < 0) {
      outercoords[0] = 0;
      outercoords[1] = y_intercept;
    } else if (outercoords[0] > MAX_X_IDX) {
      outercoords[0] = MAX_X_IDX;
      outercoords[1] = slope * MAX_X_IDX + y_intercept;
    }

    if (outercoords[1] < 0) {
      outercoords[1] = 0;
      outercoords[0] = (0 - y_intercept) / slope;
    } else if (outercoords[1] > MAX_Y_IDX) {
      outercoords[1] = MAX_Y_IDX;
      outercoords[0] = (MAX_Y_IDX - y_intercept) / slope;
    }

    outercoords[0] = Math.floor(outercoords[0]);
    outercoords[1] = Math.floor(outercoords[1]);

    /* Just in case... */
    outercoords[0] = Math.max(Math.min(outercoords[0], MAX_X_IDX), 0);
    outercoords[1] = Math.max(Math.min(outercoords[1], MAX_Y_IDX), 0);

    return outercoords;
  }

  /*
   * This is a lot of code, but the idea here is simple.
   * We use a subset of the cursor canvas just big enough
   * to fit the user stroke (ie. why bother use the entire
   * width*height canvas if the userstroke is really small).
   * However, this means that we need to do a bit of math to
   * translate the cursor stroke into its proper position on
   * the main canvas.
   */
  drawStroke(): void {
    if (!this.isDown) return;
    if (!this.inCanvas) {
      if (this.prevX === this.x && this.prevY === this.y) return;
    }

    const color = getSELECTED_ELEM();

    /* We only want to drag zombies, not draw them */
    if (color === ZOMBIE) {
      return;
    }

    /*
     * A bit of a hack, but the goal is to avoid triggering soft body dragging if
     * we're already drawing an element stroke. There are two parts to the hack.
     * Part 1 is detection by using a rough heuristic of whether the cursor has
     * moved enough since the last stroke draw. Part 2 is "disabling" the soft body
     * mouse by moving it outside the canvas.
     */
    if (
      softBodyMouse.offset.x === 0 &&
      Math.pow(this.initX - this.x, 2) + Math.pow(this.initY - this.y, 2) >
        Math.pow(20, 2)
    ) {
      Matter.Mouse.setOffset(softBodyMouse, { x: width + 1, y: height + 1 });
    }

    const overwrite = getOVERWRITE_ENABLED() || color === BACKGROUND;
    const r = color & 0xff;
    const g = (color & 0xff00) >>> 8;
    const b = (color & 0xff0000) >>> 16;
    /*
     * As an optimization, we skip over 0xff000000 below.
     * If this is our color (ie. eraser), we need to slightly
     * modify it.
     */
    const colorString =
      color !== 0xff000000
        ? "rgba(" + r + "," + g + "," + b + ", 1)"
        : "rgba(1, 0, 0, 1)";

    /* (x1, y1) is the leftmost coordinate */
    const x1 = Math.min(this.prevX, this.x);
    const x2 = Math.max(this.prevX, this.x);
    const y1 = this.prevX <= this.x ? this.prevY : this.y;
    const y2 = this.prevX <= this.x ? this.y : this.prevY;

    this.prevX = this.x;
    this.prevY = this.y;

    const pensize = getPENSIZE();
    const strokeBuffer = Math.ceil(pensize / 2);
    const x_translate = x1 - strokeBuffer;
    const y_translate = Math.min(y1, y2) - strokeBuffer;
    const x1_relative = x1 - x_translate;
    const y1_relative = y1 - y_translate;
    const x2_relative = x2 - x_translate;
    const y2_relative = y2 - y_translate;

    /* Initialize offscreen canvas. Ensure our drawing area starts black */
    const userstroke_width = x2_relative + pensize + 2;
    const userstroke_height = Math.max(y1_relative, y2_relative) + pensize + 2;
    if (userstrokeCanvas.width < userstroke_width)
      userstrokeCanvas.width = userstroke_width;
    if (userstrokeCanvas.height < userstroke_height)
      userstrokeCanvas.height = userstroke_height;

    userstrokeCtx.beginPath();
    userstrokeCtx.rect(0, 0, userstroke_width, userstroke_height);
    userstrokeCtx.fillStyle = "rgba(0, 0, 0, 1)";
    userstrokeCtx.fill();

    /*
     * Some browsers *cough* Edge *cough* Safari *cough* can't
     * handle drawing a line if the start and end are the same point.
     * So, special case this and draw a circle instead.
     */
    if (x1_relative === x2_relative && y1_relative === y2_relative) {
      userstrokeCtx.beginPath();
      userstrokeCtx.lineWidth = 0;
      userstrokeCtx.fillStyle = colorString;
      userstrokeCtx.arc(x1_relative, y1_relative, pensize / 2, 0, TWO_PI);
      userstrokeCtx.fill();
    } else {
      userstrokeCtx.lineWidth = pensize;
      userstrokeCtx.strokeStyle = colorString;
      userstrokeCtx.lineCap = "round";
      userstrokeCtx.beginPath();
      userstrokeCtx.moveTo(x1_relative, y1_relative);
      userstrokeCtx.lineTo(x2_relative, y2_relative);
      userstrokeCtx.stroke();
    }

    const strokeImageData = userstrokeCtx.getImageData(
      0,
      0,
      userstroke_width,
      userstroke_height
    );
    const strokeImageData32 = new Uint32Array(strokeImageData.data.buffer);

    /* Transfer line from offscreen canvas to main canvas */
    let x: number, y: number;
    const xStart = Math.max(0, -1 * x_translate);
    const yStart = Math.max(0, -1 * y_translate);
    const xTerminate = Math.min(userstroke_width, width - x_translate);
    const yTerminate = Math.min(userstroke_height, height - y_translate);
    if (xStart > xTerminate || yStart > yTerminate) {
      console.log("Bug in userstroke drawing");
      return;
    }
    for (y = yStart; y !== yTerminate; y++) {
      const y_absolute = y + y_translate;
      const offset_absolute = y_absolute * width;
      const offset_relative = y * userstroke_width;
      for (x = xStart; x !== xTerminate; x++) {
        const x_absolute = x + x_translate;

        /*
         * Note that not all pixels will be equal to 'color'; browser will
         * anti-alias the line, which will result in some grayscale colors as
         * well. So, it is sufficient (and necessary) to consider a pixel
         * colored as long as it is not black.
         */
        if (strokeImageData32[x + offset_relative] !== 0xff000000) {
          const absIdx = x_absolute + offset_absolute;
          if (overwrite || gameImagedata32[absIdx] === BACKGROUND)
            gameImagedata32[absIdx] = color;
        }
      }
    }
  }
}

/*
 * Note: the name "Mouse" conflicts with the declaration from matter.js,
 * so we use MouseCursor.
 */
class MouseCursor extends Cursor {
  shiftStartX: number;
  shiftStartY: number;
  shiftPressed: boolean;
  lineDirection: number; /* for use with shift key */

  static NO_DIRECTION: number;
  static HORIZONTAL: number;
  static VERTICAL: number;
  static DIAGONAL_UP: number;
  static DIAGONAL_DOWN: number;

  constructor(canvas: HTMLCanvasElement) {
    super(canvas);

    this.shiftStartX = 0;
    this.shiftStartY = 0;
    this.shiftPressed = false;
    this.lineDirection = MouseCursor.NO_DIRECTION; /* for use with shift key */
  }

  canvasMouseDown(e: MouseEvent): void {
    const mousePos = MouseCursor.getMousePos(e, true, this.canvas);

    /* Fix bug that left the canvas stuck in "shift" mode */
    if (this.shiftPressed && !e.shiftKey) this.shiftPressed = false;

    if (this.shiftPressed) {
      this.shiftStartX = mousePos[0];
      this.shiftStartY = mousePos[1];
      this.lineDirection = MouseCursor.NO_DIRECTION;
    }

    super.canvasCursorDown(mousePos[0], mousePos[1]);
  }

  canvasMouseMove(e: MouseEvent): void {
    const canvas = this.canvas;
    const getPos = function (): [number, number] {
      return MouseCursor.getMousePos(e, true, canvas);
    };

    super.canvasCursorMove(getPos);
  }

  canvasMouseEnter(e: MouseEvent): void {
    const canvas = this.canvas;
    const getInnerPos = function (_self: Cursor): [number, number] {
      return MouseCursor.getMousePos(e, true, canvas);
    };
    const getOuterPos = function (self: Cursor): [number, number] {
      return [self.documentX, self.documentY];
    };

    super.canvasCursorEnter(getInnerPos, getOuterPos);

    /*
     * relies on the fact that super.CanvasCursorEnter has already fixed
     * prevX/prevY to be on the canvas border
     */
    if (
      this.isDown &&
      this.shiftPressed &&
      this.lineDirection === MouseCursor.NO_DIRECTION
    ) {
      this.shiftStartX = this.prevX;
      this.shiftStartY = this.prevY;
    }
  }

  canvasMouseLeave(e: MouseEvent): void {
    const canvas = this.canvas;
    const getOuterPos = function (_self: Cursor): [number, number] {
      return MouseCursor.getMousePos(e, false, canvas);
    };

    super.canvasCursorLeave(getOuterPos);
  }

  documentMouseMove(e: MouseEvent): void {
    if (e.target == onscreenCanvas) return;

    const canvas = this.canvas;
    const getPos = function (): [number, number] {
      return MouseCursor.getMousePos(e, false, canvas);
    };

    super.documentCursorMove(getPos);
  }

  documentMouseUp(_e: MouseEvent | null): void {
    /*
     * Don't use e, may be passed as null. Assigning here explicitly to avoid
     * bugs.
     */
    // e is intentionally set to null to prevent memory leaks

    this.lineDirection = MouseCursor.NO_DIRECTION;

    super.documentCursorUp();
  }

  documentMouseDown(e: MouseEvent): void {
    /* only need handling when clicking outside the canvas */
    if (e.target == onscreenCanvas) return;

    const canvas = this.canvas;
    const getPos = function (self: Cursor): [number, number] {
      return MouseCursor.getMousePos(e, false, canvas);
    };

    /* Fix bug that left the canvas stuck in "shift" mode */
    if (this.shiftPressed && !e.shiftKey) this.shiftPressed = false;

    if (this.shiftPressed) this.lineDirection = MouseCursor.NO_DIRECTION;

    super.documentCursorDown(e, getPos);
  }

  static getMousePos(
    e: MouseEvent,
    withinCanvas: boolean,
    canvas: HTMLCanvasElement
  ): [number, number] {
    let x: number, y: number;

    if (withinCanvas) {
      x = e.offsetX;
      y = e.offsetY;

      if (x < 0) x = 0;
      else if (x >= width) x = MAX_X_IDX;

      if (y < 0) y = 0;
      else if (y >= height) y = MAX_Y_IDX;
    } else {
      x = e.pageX - docOffsetLeft(canvas);
      y = e.pageY - docOffsetTop(canvas);
    }

    return [Math.round(x), Math.round(y)];
  }

  documentKeyDown(e: KeyboardEvent): void {
    if (!e.shiftKey) return;

    if (this.shiftPressed) return;

    this.shiftPressed = true;
    this.lineDirection = MouseCursor.NO_DIRECTION;

    if (!this.isDown) return;

    if (!this.inCanvas) return;

    this.shiftStartX = this.x;
    this.shiftStartY = this.y;
  }

  documentKeyUp(e: KeyboardEvent): void {
    if (!e.shiftKey && this.shiftPressed) this.shiftPressed = false;
  }

  documentVisibilityChange(e: Event): void {
    const visibilityState = document.visibilityState;
    if (visibilityState == "hidden") {
      this.documentMouseUp(null);
      this.shiftPressed = false;
    }

    super.documentVisibilityChange(e);
  }

  /*
   * We draw straight lines when shift is held down.
   *
   * If this returns true, skip drawing the stroke (we need
   * to figure out what direction the line is going).
   */
  handleShift(): boolean {
    if (!this.isDown) return false;

    if (!this.shiftPressed) return false;

    if (!this.inCanvas) {
      if (this.prevX === this.x && this.prevY === this.y) return false;
    }

    if (this.lineDirection === MouseCursor.NO_DIRECTION) {
      if (!this.inCanvas) return false;

      const dx = this.x - this.shiftStartX;
      const dy = this.y - this.shiftStartY;
      const absDx = Math.abs(dx);
      const absDy = Math.abs(dy);

      /* Wait to see what direction the mouse is going in */
      if (Math.max(absDx, absDy) < 8) return true;

      if (Math.abs(absDx - absDy) < 5) {
        if (dy * dx < 0) this.lineDirection = MouseCursor.DIAGONAL_DOWN;
        else this.lineDirection = MouseCursor.DIAGONAL_UP;
      } else if (absDx > absDy) {
        this.lineDirection = MouseCursor.HORIZONTAL;
      } else {
        this.lineDirection = MouseCursor.VERTICAL;
      }
    }

    const direction = this.lineDirection;
    if (direction === MouseCursor.HORIZONTAL) {
      this.prevY = this.shiftStartY;
      this.y = this.shiftStartY;
    } else if (direction === MouseCursor.VERTICAL) {
      this.prevX = this.shiftStartX;
      this.x = this.shiftStartX;
    } else if (
      direction === MouseCursor.DIAGONAL_DOWN ||
      direction === MouseCursor.DIAGONAL_UP
    ) {
      this.prevX = this.shiftStartX;
      this.prevY = this.shiftStartY;
      const slope = direction === MouseCursor.DIAGONAL_DOWN ? -1 : 1;
      const yIntercept = this.shiftStartY - slope * this.shiftStartX;

      const yAdjusted = slope * this.x + yIntercept;
      const xAdjusted = (this.y - yIntercept) / slope;
      if (
        distance(xAdjusted, this.y, this.shiftStartX, this.shiftStartY) >
        distance(this.x, yAdjusted, this.shiftStartX, this.shiftStartY)
      ) {
        this.x = xAdjusted;
      } else {
        this.y = yAdjusted;
      }
    }

    return false;
  }

  drawStroke(): void {
    /* alters prevX, prevY, x, and y to handle drawing in straight lines */
    if (this.handleShift()) return;

    super.drawStroke();
  }
}

/* Touch cursor (ie. mobile users) */
class TouchCursor extends Cursor {
  constructor(canvas: HTMLCanvasElement) {
    super(canvas);
  }

  canvasTouchStart(e: TouchEvent): boolean {
    const pos = TouchCursor.getTouchPos(e);

    if (!pos) return false;

    super.canvasCursorDown(pos[0], pos[1]);

    /* prevent scrolling */
    e.preventDefault();

    return false;
  }

  canvasTouchEnd(e: TouchEvent): boolean {
    super.documentCursorUp();

    /* prevent scrolling */
    e.preventDefault();

    return false;
  }

  canvasTouchMove(e: TouchEvent): boolean {
    const pos = TouchCursor.getTouchPos(e);

    if (!pos) return false;

    const getPos = function (): [number, number] {
      return pos;
    };

    super.canvasCursorMove(getPos);

    /* prevent scrolling */
    e.preventDefault();

    return false;
  }

  static getTouchPos(e: TouchEvent): [number, number] | null {
    if (!e.touches) return null;

    const touch = e.touches[0];
    if (!touch) return null;

    const rect = (e.target as HTMLElement).getBoundingClientRect();
    let x = Math.round(touch.pageX - rect.left - window.scrollX);
    let y = Math.round(touch.pageY - rect.top - window.scrollY);

    if (x < 0) x = 0;
    else if (x >= width) x = MAX_X_IDX;

    if (y < 0) y = 0;
    else if (y >= height) y = MAX_Y_IDX;

    return [x, y];
  }
}

export function initCursors(): void {
  setPENSIZE(PEN_SIZES[DEFAULT_PEN_IDX]);
  setSELECTED_ELEM(WALL);
  setOVERWRITE_ENABLED(true);

  /* Set up direction constants for drawing straight lines */
  MouseCursor.NO_DIRECTION = 0;
  MouseCursor.HORIZONTAL = 1;
  MouseCursor.VERTICAL = 2;
  MouseCursor.DIAGONAL_UP = 3;
  MouseCursor.DIAGONAL_DOWN = 4;

  /*
   * Setting the event handler functions in this way allows the handlers
   * to properly access the 'this' pointer.
   */
  const mouse = new MouseCursor(onscreenCanvas);
  onscreenCanvas.onmousedown = function (e: MouseEvent) {
    mouse.canvasMouseDown(e);
  };
  onscreenCanvas.onmousemove = function (e: MouseEvent) {
    mouse.canvasMouseMove(e);
  };
  onscreenCanvas.onmouseleave = function (e: MouseEvent) {
    mouse.canvasMouseLeave(e);
  };
  onscreenCanvas.onmouseenter = function (e: MouseEvent) {
    mouse.canvasMouseEnter(e);
  };
  document.onmouseup = function (e: MouseEvent) {
    mouse.documentMouseUp(e);
  };
  document.onmousedown = function (e: MouseEvent) {
    mouse.documentMouseDown(e);
  };
  document.onmousemove = function (e: MouseEvent) {
    mouse.documentMouseMove(e);
  };
  document.onkeydown = function (e: KeyboardEvent) {
    mouse.documentKeyDown(e);
  };
  document.onkeyup = function (e: KeyboardEvent) {
    mouse.documentKeyUp(e);
  };
  document.onvisibilitychange = function (e: Event) {
    mouse.documentVisibilityChange(e);
  };

  const touchCursor = new TouchCursor(onscreenCanvas);
  onscreenCanvas.addEventListener("touchstart", function (e: TouchEvent) {
    touchCursor.canvasTouchStart(e);
  });
  onscreenCanvas.addEventListener("touchend", function (e: TouchEvent) {
    touchCursor.canvasTouchEnd(e);
  });
  onscreenCanvas.addEventListener("touchmove", function (e: TouchEvent) {
    touchCursor.canvasTouchMove(e);
  });

  CURSORS.push(mouse);
  CURSORS.push(touchCursor);
  Object.freeze(CURSORS);
}

/* Draw the userstroke on the stroke canvas */
export function updateUserStroke(): void {
  const numCursors = CURSORS.length;
  for (let i = 0; i !== numCursors; i++) {
    CURSORS[i].drawStroke();
  }
}

