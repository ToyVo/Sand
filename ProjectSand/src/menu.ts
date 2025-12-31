/*
 * Code for the menu options.
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
  BACKGROUND,
  WALL,
  SAND,
  WATER,
  PLANT,
  FIRE,
  SALT,
  OIL,
  SPOUT,
  WELL,
  TORCH,
  GUNPOWDER,
  WAX,
  NITRO,
  NAPALM,
  C4,
  LAVA,
  CRYO,
  FUSE,
  MYSTERY,
  CONCRETE,
  METHANE,
  SOIL,
  ACID,
  THERMITE,
  ICE,
  ZOMBIE,
  SALT_WATER,
  FALLING_WAX,
  CHILLED_ICE,
  ROCK,
  STEAM,
  WET_SOIL,
  BRANCH,
  LEAF,
  POLLEN,
  CHARGED_NITRO,
  BURNING_THERMITE,
  getVOID_MODE_ENABLED,
  setVOID_MODE_ENABLED,
} from "./elements.js";
import { setZombieCount } from "./zombies.js";
import {
  setFPS,
  clearGameCanvas,
  saveGameCanvas,
  loadGameCanvas,
} from "./game.js";
import {
  setPENSIZE,
  getSELECTED_ELEM,
  setSELECTED_ELEM,
  getOVERWRITE_ENABLED,
  setOVERWRITE_ENABLED,
} from "./cursor.js";
import { DEFAULT_FPS, MAX_ZOMBIES, MAX_FPS } from "./canvasConfig.js";
import {
  SPIGOT_ELEMENT_OPTIONS,
  SPIGOT_SIZE_OPTIONS,
  DEFAULT_SPIGOT_SIZE_IDX,
  SPIGOT_ELEMENTS,
  SPIGOT_SIZES,
} from "./spigots.js";

/* Configuration of the menu */
const ELEMENT_MENU_ELEMENTS_PER_ROW: number = 4;
export const PEN_SIZES: number[] = [2, 4, 8, 16, 32, 64];
const PEN_SIZE_LABELS: string[] = ["1px", "2px", "4px", "8px", "16px", "32px"];
export const DEFAULT_PEN_IDX: number = 1;

/* Elements listed in the menu */
// prettier-ignore
const elementMenuItems: number[] = [
  WALL, SAND, WATER, PLANT,
  FIRE, SPOUT, WELL, SALT,
  OIL, WAX, TORCH, ICE,
  GUNPOWDER, NAPALM, NITRO, C4,
  LAVA, CRYO, FUSE, MYSTERY,
  CONCRETE, METHANE, SOIL, ACID,
  THERMITE, BACKGROUND, ZOMBIE,
  SALT_WATER, FALLING_WAX, CHILLED_ICE,
  ROCK, STEAM, WET_SOIL, BRANCH,
  LEAF, POLLEN, CHARGED_NITRO, BURNING_THERMITE,
];

const menuNames: Record<number, string> = {};
menuNames[WALL] = "WALL";
menuNames[SAND] = "SAND";
menuNames[WATER] = "WATER";
menuNames[PLANT] = "PLANT";
menuNames[FIRE] = "FIRE";
menuNames[SALT] = "SALT";
menuNames[OIL] = "OIL";
menuNames[SPOUT] = "SPOUT";
menuNames[WELL] = "WELL";
menuNames[TORCH] = "TORCH";
menuNames[GUNPOWDER] = "GUNPOWDER";
menuNames[WAX] = "WAX";
menuNames[NITRO] = "NITRO";
menuNames[NAPALM] = "NAPALM";
menuNames[C4] = "C-4";
menuNames[CONCRETE] = "CONCRETE";
menuNames[BACKGROUND] = "ERASER";
menuNames[FUSE] = "FUSE";
menuNames[ICE] = "ICE";
menuNames[LAVA] = "LAVA";
menuNames[METHANE] = "METHANE";
menuNames[CRYO] = "CRYO";
menuNames[MYSTERY] = "???";
menuNames[SOIL] = "SOIL";
menuNames[ACID] = "ACID";
menuNames[THERMITE] = "THERMITE";
menuNames[ZOMBIE] = "HAND";
menuNames[SALT_WATER] = "SALT WATER";
menuNames[FALLING_WAX] = "FALLING WAX";
menuNames[CHILLED_ICE] = "CHILLED ICE";
menuNames[ROCK] = "ROCK";
menuNames[STEAM] = "STEAM";
menuNames[WET_SOIL] = "WET SOIL";
menuNames[BRANCH] = "BRANCH";
menuNames[LEAF] = "LEAF";
menuNames[POLLEN] = "POLLEN";
menuNames[CHARGED_NITRO] = "CHARGED NITRO";
menuNames[BURNING_THERMITE] = "BURNING THERMITE";

/*
 * Some element colors do not have very good contrast against
 * the menu background. For these elements, we use a replacement
 * color for the menu text.
 */
const menuAltColors: Record<number, string> = {};
menuAltColors[WATER] = "rgb(0, 130, 255)";
menuAltColors[WALL] = "rgb(160, 160, 160)";
menuAltColors[BACKGROUND] = "rgb(200, 100, 200)";
menuAltColors[WELL] = "rgb(158, 13, 33)";
menuAltColors[SOIL] = "rgb(171, 110, 53)";

export function initMenu(): void {
  /* The wrapper div that holds the entire menu */
  document.getElementById("menuWrapper");

  /* Set up the wrapper div that holds the element selectors */
  const elementMenu = document.getElementById(
    "elementMenu"
  ) as HTMLDivElement;
  if (!elementMenu) throw new Error("elementMenu not found");
  elementMenu.style.width =
    "50%"; /* force browser to scrunch the element menu */
  const numRows = Math.ceil(
    elementMenuItems.length / ELEMENT_MENU_ELEMENTS_PER_ROW
  );
  let elemIdx = 0;
  let i: number, k: number;
  for (i = 0; i < numRows; i++) {
    for (k = 0; k < ELEMENT_MENU_ELEMENTS_PER_ROW; k++) {
      if (elemIdx >= elementMenuItems.length) break;

      const elemButton = document.createElement("input");
      elementMenu.appendChild(elemButton);

      elemButton.type = "button";
      elemButton.className = "elementMenuButton";

      const elemType = elementMenuItems[elemIdx];
      if (!(elemType in menuNames))
        throw new Error("element is missing a canonical name: " + elemType);
      elemButton.value = menuNames[elemType];

      const elemColorRGBA = elemType;
      elemButton.id = elemColorRGBA.toString();

      let elemMenuColor: string;
      if (elemType in menuAltColors) elemMenuColor = menuAltColors[elemType];
      else
        elemMenuColor =
          "rgb(" +
          (elemColorRGBA & 0xff) +
          ", " +
          ((elemColorRGBA & 0xff00) >>> 8) +
          ", " +
          ((elemColorRGBA & 0xff0000) >>> 16) +
          ")";
      elemButton.style.color = elemMenuColor;

      elemButton.addEventListener("click", function () {
        const selectedElem = document.getElementById(
          getSELECTED_ELEM().toString()
        ) as HTMLInputElement;
        if (selectedElem) {
          selectedElem.classList.remove("selectedElementMenuButton");
        }
        elemButton.classList.add("selectedElementMenuButton");
        setSELECTED_ELEM(parseInt(elemButton.id, 10));
      });

      elemIdx++;
    }
  }
  const defaultButton = document.getElementById(
    getSELECTED_ELEM().toString()
  ) as HTMLInputElement;
  if (defaultButton) {
    defaultButton.click();
  }

  /* Set up pensize options */
  const pensizes = document.getElementById("pensize") as HTMLSelectElement;
  if (!pensizes) throw new Error("pensize not found");
  for (i = 0; i < PEN_SIZES.length; i++) {
    const p = document.createElement("option");
    p.value = PEN_SIZES[i].toString();
    p.text = PEN_SIZE_LABELS[i];
    if (i === DEFAULT_PEN_IDX) {
      p.selected = true;
      setPENSIZE(parseInt(p.value, 10));
    }
    pensizes.add(p);
  }
  pensizes.addEventListener("change", function () {
    setPENSIZE(parseInt(pensizes.value, 10));
  });

  /* Set up spigot size options */
  const spigotTypes: HTMLSelectElement[] = [
    document.getElementById("spigot1Type") as HTMLSelectElement,
    document.getElementById("spigot2Type") as HTMLSelectElement,
    document.getElementById("spigot3Type") as HTMLSelectElement,
    document.getElementById("spigot4Type") as HTMLSelectElement,
  ];
  const spigotSizes: HTMLSelectElement[] = [
    document.getElementById("spigot1Size") as HTMLSelectElement,
    document.getElementById("spigot2Size") as HTMLSelectElement,
    document.getElementById("spigot3Size") as HTMLSelectElement,
    document.getElementById("spigot4Size") as HTMLSelectElement,
  ];
  if (spigotTypes.length !== spigotSizes.length)
    throw new Error("should be same length");
  for (i = 0; i < spigotTypes.length; i++) {
    const typeSelector = spigotTypes[i];
    const sizeSelector = spigotSizes[i];
    if (!typeSelector || !sizeSelector) continue;
    for (k = 0; k < SPIGOT_ELEMENT_OPTIONS.length; k++) {
      const type = SPIGOT_ELEMENT_OPTIONS[k];
      const option = document.createElement("option");
      option.value = type.toString();
      option.text = menuNames[type];
      if (i === k) {
        option.selected = true;
        SPIGOT_ELEMENTS[i] = type;
      }
      typeSelector.add(option);
    }
    for (k = 0; k < SPIGOT_SIZE_OPTIONS.length; k++) {
      const size = SPIGOT_SIZE_OPTIONS[k];
      const option = document.createElement("option");
      option.value = size.toString();
      option.text = k.toString(10);
      if (k === DEFAULT_SPIGOT_SIZE_IDX) {
        option.selected = true;
        SPIGOT_SIZES[i] = size;
      }
      sizeSelector.add(option);
    }
  }
  spigotTypes[0]?.addEventListener("change", function () {
    SPIGOT_ELEMENTS[0] = parseInt(spigotTypes[0].value, 10);
  });
  spigotTypes[1]?.addEventListener("change", function () {
    SPIGOT_ELEMENTS[1] = parseInt(spigotTypes[1].value, 10);
  });
  spigotTypes[2]?.addEventListener("change", function () {
    SPIGOT_ELEMENTS[2] = parseInt(spigotTypes[2].value, 10);
  });
  spigotTypes[3]?.addEventListener("change", function () {
    SPIGOT_ELEMENTS[3] = parseInt(spigotTypes[3].value, 10);
  });
  spigotSizes[0]?.addEventListener("change", function () {
    SPIGOT_SIZES[0] = parseInt(spigotSizes[0].value, 10);
  });
  spigotSizes[1]?.addEventListener("change", function () {
    SPIGOT_SIZES[1] = parseInt(spigotSizes[1].value, 10);
  });
  spigotSizes[2]?.addEventListener("change", function () {
    SPIGOT_SIZES[2] = parseInt(spigotSizes[2].value, 10);
  });
  spigotSizes[3]?.addEventListener("change", function () {
    SPIGOT_SIZES[3] = parseInt(spigotSizes[3].value, 10);
  });

  /* 'overwrite' checkbox */
  const overwriteCheckbox = document.getElementById(
    "overwriteCheckbox"
  ) as HTMLInputElement;
  if (overwriteCheckbox) {
    overwriteCheckbox.checked = getOVERWRITE_ENABLED();
    overwriteCheckbox.addEventListener("click", function () {
      setOVERWRITE_ENABLED(overwriteCheckbox.checked);
    });
  }

  /* 'void mode' checkbox */
  const voidModeCheckbox = document.getElementById(
    "voidModeCheckbox"
  ) as HTMLInputElement;
  if (voidModeCheckbox) {
    voidModeCheckbox.checked = getVOID_MODE_ENABLED();
    voidModeCheckbox.addEventListener("click", function () {
      setVOID_MODE_ENABLED(voidModeCheckbox.checked);
    });
  }

  /* speed slider */
  const speedSlider = document.getElementById(
    "speedSlider"
  ) as HTMLInputElement;
  if (speedSlider) {
    speedSlider.min = "0";
    speedSlider.max = MAX_FPS.toString();
    speedSlider.value = DEFAULT_FPS.toString();
    speedSlider.addEventListener("input", function () {
      const val = parseInt(speedSlider.value, 10);
      /* make 'magnetic' towards the default */
      if (Math.abs(val - DEFAULT_FPS) < 10)
        speedSlider.value = DEFAULT_FPS.toString();
      setFPS(parseInt(speedSlider.value, 10));
    });
  }

  /* zombie slider */
  const zombieSlider = document.getElementById(
    "zombieSlider"
  ) as HTMLInputElement;
  if (zombieSlider) {
    zombieSlider.min = "0";
    zombieSlider.max = MAX_ZOMBIES.toString();
    zombieSlider.value = "0";
    zombieSlider.addEventListener("input", function () {
      setZombieCount(parseInt(zombieSlider.value, 10));
    });
  }

  /* clear button */
  const clearButton = document.getElementById(
    "clearButton"
  ) as HTMLInputElement;
  if (clearButton) {
    clearButton.onclick = clearGameCanvas;
  }

  /* save button */
  const saveButton = document.getElementById("saveButton") as HTMLInputElement;
  if (saveButton) {
    saveButton.onclick = saveGameCanvas;
  }

  /* load button */
  const loadButton = document.getElementById("loadButton") as HTMLInputElement;
  if (loadButton) {
    loadButton.onclick = loadGameCanvas;
  }
}

export function drawFPSLabel(fps: number): void {
  const fpsCounter = document.getElementById("fps-counter");
  if (fpsCounter) {
    fpsCounter.innerText = "FPS: " + fps;
  }
}

export function drawZombieCount(val: number): void {
  const zombieCount = document.getElementById("zombieCount");
  if (zombieCount) {
    zombieCount.innerText = val.toString();
  }
}
