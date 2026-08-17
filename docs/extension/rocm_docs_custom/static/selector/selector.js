import { domReady, logDebug } from "./utils.js";
import {
  updateTOC2ContentsList,
  updateTOC2OptionsList,
} from "./selector-toc.js";

const GROUP_QUERY = ".rocm-docs-selector-group";
const OPTION_QUERY = ".rocm-docs-selector-option";
const COND_QUERY = "[data-show-when],[data-disable-when]";

const DEFAULT_OPTION_CLASS = "rocm-docs-selector-option-default";
const DISABLED_CLASS = "rocm-docs-disabled";
const HIDDEN_CLASS = "rocm-docs-hidden";
const SELECTED_CLASS = "rocm-docs-selected";

// Toggle helpers -------------------------------------------------------------

const isDefaultOption = (elem) => elem.classList.contains(DEFAULT_OPTION_CLASS);

const disable = (elem) => {
  elem.classList.add(DISABLED_CLASS);
  elem.setAttribute("aria-disabled", "true");
  elem.setAttribute("tabindex", "-1");
};

const enable = (elem) => {
  elem.classList.remove(DISABLED_CLASS);
  elem.setAttribute("aria-disabled", "false");
  elem.setAttribute("tabindex", "0");
};

const hide = (elem) => {
  elem.classList.add(HIDDEN_CLASS);
  elem.setAttribute("aria-hidden", "true");
};

const show = (elem) => {
  elem.classList.remove(HIDDEN_CLASS);
  elem.setAttribute("aria-hidden", "false");
};

const select = (elem) => {
  elem.classList.add(SELECTED_CLASS);
  elem.setAttribute("aria-checked", "true");
};

const deselect = (elem) => {
  elem.classList.remove(SELECTED_CLASS);
  elem.setAttribute("aria-checked", "false");
};

// Global selector state ------------------------------------------------------

const state = {};

function getState() {
  return { ...state };
}

function setState(updates) {
  Object.assign(state, updates);
  logDebug("State updated:", state);
}

// Condition handling ---------------------------------------------------------

/**
 * Safely parse JSON-encoded conditions from a data-* attribute.
 * Expects a key/value object, where values may be strings or arrays of strings.
 */
function parseConditions(attrName, raw) {
  if (!raw) return null;

  try {
    const conditions = JSON.parse(raw);
    if (typeof conditions !== "object" || Array.isArray(conditions)) {
      console.warn(
        `[ROCmDocsSelector] Invalid '${attrName}' format ` +
          "(must be a key/value object):",
        raw,
      );
      return null;
    }
    return conditions;
  } catch (err) {
    console.error(
      `[ROCmDocsSelector] Couldn't parse '${attrName}' conditions:`,
      err,
    );
    return null;
  }
}

/**
 * Return true iff all conditions match the current state.
 * - Values can be a string or an array of strings.
 * - A condition with an undefined state key is treated as not matching.
 */
function matchesConditions(conditions, currentState) {
  for (const [key, expected] of Object.entries(conditions)) {
    const actual = currentState[key];

    // If no value yet, this condition does not match.
    if (actual === undefined) return false;

    if (Array.isArray(expected)) {
      if (!expected.includes(actual)) return false;
    } else if (actual !== expected) {
      return false;
    }
  }
  return true;
}

function shouldBeDisabled(elem) {
  const raw = elem.dataset.disableWhen;
  if (!raw) return false; // no conditions => never disabled

  const conditions = parseConditions("disable-when", raw);
  if (!conditions) {
    console.warn(
      "[ROCmDocsSelector] Invalid 'show-when' conditions; " +
        "hiding affected element.",
    );
    return false;
  }

  return matchesConditions(conditions, state);
}

function shouldBeShown(elem) {
  const raw = elem.dataset.showWhen;
  if (!raw) return true; // no conditions => always visible

  const conditions = parseConditions("show-when", raw);
  if (!conditions) return true;

  // When type=all is selected, treat it as a wildcard over type so that every
  // content block for the current level is shown without requiring an explicit
  // "level=X type=all" authored block:
  //   level=all + type=all → show only the master catch-all block
  //   level=X   + type=all → show every block whose level condition is X
  //
  // Note: condition values are encoded as arrays by the Sphinx extension,
  // e.g. {"level": ["beginner"], "type": ["inference"]}, so index into [0].
  if (state.type === "all" && "type" in conditions) {
    const condLevel = Array.isArray(conditions.level)
      ? conditions.level[0]
      : conditions.level;
    const condType = Array.isArray(conditions.type)
      ? conditions.type[0]
      : conditions.type;
    if (state.level === "all") {
      return condLevel === "all" && condType === "all";
    }
    return condLevel === state.level;
  }

  return matchesConditions(conditions, state);
}

// Event handlers -------------------------------------------------------------

function handleOptionSelect(e) {
  const option = e.currentTarget;

  // Ignore interaction with disabled or already selected options
  if (
    option.classList.contains(DISABLED_CLASS) ||
    option.classList.contains(SELECTED_CLASS)
  ) {
    return;
  }

  const { selectorKey: key, selectorValue: value } = option.dataset;
  if (!key || !value) return;

  // Update all selectors sharing the same key
  const allOptions = document.querySelectorAll(
    `${OPTION_QUERY}[data-selector-key="${key}"]`,
  );

  allOptions.forEach((opt) => {
    if (opt.dataset.selectorValue === value) {
      select(opt);
    } else {
      deselect(opt);
    }
  });

  // Update global state
  setState({ [key]: value });

  // Re-run visibility rules and TOC sync
  updateVisibility();
}

function handleOptionKeydown(e) {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    handleOptionSelect(e);
  }
}

// Visibility / enablement update --------------------------------------------

// Ensure each selector group always has a valid selected option.
// If the current selection becomes disabled/hidden due to another selector's
// change, automatically pick a replacement.
function reconcileGroupSelections() {
  const currentState = getState();
  const updates = {};

  document.querySelectorAll(GROUP_QUERY).forEach((group) => {
    // Skip groups that are themselves hidden
    if (group.classList.contains(HIDDEN_CLASS)) return;

    const options = Array.from(group.querySelectorAll(OPTION_QUERY));
    if (!options.length) return;

    const groupKey = group.dataset.selectorKey ||
      options[0].dataset.selectorKey;
    if (!groupKey) return;

    // Options that are both enabled and visible
    const enabledVisible = options.filter(
      (opt) =>
        !opt.classList.contains(DISABLED_CLASS) &&
        !opt.classList.contains(HIDDEN_CLASS),
    );

    if (!enabledVisible.length) {
      // No valid options left; just clear visual selection.
      options.forEach(deselect);
      return;
    }

    const currentlySelected = options.find((opt) =>
      opt.classList.contains(SELECTED_CLASS)
    );

    const selectedStillValid = currentlySelected &&
      enabledVisible.includes(currentlySelected);

    if (selectedStillValid) {
      const selectedValue = currentlySelected.dataset.selectorValue;
      if (selectedValue && currentState[groupKey] !== selectedValue) {
        updates[groupKey] = selectedValue;
      }
      return;
    }

    // Need a new selection: prefer a default option, otherwise the first
    // enabled+visible option in DOM order.
    const replacement = enabledVisible.find(isDefaultOption) ||
      enabledVisible[0];
    if (!replacement) return;

    options.forEach(deselect);
    select(replacement);

    const newValue = replacement.dataset.selectorValue;
    if (newValue && currentState[groupKey] !== newValue) {
      updates[groupKey] = newValue;
    }
  });

  const changedKeys = Object.keys(updates);
  if (changedKeys.length > 0) {
    setState(updates);
    return true;
  }
  return false;
}

let isUpdatingVisibility = false;

function updateVisibility() {
  // Prevent re-entrancy if something triggers updateVisibility
  // while it is already running.
  if (isUpdatingVisibility) return;
  isUpdatingVisibility = true;

  try {
    let stateChanged = false;
    let iterations = 0;

    // We may need multiple passes: reconciling selections can change the
    // global state, which in turn affects show/disable conditions.
    do {
      document.querySelectorAll(COND_QUERY).forEach((elem) => {
        // Show/hide only if element has show-when
        if (elem.dataset.showWhen !== undefined) {
          if (shouldBeShown(elem)) {
            show(elem);
          } else {
            hide(elem);
          }
        }

        // Enable/disable only if element has disable-when
        if (elem.dataset.disableWhen !== undefined) {
          if (shouldBeDisabled(elem)) {
            disable(elem);
          } else {
            enable(elem);
          }
        }
      });

      stateChanged = reconcileGroupSelections();
      iterations += 1;
      // Hard stop to avoid infinite loops in case of conflicting rules.
    } while (stateChanged && iterations < 5);

    updateTOC2OptionsList();
    updateTOC2ContentsList();

    // Show type-category headings only when type=all + a specific level is
    // active. Each heading is shown iff its immediately following content block
    // is visible (some level+type combos have no tutorials).
    const showTypeHeadings = state.type === "all" && state.level !== "all";
    document.querySelectorAll(".rocm-docs-selector-type-heading").forEach((heading) => {
      if (!showTypeHeadings) {
        hide(heading);
        return;
      }
      const contentBlock = heading.nextElementSibling;
      if (contentBlock && !contentBlock.classList.contains(HIDDEN_CLASS)) {
        show(heading);
      } else {
        hide(heading);
      }
    });

    // Show "no results" message when no content block is visible
    const anyVisible = Array.from(
      document.querySelectorAll(".rocm-docs-selected-content")
    ).some((el) => !el.classList.contains(HIDDEN_CLASS));
    const noResults = document.getElementById("rocm-docs-selector-no-results");
    if (noResults) {
      anyVisible ? hide(noResults) : show(noResults);
    }
  } finally {
    isUpdatingVisibility = false;
  }
}

// Initialization -------------------------------------------------------------

domReady(() => {
  const selectorOptions = document.querySelectorAll(OPTION_QUERY);
  const initialState = {};

  // Attach listeners and gather defaults
  selectorOptions.forEach((option) => {
    option.addEventListener("click", handleOptionSelect);
    option.addEventListener("keydown", handleOptionKeydown);

    if (isDefaultOption(option)) {
      select(option);
      const { selectorKey: key, selectorValue: value } = option.dataset;
      if (key && value && initialState[key] === undefined) {
        initialState[key] = value;
      }
    }
  });

  // Inject no-results message before the first content block
  const firstContent = document.querySelector(".rocm-docs-selected-content");
  if (firstContent) {
    const noResults = document.createElement("p");
    noResults.id = "rocm-docs-selector-no-results";
    noResults.className = HIDDEN_CLASS;
    noResults.setAttribute("aria-hidden", "true");
    noResults.textContent = "No tutorials match the selected filters.";
    firstContent.parentNode.insertBefore(noResults, firstContent);
  }

  // Dynamically grey out selector buttons whose combinations have no content.
  // Scan content blocks to learn which (key=value) combos actually exist, then
  // attach data-disable-when to buttons whose value never appears alongside
  // each value of every other selector key.
  (function applyEmptyComboDisabling() {
    const contentBlocks = document.querySelectorAll(
      ".rocm-docs-selected-content[data-show-when]",
    );

    // Collect all selector keys present in the page (e.g. "level", "type").
    const allKeys = new Set();
    document.querySelectorAll(OPTION_QUERY).forEach((opt) => {
      if (opt.dataset.selectorKey) allKeys.add(opt.dataset.selectorKey);
    });

    // Build a set of existing combos as "key1=val1|key2=val2" strings.
    const existingCombos = new Set();
    contentBlocks.forEach((block) => {
      const conditions = parseConditions("show-when", block.dataset.showWhen);
      if (!conditions) return;
      // Only record combos that specify every known selector key (i.e. fully
      // qualified blocks, not "all" catch-alls that use the wildcard value).
      const keys = Object.keys(conditions);
      if (keys.length === allKeys.size) {
        // Condition values are arrays (e.g. ["advanced"]); extract the first element.
        const combo = keys.sort()
          .map((k) => {
            const v = conditions[k];
            return `${k}=${Array.isArray(v) ? v[0] : v}`;
          })
          .join("|");
        existingCombos.add(combo);
      }
    });

    // For each option button, collect the values of all *other* selector keys
    // and check whether any existing combo pairs this button's value with each
    // of those peer values. If no combo exists for a particular peer value,
    // attach a data-disable-when rule for that peer state.
    //
    // Exceptions: never grey out knowledge-level buttons (they set context, not
    // content) and never grey out any "all" option (it always means "show
    // whatever exists for the current state").
    document.querySelectorAll(OPTION_QUERY).forEach((opt) => {
      const myKey = opt.dataset.selectorKey;
      const myVal = opt.dataset.selectorValue;
      if (!myKey || !myVal) return;
      if (myKey === "level" || myVal === "all") return;

      // Collect all (key, value) pairs for every other selector group.
      const peerGroups = {};
      document.querySelectorAll(OPTION_QUERY).forEach((peer) => {
        const pk = peer.dataset.selectorKey;
        const pv = peer.dataset.selectorValue;
        if (!pk || !pv || pk === myKey) return;
        if (!peerGroups[pk]) peerGroups[pk] = new Set();
        peerGroups[pk].add(pv);
      });

      // For each peer key, find which peer values yield no valid combo.
      const disableConditions = [];
      for (const [peerKey, peerVals] of Object.entries(peerGroups)) {
        for (const peerVal of peerVals) {
          // Build the canonical combo string for (myKey=myVal, peerKey=peerVal).
          const pair = [
            `${myKey}=${myVal}`,
            `${peerKey}=${peerVal}`,
          ].sort().join("|");
          if (!existingCombos.has(pair)) {
            disableConditions.push({ [peerKey]: peerVal });
          }
        }
      }

      if (!disableConditions.length) return;

      // Encode each disable condition as its own data-disable-when-* attribute
      // so the existing shouldBeDisabled logic (which checks a single object)
      // can OR across them. We achieve OR by adding multiple attributes and
      // overriding shouldBeDisabled to check any of them — but since the
      // existing engine only reads data-disable-when (singular), we store all
      // disabling peer-values for each peer key as an array in one object.
      //
      // Structure: { peerKey: [val1, val2, ...] } — matchesConditions already
      // supports array values, so a single data-disable-when covers all cases.
      const merged = {};
      for (const cond of disableConditions) {
        for (const [k, v] of Object.entries(cond)) {
          if (!merged[k]) merged[k] = [];
          merged[k].push(v);
        }
      }
      opt.dataset.disableWhen = JSON.stringify(merged);
      logDebug("applyEmptyComboDisabling:", myKey, myVal, "->", merged);
    });
  })();

  // Inject type-category headings before each type-specific content block so
  // they can be shown when type=all + a specific level is active. Heading text
  // is sourced from the catch-all (level=all type=all) block so it always
  // matches the authoritative RST wording.
  (function injectTypeHeadings() {
    // Find the catch-all block and map type value → heading text from its
    // <strong> elements. Match each heading to a type value by checking whether
    // the button's value slug appears in the heading text (case-insensitive,
    // dashes treated as spaces so "fine-tuning" matches "Fine-tuning tutorials").
    const typeValues = [];
    document.querySelectorAll(`${OPTION_QUERY}[data-selector-key="type"]`).forEach((opt) => {
      const v = opt.dataset.selectorValue;
      if (v && v !== "all") typeValues.push(v);
    });

    const typeHeadingText = {};
    const catchAllBlock = Array.from(
      document.querySelectorAll(".rocm-docs-selected-content[data-show-when]"),
    ).find((block) => {
      const cond = parseConditions("show-when", block.dataset.showWhen);
      if (!cond) return false;
      const lv = Array.isArray(cond.level) ? cond.level[0] : cond.level;
      const tv = Array.isArray(cond.type) ? cond.type[0] : cond.type;
      return lv === "all" && tv === "all";
    });

    if (catchAllBlock) {
      catchAllBlock.querySelectorAll("strong").forEach((strong) => {
        const text = strong.textContent.trim();
        const normalized = text.toLowerCase().replace(/-/g, " ");
        for (const typeVal of typeValues) {
          const slug = typeVal.toLowerCase().replace(/-/g, " ");
          // Match if the whole slug is a substring, or if any individual word
          // of the slug (length > 2) appears in the heading — needed for slugs
          // like "gpu-dev-opt" whose words expand differently in full headings.
          const slugWords = slug.split(/\s+/).filter((w) => w.length > 2);
          if (normalized.includes(slug) || slugWords.some((w) => normalized.includes(w))) {
            typeHeadingText[typeVal] = text;
            break;
          }
        }
      });
    }

    // Inject a hidden heading paragraph immediately before each type-specific
    // (non-all level, non-all type) content block.
    document.querySelectorAll(".rocm-docs-selected-content[data-show-when]").forEach((block) => {
      const cond = parseConditions("show-when", block.dataset.showWhen);
      if (!cond) return;
      const levelVal = Array.isArray(cond.level) ? cond.level[0] : cond.level;
      const typeVal = Array.isArray(cond.type) ? cond.type[0] : cond.type;
      if (levelVal === "all" || typeVal === "all" || !typeHeadingText[typeVal]) return;

      const heading = document.createElement("p");
      heading.className = `rocm-docs-selector-type-heading ${HIDDEN_CLASS}`;
      heading.setAttribute("aria-hidden", "true");
      heading.dataset.headingForType = typeVal;
      const strong = document.createElement("strong");
      strong.textContent = typeHeadingText[typeVal];
      heading.appendChild(strong);
      block.parentNode.insertBefore(heading, block);
    });
  })();

  setState(initialState);
  updateVisibility();

  // Mark all selector groups as initialized to make them visible
  document.querySelectorAll(GROUP_QUERY).forEach((group) => {
    group.classList.add("rocm-docs-selector-initialized");
  });
});
