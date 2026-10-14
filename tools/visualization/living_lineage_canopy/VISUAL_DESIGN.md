# Living Lineage Canopy V0 Visual Design Specification

## Visual Mood

The prototype should read as living microscopy: suspended particles in liquid,
soft biological colonies, and slow organic drift. The image must be calm,
restrained, readable at a glance, and presentation-ready. It should not look
like a debug canvas, arcade particle effect, or ecological world map.

## Background

Use a deep ink-to-charcoal field with a subtle radial vignette and very fine
liquid-grain texture. The background stays quiet and non-directional so lineage
canopies carry the composition. Avoid neon gradients, bokeh blobs, hard grids,
terrain cues, or anything that suggests Patch World coordinates.

## Particle Appearance

Particles are small soft cells with:

- semi-opaque lineage color fill;
- a faint lighter core;
- low-strength edge bloom;
- radius easing during birth and death;
- no hard outline in normal state.

Particles should feel suspended in fluid. They must not flicker, snap, bounce,
or leave noisy permanent trails.

## Lineage Color System

Founder lineages use separated, moderately saturated hues distributed around
the color wheel with luminance held in a narrow readable band. Descendant
lineages and branches inherit related colors through small deterministic hue
and lightness shifts. Dominance must not change color identity.

Selected lineages are emphasized by opacity, size, and a subtle halo, not by
recoloring. Isolated mode dims unrelated lineages toward translucent gray-blue.

## Branch Differentiation

Branches appear as soft lobes through stronger branch attraction and slightly
shifted colors. V0 may add a very faint density wash behind active branch
particles only when it improves readability. Do not draw rigid hulls, boxes, or
tree edges by default.

## Typography

Use a modern sans-serif stack with compact numeric panels. Type is small,
crisp, and secondary to the visualization. Labels are sparse. Use tabular
figures for metrics. Avoid oversized headings and explanatory text inside the
app surface.

## Spacing

The canvas owns most of the viewport. Controls sit in a restrained side rail on
wide screens and a compact bottom sheet on narrow screens. Panels should have
consistent 8px radii, thin translucent borders, and enough internal spacing for
touch targets without competing with the canopy.

## Control Layout

Controls are grouped by workflow:

- transport: play/pause, reset, jump to end, scrubber, speed;
- scenario gallery: direct scenario cards with description, observation cue,
  source status, semantics, peak population, lineage count, and duration;
- Showcase mode: automatic scenario sequencing with pause, resume, skip,
  restart, and loop;
- view: particle size, motion, branch attraction, lineage attraction;
- selection: selected lineage, compact ranked lineage list, isolate, show all;
- metrics: current tick, living count, lineage count, branch count, largest
  share, FPS and timing.

Buttons use icons or short text only. Default browser controls must be styled.

## Animation Timing And Easing

Use slow easing:

- birth pulse: 420-650 ms ease-out;
- death fade: 600-900 ms ease-in-out;
- selection halo: 180-260 ms ease-out;
- scrub rebuild fade-in: 140-220 ms;
- control transitions: 120-180 ms.

Avoid sudden opacity jumps, instantaneous canopy relocation, and linear harsh
motion.

## Motion Speed

Default motion is deliberately slow. Settled canopies should continue to breathe
slightly but not jitter. Use damping, bounded force, low drift, and capped
per-step velocity. Normal playback should be readable at 1x; higher speeds may
increase lifecycle progression but should not make particle physics frantic.

## Selection And Isolation

Clicking a particle selects the agent and its lineage. Selected particles get a
soft halo and a small radius lift. The selected lineage remains vivid;
unselected lineages dim but remain spatially coherent. Isolated mode may hide or
strongly dim unrelated particles while preserving the selected lineage layout.
Selection never changes the data or color identity.

## Clutter Limits

V0 should avoid persistent labels on particles and avoid drawing parent-child
edges by default. At high density, prioritize aggregate visual readability over
individual inspection. Metrics remain compact; branch/lineage labels appear only
in panels or hover/selection context. The 5,000-particle stress view must remain
composed, not saturated with text or glow.

## Visual Acceptance Loop

Before LLC-V0-A, run every synthetic scenario in a browser, capture the required
screenshots, inspect composition and motion, identify the three largest visual
defects, repair them, and repeat until no major visual defect remains. A merely
functional renderer is LLC-V0-B until the visual evidence is presentation-ready.
