# Forearm Atlas

Live site: https://eligoumas.github.io/forearm-atlas/ (GitHub Pages, rebuilds 1–2 minutes after a push to `main`)
Repo: https://github.com/eligoumas/forearm-atlas
Working copy on Eli's PC: `C:\Users\elgou\OneDrive\Documents\Cool Project\forearm-atlas` (this folder)

## What it is

One page, `index.html`. The webcam tracks the hand with MediaPipe; a 3D forearm and hand (three.js) is fitted over the arm
on the camera view and also shown in a side panel you can turn. Muscles that are working glow electric blue. Without a camera
a demo hand runs. The camera only works when the page is opened directly over https or localhost, not embedded.

- `index.html`: the whole app (HTML, CSS, one module script). three.js and MediaPipe load from cdn.jsdelivr.net.
- `models/forearm.glb`: the 3D anatomy (187 named parts plus a rig). Do not edit by hand.
- `blender/build_forearm.py`: builds that model in Blender and exports it to `models/forearm.glb`. Run instructions are at the top of the file.

## Rules

- This repo is the only copy. The old version with a 2D "Section A–A" cross-section panel is retired; never restore it or
  rebuild from an old copy. If `index.html` mentions `SECTIONS` or a cross-section, something has gone wrong.
- After changing anything, test it locally (`python -m http.server 8766 --directory .` here, then open http://127.0.0.1:8766),
  then commit and push to `main` so the live site updates. Uncommitted edits never reach the link.
- Object names in the model are what the app reads: `M_<muscle key>`, `T_<tendon key>`, `B_` bones, `N_`/`A_`/`V_` nerves,
  arteries and veins. Muscle keys match the activity keys in `index.html` (`FDS_i`, `ED_p`, `DI1`, ...); keep them in step.
- Pushing needs write access to the `eligoumas` GitHub account. Anyone can clone and view.
