# Nine-case minimal visualization

Open `index.html` to view nine cases: three each for L1, L2, and L3. Every case is fully detected, passes the existing evaluation, and was manually checked for useful depth visualization.

Each case contains only:

- `image.png`: generated image;
- `bbox.jpg`: union boxes (blue OWL-ViT, orange SAM3.1);
- `sam.jpg`: SAM3.1 fallback boxes only, present only when fallback was triggered;
- `depth.png`: DA3 metric depth visualization;
- `info.json`: bbox coordinates, detector source, scores, and object depth statistics.
