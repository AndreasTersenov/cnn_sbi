# BNT l1 auto / +product — compressor-ENSEMBLE calibration

Single BNT arms over-confident (auto sup 0.077/0.077/0.068, +product 0.054/0.072/0.062;
simultaneous band d99=0.066). Fix = 3-compressor deep ensemble (seeds 41/42/43), the joint-l1 lever.

| arm | variant | SBC std (Om/s8/w0) | sup\|F-r\| (Om/s8/w0) | inside band? |
|---|---|---|---|---|
| product | single (s41) | 0.311/0.313/0.301 | 0.054/0.072/0.062 | NO (max 0.072) |
| product | **ensemble x3** | 0.304/0.302/0.297 | 0.049/0.040/0.055 | yes |
| auto | single (s41) | 0.320/0.319/0.299 | 0.077/0.077/0.068 | NO (max 0.077) |
| auto | **ensemble x3** | 0.303/0.297/0.296 | 0.052/0.036/0.065 | yes |
