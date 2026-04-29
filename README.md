# IT5437 Computer Vision — Assignment 02: Fitting and Alignment

**Weerarathna T.M.P. | Index: 258843A**

## Setup

```bash
uv sync
```

## Running

```bash
# Q1 - Line fitting (TLS and RANSAC)
uv run python src/q1_line_fitting.py

# Q2 - Earring size estimation
uv run python src/q2_earring_size.py

# Q3 - Circuit board homography (manual + SIFT)
uv run python src/q3_circuit_board.py

# Q3 - Interactive point collection tool (Listing 1)
uv run python src/listing1_click_points.py
```

## Output

Results are saved to the `output/` folder:

| Folder | Files |
|--------|-------|
| `output/q1/` | `q1a_tls_line1.png`, `q1b_ransac_3lines.png`, `q1_terminal_output.png` |
| `output/q2/` | `q2_earring_size.png`, `q2_terminal_output.png` |
| `output/q3/` | `q3a_warped_manual.png`, `q3b_diff_manual.png`, `q3c_sift_matches.png`, `q3d_warped_sift.png`, `q3d_diff_sift.png`, `grid_view.png`, `point_zoom.png`, `q3_terminal_output.png` |

## Structure

```
cv-assignment02/
├── src/
│   ├── q1_line_fitting.py        # Q1a TLS + Q1b RANSAC
│   ├── q2_earring_size.py        # Q2 lens formula + earring measurement
│   ├── q3_circuit_board.py       # Q3a-d homography and SIFT
│   └── listing1_click_points.py  # Listing 1 interactive point selection tool
├── a2_images/                    # Input images (c1.jpg, c2.jpg, earrings.jpg)
├── lines.csv                     # Line fitting dataset
└── output/                       # All generated results
```
