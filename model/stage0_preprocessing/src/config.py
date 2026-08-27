stage0:
  quality:
    blur_threshold: 100
    skew_threshold_degrees: 2.0
    contour_area_ratio: 0.90
  clahe:
    clip_limit: 2.0
    tile_grid_size: 8
  sharpen:
    kernel_size: 3
    strength: 1.5
  perspective:
    min_contour_area: 1000
    min_area_ratio: 0.50
    output_size: 1024
  pdf:
    render_dpi: 300
    max_pages: 10
