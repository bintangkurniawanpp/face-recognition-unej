import splitfolders  # or import split_folders

# Split dengan ratio.
# Untuk hanya membagi menjadi set pelatihan dan validasi, setel tuple menjadi `ratio`, i.e, `(.8, .2)`.
splitfolders.ratio("dataset", output="output", seed=1337, ratio=(.8,.2), group_prefix=None)