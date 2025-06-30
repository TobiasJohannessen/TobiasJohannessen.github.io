
set terminal pngcairo size 300,800 enhanced font 'Arial,10'
set output 'plots/compare_grids.png'

set datafile separator "\t"
set palette rgbformulae 33,13,10

set multiplot layout 2,1 title "Heatmap" margins 0.1,0.8,0.1,0.8 spacing 0.1,0.1

# === Plot 1: Full heatmap (top) ===
set title "2D Heatmap (Bilinear Interpolation)"
set xlabel "x"
set ylabel "y"
unset ztics
set view map
set colorbox
unset key # Or set key bottom right etc, if you want only the 'Original Points' legend

splot 'data/bilinear_diffs.txt' using 1:2:3 with points palette pointtype 7 pointsize 1 notitle, \
    'data/grid.txt' using 1:2:3 with points pt 7 ps 1.2 lc rgb "black" notitle, \
    'data/grid.txt' using 1:2:3 with points pt 7 ps 1 palette title 'Original Points'


# === Plot 2: Full heatmap (top) ===
set title "2D Heatmap (Bicubic Interpolation)"
set xlabel "x"
set ylabel "y"
unset ztics
set view map
set colorbox
unset key # Or set key bottom right etc, if you want only the 'Original Points' legend

splot 'data/bicubic_diffs.txt' using 1:2:3 with points palette pointtype 7 pointsize 1 notitle, \
    'data/grid.txt' using 1:2:3 with points pt 7 ps 1.2 lc rgb "black" notitle, \
    'data/grid.txt' using 1:2:3 with points pt 7 ps 1 palette title 'Original Points'
unset multiplot