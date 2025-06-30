stats 'data/constant_x.txt' using 1 nooutput
const_x = STATS_min

stats 'data/constant_y.txt' using 2 nooutput
const_y = STATS_min

set terminal pngcairo size 300,800 enhanced font 'Arial,10'
set output 'plots/interpolated_slices.png'

set datafile separator "\t"
set palette rgbformulae 33,13,10

set multiplot layout 3,1 title "Heatmap and Line Slices" margins 0.1,0.8,0.1,0.8 spacing 0.1,0.1

# === Plot 1: Full heatmap (top) ===
set title "2D Heatmap (Bilinear Interpolation)"
set xlabel "x"
set ylabel "y"
unset ztics
set view map
set colorbox
unset key # Or set key bottom right etc, if you want only the 'Original Points' legend

splot 'data/interpolated_grid.txt' using 1:2:3 with points palette pointtype 7 pointsize 1 notitle, \
      'data/grid.txt' using 1:2:3 with points pt 7 ps 1.2 lc rgb "black" notitle, \
      'data/grid.txt' using 1:2:3 with points pt 7 ps 1 palette title 'Original Points'

# === Plot 2: z vs y at constant x ===
set title sprintf("Slice at x = %.2g", const_x)
set xlabel "y"
set ylabel "z"
unset colorbox
unset view
unset xtics # Keep these or comment them out based on whether you want ticks
unset ytics # Keep these or comment them out based on whether you want ticks
set tics out nomirror
set autoscale x
set autoscale y
# Plot interpolated slice (red points)
plot 'data/interpolated_constant_x.txt' using 2:3 with points pt 7 lc rgb "red" title 'Interpolated', \
     'data/constant_x.txt' using 2:3 with points pt 7 ps 1.2 lc rgb "black" notitle, \
     'data/constant_x.txt' using 2:3 with points pt 7 ps 1 lc rgb "blue" title 'Original Points'

# === Plot 3: z vs x at constant y ===
set title sprintf("Slice at y = %.2g", const_y)
set xlabel "x"
set ylabel "z"
# Plot interpolated slice (red points)
plot 'data/interpolated_constant_y.txt' using 1:3 with points pt 7 lc rgb "red" title 'Interpolated', \
     'data/constant_y.txt' using 1:3 with points pt 7 ps 1.2 lc rgb "black" notitle, \
     'data/constant_y.txt' using 1:3 with points pt 7 ps 1 lc rgb "blue" title 'Original Points'


unset multiplot