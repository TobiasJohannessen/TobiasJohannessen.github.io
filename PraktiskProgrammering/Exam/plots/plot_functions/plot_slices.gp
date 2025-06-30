stats 'data/constant_x.txt' using 1 nooutput
const_x = STATS_min

stats 'data/constant_y.txt' using 2 nooutput
const_y = STATS_min

set terminal pngcairo size 300,800 enhanced font 'Arial,10'
set output 'plots/slices.png'

set datafile separator "\t"
set palette rgbformulae 33,13,10

set multiplot layout 3,1 title "Heatmap and Line Slices" margins 0.1,0.8,0.1,0.8 spacing 0.1,0.1

# === Plot 1: Full heatmap (top) ===
set title "2D Heatmap"
set xlabel "x"
set ylabel "y"
unset ztics
set view map
set colorbox
unset key

splot 'data/grid.txt' using 1:2:3 with points palette pointtype 7 pointsize 1

# === Plot 2: z vs y at constant x ===
set title sprintf("Slice at x = %.2g", const_x)
set xlabel "y"
set ylabel "z"
unset colorbox
unset view
unset xtics
unset ytics
set tics out nomirror
set autoscale x
set autoscale y
plot 'data/constant_x.txt' using 2:3 with points lt rgb "blue" title 'x = constant'

# === Plot 3: z vs x at constant y ===
set title sprintf("Slice at y = %.2g", const_y)
set xlabel "x"
set ylabel "z"
plot 'data/constant_y.txt' using 1:3 with points lt rgb "red" title 'y = constant'

unset multiplot
