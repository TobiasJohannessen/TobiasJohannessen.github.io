set terminal pngcairo size 800,600 enhanced font 'Arial,10'
set output 'plots/bicubic_interpolated_grid.png'

set datafile separator "\t"
set title "Interpolated Grid"
set xlabel "x"
set ylabel "y"
unset ztics

set view map         # 2D view
unset key            # no legend for interpolated data
set palette rgbformulae 33,13,10
set colorbox         # color scale for interpolated grid

# Plot interpolated grid with its palette
splot 'data/bicubic_interpolated_grid.txt' using 1:2:3 with points palette pointtype 7 pointsize 1 notitle, \
      'data/grid.txt' using 1:2:3 with points pt 7 ps 1.2 lc rgb "black" notitle, \
      'data/grid.txt' using 1:2:3 with points pt 7 ps 1 palette title 'Original Points'