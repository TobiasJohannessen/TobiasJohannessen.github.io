set terminal pngcairo size 800,600 enhanced font 'Arial,10'
set output 'plots/grid.png'

set datafile separator "\t"
set title "Color-coded Recti-Linear Grid"
set xlabel "x"
set ylabel "y"
unset ztics

set view map             # 2D view
unset key                # no legend
set palette rgbformulae 33,13,10
set colorbox             # show color scale

# size variable sets point size
splot 'data/grid.txt' using 1:2:3 with points palette pointtype 7 pointsize 1
