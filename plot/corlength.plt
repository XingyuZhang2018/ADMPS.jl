set terminal png
set output "./plot/corl.png"
set term png size 1200, 800

set logscale y
set xlabel "Itertion"
set ylabel "Correlation length"
plot "./data/corl.dat" using 1:2 title "l", "./data/corl.dat" using 1:3 title "r"