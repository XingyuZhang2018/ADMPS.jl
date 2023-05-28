set terminal png
set output "./plot/triisingAT.png"
set term png size 1200, 800

filename(n) = sprintf("./data/triisingAT-chi%d.dat", 4*i)
labelname(n) = sprintf("chi=%d", 4*i)
logZ = 1.29226387

set key left bottom

set multiplot layout 2,2 columns

    unset logscale
    set logscale xy
    set ylabel "error"
    set xlabel "iteration"
    set format y "10^{%L}"

    maxn = 4

    set xr [1:400]
    # set yr [1.6:1.8]

    set ylabel "error of <l|M|r>"
    plot for [i=1:maxn] filename(i) using ($1):(abs(logZ-($5))) title labelname(n)

    set ylabel "power convergence"
    plot for [i=1:maxn] filename(i) using ($1):(1-$9) title labelname(n),for [i=1:maxn] filename(i) using ($1):(1-$10) title labelname(n)
    

    set ylabel "1 - power fidelity"
    plot for [i=1:maxn] filename(i) using ($1):(1-$7) title labelname(n), for [i=1:maxn] filename(i) using ($1):(1-$8) title labelname(n)

    set ylabel "error of <r|M|r>"
    plot for [i=1:maxn] filename(i) using ($1):(abs(logZ-($2))) title labelname(n), for [i=1:maxn] filename(i) using ($1):(abs(logZ-($3))) title labelname(n)

unset multiplot