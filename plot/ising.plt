set terminal png
set output "./plot/ising.png"
set term png size 1200, 800

filename(n) = sprintf("./data/ising-chi%d.dat", 2*2**i)
labelname(n) = sprintf("chi=%d", 2*2**i)
logZ = 0.92969540209

set multiplot layout 2,2 columns

    set logscale xy
    set ylabel "error"
    set xlabel "iteration"
    set format y "10^{%L}"

    maxn = 5

    # set xr [1:400]
    # set yr [1.6:1.8]

    set ylabel "error of <l|M|r>"
    plot for [i=1:maxn] filename(i) using ($1):(abs(0.92969540209-($4))) title labelname(n)

    set ylabel "error of <r|M|r>"
    plot for [i=1:maxn] filename(i) using ($1):(abs(0.92969540209-($2))) title labelname(n)

    set ylabel "1 - power fidelity"
    plot for [i=1:maxn] filename(i) using ($1):(1-$6) title labelname(n)

    set ylabel "1 - (un-dn) overlap"
    plot for [i=1:maxn] filename(i) using ($1):(abs(1-($5))) title labelname(n)

unset multiplot