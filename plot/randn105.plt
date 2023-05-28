set terminal png
set output "./plot/randn105.png"
set term png size 1200, 800

filename(n) = sprintf("./data/randn105-chi%d.dat", 4*i)
labelname(n) = sprintf("chi=%d", 4*i)
logZ = 1.1358561247467855

set key left bottom

set multiplot layout 2,2 columns

    set logscale x
    set format y "10^{%L}"
    set xr [1:400]
    
    set ylabel "error"
    set xlabel "iteration"

    maxn = 5

    # set yr [1.6:1.8]

    set ylabel "error of <l|M|r>"
    plot for [i=1:maxn] filename(i) using ($1):(abs(logZ-($4))) title labelname(n)

    set ylabel "power convergence"
    plot for [i=1:maxn] filename(i) using ($1):(1-$8) title labelname(n),for [i=1:maxn] filename(i) using ($1):(1-$9) title labelname(n)
    
    set ylabel "1 - power fidelity"
    plot for [i=1:maxn] filename(i) using ($1):(1-$6) title labelname(n)

    set ylabel "1 - (un-dn) overlap"
    plot for [i=1:maxn] filename(i) using ($1):(abs(1-($5))) title labelname(n)

unset multiplot