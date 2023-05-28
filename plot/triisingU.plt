set terminal png
set output "./plot/triisingU.png"
set term png size 1200, 800

filename(n) = sprintf("./data/triisingU-chi%d.dat", 4*i)
labelname(n) = sprintf("chi=%d", 4*i)
logZ = 0.3230659669

set multiplot layout 2,2 columns

    unset logscale
    set logscale xy
    set ylabel "error"
    set xlabel "iteration"
    set format y "10^{%L}"

    maxn = 4

    set xr [10:400]
    # set yr [1.6:1.8]

    set ylabel "error of <l|M|r>"
    plot for [i=1:maxn] filename(i) using ($1):(abs(logZ-($4))) title labelname(n)#, "./data/triising-chi16-corl.dat" using ($1/100000) title "cor-length"

    set ylabel "error of <r|M|r>"
    plot for [i=1:maxn] filename(i) using ($1):(abs(logZ-($2))) title labelname(n)
    

    set ylabel "1 - power fidelity"
    plot for [i=1:maxn] filename(i) using ($1):(1-$6) title labelname(n)

    set ylabel "1 - (un-dn) overlap"
    plot for [i=1:maxn] filename(i) using ($1):(abs(1-($5))) title labelname(n), "./data/triising-chi16-corl.dat" using ($1/1000) title "cor-length"

unset multiplot