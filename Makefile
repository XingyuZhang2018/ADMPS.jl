all:

plot: all
	singularity exec /data/dp/singularity_image/gnuplot_5.2.6.sif gnuplot plot/*.plt

test: always
	julia --project=@. -e "using Pkg;Pkg.test()"

always: