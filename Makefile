all:

test: always
	julia --project=@. -e "using Pkg;Pkg.test()"

always: