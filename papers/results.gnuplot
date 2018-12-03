# tutorial used:
# http://physics.ucsc.edu/~medling/programming/gnuplot_tutorial_1/index.html
set terminal postscript color "Times,25"
set output 'results.eps'
set title "Results"
set xlabel "n"
set ylabel "t (seconds)"
set key top left
f(x) = 0.00000043*(x**2)-0.035
g(x) = 0.000015*x*log(x)
plot [1:1056] 'results.dat' using 1:2:xtic(1) title "t", \
        "" smooth csplines title "csplines(t)", \
        f(x) title "0.00000043x^2-0.035", \
        g(x) title "0.000015xlog(x)"
