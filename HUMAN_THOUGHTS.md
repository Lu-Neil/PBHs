# Tasks that need to be completed:
- Verifying the behaviour with signals longer than 2 days that get split up and incoherently combined
- Understanding how the PSD affects the analysis. Known lines will be smeared across many frequency bins, how to account for this

# Notes about the implementation
- The t_crossover values is extremely important and depends on f0. For each beta we search for signals from many f0, will need to take the lower bound f0 (=20Hz) to determine the largest permitted \Delta \beta
