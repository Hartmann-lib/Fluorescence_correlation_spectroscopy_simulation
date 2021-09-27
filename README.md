# Fluorescence_correlation_spectroscopy_simulation

Fluorescence correlation spectroscopy (FCS) is a versatile measurement technique to investigate interactions of biomolecules, binding affinities and dimensions of single-molecules and bigger complexes. To this end, the auto-correlation curve is extracted from intensity time traces. In the simple case of one diffusing species 
with unique size, a theoretical function can be applied to obtain the diffusion constant and hence the size of the molecules. However, the interpretation of correlation curves becomes more difficult in the presence of multiple species with different diffusion constants and molecular brightnesses. Beside heterogeneity, interconversion dynamics between two species represents just another layer of complexity. In this software package two simulation tools are provided helping to explore and understand measured FCS data.

'simFCS_species.py' - A tool for simulating multiple fractions of molecules with different diffusion coefficient and molecular brightness.

'simFCS_dynamic.py' - A tool for simulating two interconverting species of molecules with different diffusion coefficient and molecular brightness.

Both simulation tools take the confocal geometry (kappa, Veff) into account and allow the manipulation of time step length, number of simulation steps and number of
iterations.

You can find the parameter section commented in the respective main file.

Have fun!

![Figure_simFCS_species](https://user-images.githubusercontent.com/58071484/134956332-4ecbb059-c33c-4b0e-85ea-cacbdb648beb.png)

![Figure_simFCS_dynamics](https://user-images.githubusercontent.com/58071484/134976700-ee7a07fe-b618-4aa7-92a2-cab62ac1c30f.png)
