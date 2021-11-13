# Liquid Crystals Project
### Part III Project modelling the phase behaviour of DNA nanoparticles using LAMMPS

The programmability of DNA allows us to design anisotropic nano-particles such as rods, triangles and many other shapes. 
Such anisotropy is a pre-requisite to form liquid crystalline structures, and allows the construction of new shapes where we expect to obtain completely new liquid crystalline symmetries. Based on oxDNA, a semi-coarse-grained, freely available simulation package that can provide the topological and stability criteria of any DNA nano-particle, 
the Eiser group have developed a more coarse-grained model to simulate large numbers of these mesogens, such that their phase-behavior can be studied.

Here I use a course-grained model, based on LAMMPS, to study the phase behaviour of ‘nunchuks’, strands of double stranded DNA with a central section where one of the strands are missing.These are modelled as two hard rods connected via a flexible linker (hence the 'nunchucks' moniker), such that they can vary their configuration from fully stretched to folded.  Such systems are expected to form smectic (layered) phases at high volume fractions. My work also demonstrated the existence of a biaxial phase, and introduced novel dynamic methods to characterise new phases.

This work formed my masters report for a Masters in Science at the University of Cambridge, 2021.

### External Use

This repo has been published publically so that others can access my code, replicate my results and use/continue my research. However this was not (and has never been) the primary purpose of this repository - it has always been to offer version control and remote backup of my research. As such, many of the files include will have little public interest, and may not be written with external users in mind.

I therefore wish to highlight a number of aspects of this repository that have been added with public/externals users specifically in mind.

**Simulation_Scripts** - Contains all files required to initialise and run a molecular dynamics simulation on this system, using LAMMPS  
**Analysis_Scripts** - Contains all the python scripts used to analyse and plot the output files of the simulation scripts  
**Final_Documents/ProjectReport_final** - My Masters report, containing detailed accounts of the methodology and results  

A more complete explanation of the structure of this repository is avaliable in the ```contents.txt``` file.


