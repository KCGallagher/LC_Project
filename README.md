# Liquid Crystals Project
### Part III Project modelling the phase behaviour of DNA nanoparticles using LAMMPS

The programmability of DNA allows us to design anisotropic nano-particles such as rods, triangles and many other shapes. 
Such anisotropy is a pre-requisite to form liquid crystalline structures, and allows the construction of new shapes where we expect to obtain completely new liquid crystalline symmetries. 
Based on oxDNA, a semi-coarse-grained, freely available simulation package that can provide the topological and stability criteria of any DNA nano-particle, 
the Eiser group have developed a more coarse-grained model to simulate large numbers of these mesogens, such that their phase-behavior can be studied.

Here I use a course-grained model, based on LAMMPS, to study the phase behaviour of ‘nunchuks’, strands of double stranded DNA with a central section where one of the strands are missing.
These are modelled as two hard rods connected via a flexible linker (hence the 'nunchucks' moniker), such that they can vary their configuration from fully stretched to folded.  
Such systems are expected to form smectic (layered) phases at high volume fractions.
