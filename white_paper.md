#PDF Modeling Whitepaper
##Introduction:
PDF is a very powerful technique for determining atomic structure in non-crystalline materials.  It has the ability to significantly help with the nanostructure problem, enabling in-situ/operando examination of structural changes in NPs.  However, current techniques for modeling PDFs are not up to the challenge of very complex and disordered NP structure.

##Literature review:
###“Building and refining complete nanoparticle structures with total scattering data”
This paper details the work underway with DISCUS and DIFFEV to model NP structure.  DISCUS is a fortran based program which builds NPs and generates their associated PDFs.  The PDF is then fed into DIFFEV, which examines the difference between the experimental and calculated PDFs, 'evolves' the structure and comes up with new values for the set of parameters to use in the rebuilding of the NP.  While this approach has lead to some interesting results, it seems to not have beaten PDFgui at fitting the structure of their Au model system.  It is interesting to see the ligand's effect on the PDF, although this is using neutrons, not x-rays.
####Pros:
1. The system is already built
1. The system can handle large NPs with ligands
1. They seem to have a method for calculating the local density rho(r)

####Cons:
1. The system is slow: “The refinement was run with 221 cycles, each approximately 17 h, using 32 CPUs in parallel.”
1. The number of parameters used is rather limited, only 5 or 6 used in this paper
1. Building the NP out of a bulk crystal structure without any position randomization seems to not reflect the kinetic regime in which the particles are synthesized in.
1. Fortran is a giant pain to write for/in
1. System only takes in PDF as inputs, no other calculations are used
1. How much surface information is actually gleaned, outside of the ligand count? e.g. could you calculate appropriate d-bands, is the surface relaxed enough, does it show enough disorder, 
where are the vacancies/interstitials?

###”Bulk Metallic Glass-like Scattering Signal in Small Metallic Nanoparticles”
This is cute, I like the use of Glass-like intuition for the PDF.  Maybe we can get a glassyness parameter which describes how isotropic the sample structure is by fitting a oscillatory function to it?  The paper seems to be developing a very disordered model of the NP by superimposing a bunch of icosahedral packed groups.  While the paper mentions the surface an an important part of the overall structure, it does not give an atomic configuration to this structure, simply stating that it is most likely disordered.   Interestingly they don't mention the surface ligands at all.  I like how they mention that HRTEM fails to give good local structure, I'll use that as a reference next time someone asks about HRTEM.
####Pros:
1. The system is already built
1. Use ADP values to broaden peaks

####Cons:
1. Only 5 parameters are used, which results in a lack of determination of the surface structure
1. No external calculator integration

###”3-D Structure of Nanosized Catalysts by High-Energy X-ray Diffraction and Reverse Monte Carlo Simulations: Study of Ru”
This is a standard reverse monte carlo method for NP structure refinement.  
It seems that they can deal with some non-metal ligand effects, 
e.g. their Ru-S bonds. This method can use valence/neighbor constraints.  
They seem to have interesting outputs in terms of sulfur coverage on certain 
atomic sites.
####Pros:
1. The system is already built
1. Good agreement between experiment and calculations, in general
1. Strong support for surface structure
1. Some ligand incorporation
####Cons:
1. From personal experience a *giant* pain to work with, very bad documentation
1. Poor support for external potentials
1. ADPs/DW factors are not refined
1. Standard Monte Carlo can be slow to converge

My proposed approach
====================
Use a combination of Hamiltonian Monte Carlo and ab-initio energy calculations to obtain better structures.

**_NOTE_**: When discussing "potential" and "kinetic" energies, 
it is important to note that these are not necessarily the physical energies 
of the system.  In particular the potential energies under discussion here 
are a linear combination of the RW value of the experimental and calculated 
PDFs and the electronic/force field energies from DFT/CL calculations.
####Hamiltonian Monte Carlo (HMC)
The HMC method is a variation on the standard Metropolis-Hastings Monte Carlo (MHMC) method.  In the MHMC method, a new parameter position vector **q** is chosen at random by adding a random **dq** vector to the current 
**q**.  The potential energy function E(**q**) is evaluated at the new **q** 
and compared to the old energy before the **dq** move.  If the move leads to a lower E
(**q**) then it is accepted, otherwise it is accepted or rejected randomly 
using the exponential of the energy difference as a weight.  Thus moves that 
are truly bad are rejected, and moderately bad moves can be accepted.  This 
prevents the simulation being stuck in a potential energy well.  However, 
this method can be slow to converge.  HMC supplements this by adding two 
features, both related to kinetic energy.  Firstly instead of adding a simple
 random displacement to the position coordinate, the gradient of the 
 potential energy is calculated with respect to **q**.  This leads to a 
 momentum vector which is used to guide the dynamics, pushing the trajectory 
 toward lower energies more efficiently.  Secondly, 
 instead of simply comparing potential energies the total Hamiltonian, 
 which is the sum of potential and kinetic energies are compared leading to 
 the acceptance of positions which have better agreement with experiment and 
 are more stable with respect to to **q** deviations.  This method still uses
  randomization as a driving feature, the initial momentum vector **p** is 
  chosen randomly, and the acceptance/rejection criteria still has a exp
  (deltaH) random element.
####Implmentation of external calculators
Since the HMC method simply relies on the kinetic and potential energies of 
the system we can add any number of external calculators, 
so long as they put out a scalar energy value which is minimized for more 
physical systems.  An optional constraint is that they also put out force 
vectors, which can be used to generate the **p** vector, 
this is not necessary, just nice to have.  

For example we can add a DFT calculator to the process by simply adding the 
resulting DFT electronic energy of the system, times a user supplied constant
 to the RW value, when calculating the potential energy of the system.  Some 
 DFT calculators also put out forces for each atom in the configuration, 
 which can be converted to **p** via F*delta_t = delta_p, 
 where delta_t is the time chunk used in the simulation.  Combining this with 
 the PDF will help to generate structures which have physical significance in
  addition to experimental agreement.
####Algorithm issues
HMC can be rather expensive because of the calculation of the gradient, 
which requires the calculation of the change in potential energy for each 
parameter.  Structural calculators which supply forces eliminate them from 
this recalculation.  However, the calculation of grad(RW) requires the 
calculation of the PDF/RW for each potential movement.  Thus for N atoms, 
and no other parameters than the atomic positions, 
we must calculate the PDF which loops of N twice and the RW.  Thus, we have a 
computational complexity of 3*N*((N-1)*O1+O2), if we save the initial PDF 
contributions in a matrix we will name **D**.  O1 signifies the time it takes
 to calculate the distance between two atomic positions, for one coordinate, 
 and O2 signifies the time it takes to convert the original **D** to the 
 modified **D**, and calculate the RW from the new **D**.  As you can see 
 this algorithm is ripe for embarrassingly parallel methods, 
 since none of the individual gradient elements depend on one another.  Thus,
  we can use GPU based acceleration, which contains many cores able to do 
  simple calculations to get a significant speedup, 
  distributing our calculations over at least 3*N cores, 
  if not more.  Note that we can readily reduce the number N by fixing atoms,
   like in the core of the NP to be crystalline, by simply telling the 
   program to not calculate the gradient for those atoms.  Also note that 
   ADPs/DW factors can be added in, to each atom and direction if desired, 
   this just adds to the total dimensionality of the phase space, 
   from 3N to max 6N.
#####A little bit about D
D is a NxNx3 matrix which describes the pair distances between all the atoms 
in the configuration.  I.e. D[0,1,0] is the length in the x direction between
 atom 0 and atom 1.  For each D[:,:,i] slice the sub matrix is anti-symmetric
  with respect to the diagonal and the diagonal elements are zero.  By 
  storing this information, which is further reduced to produce the PDF and 
  RW, we can avoid recalculating the entire PDF every time we wish to take 
  the gradient.  Thus, we only switch out one row/column every time we move 
  an atom. 
####Algorithm questions
We have a few parameters in this kind of simulation, 
including how do we set the unit time and length spans for our calculations? 
 How do we set the linear scaling parameters between our experimental and 
 calculator based energies.

#####Other algorithm notes
We should be able to support EXAFS, if we can get PDF like object, 
or RW value from it.  Other spectroscopy would need a model to describe how 
the absorption changes with atomic position/thermal vibrations, 
this may be able to be gleaned from CL/QM calculations.

##Global issues:
The ADP/Debye Waller factors seem to include structural disorder and thermal vibrations.  We need a way to separate these two, one could have a very tight glass which is very disordered by vibrationally limited.  Similarly, one could have a very vibrationally active structure, which is very crystalline.  Can we do this with a temperature change, possibly pulsed?  Can we look at this theoretically by calculating the phonon modes and backing out the thermal vibrations/DW factors?