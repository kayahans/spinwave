#!/usr/bin/env python3
from importlib import import_module
from importlib.metadata import version
import typing
import warnings
import sys, os

__author__ = "Kayahan Saritas"
__version__ = "0.0.1"
__email__ = "saritaskayahan@gmail.com"
__status__ = "Development"
__citation__ = "https://arxiv.org/abs/2205.00300, J. Phys.: Condens. Matter 27 (2015) 166002"

# Currently supported version dependencies
currently_supported = {
    'numpy'      : (1,   13,  1),
    'pymatgen'   : (2022, 0,  17),
    'ase'        : (3,   22,  1), 
    'numpy'      : (1,   20,  3),
    'scipy'      : (1,    7,  3),
    'atomate'    : (1,    0,  3), # optional
    'matplotlib' : (3,    5,  0), # optional
    }

required_dependencies = set(['numpy', 'pymatgen', 'ase', 'numpy', 'scipy'])
missing_dependencies = []
used_dependencies = []

def check_modules():
    """
    Track version of package dependencies
    """
    module_list = currently_supported.keys()
    for name in module_list:
        try:
            import_module(name)
            ver = tuple([int(x) for x in version(name).split(".")])
            if ver < currently_supported[name]:
                warnings.warn("Check backwards compatibility: ", name, "@", ver)
            used_dependencies.append(name)
        except:
            if name in required_dependencies:
                print("Error: Module ", name, ">=", currently_supported[name], " is a core dependency, please install\n")
                sys.exit(1)
            else:
                warnings.warn("Module ", name, " is an optional dependency, some functionality may be missing\n")
                missing_dependencies.append(name)

check_modules()

constants = {
             'kb' :  8.617 * 1e-2 # meV/K
            }

for name in used_dependencies:
    if name == 'pymatgen':
        from pymatgen.analysis.magnetism.analyzer import CollinearMagneticStructureAnalyzer
        from pymatgen.io.vasp import Poscar
        from pymatgen.symmetry.kpath import KPathSeek
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        from pymatgen.core import Structure, PeriodicSite
    if name == 'ase':
        from ase.dft.kpoints import monkhorst_pack
    if name == 'atomate':
        from atomate.vasp.drones import VaspDrone
    if name == 'matplotlib':
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        mpl.use('tkagg')
    if name == 'numpy':
        import numpy as np
        np.set_printoptions(suppress=True)
    if name == 'scipy':
        import scipy


class Coupling(object):
    def __init__(self, site1 : int, 
                 site2 : int, 
                 J : float, 
                 dij : typing.List[int], 
                 distances : typing.List[float]):
        """Pairwise coupling object

        Args:
            site1 (int): first site index
            site2 (int): second site index
            J (float): isotropic exchange coupling
            dij (typing.List[int]): translation vector
            distances (typing.List[float]): pairwise distances
        """
        self.dij = dij
        self.Jij = J
        self.i = site1
        self.j = site2
        self.distances = distances

#end class Coupling

class Spinw(object):
    """
    Calculates Spinwave hamiltonians, magnon dispersions
    This is the collinear only version of the work in
    J. Phys.: Condens. Matter 27 (2015) 166002
    """
    def __init__(self):
        self.sites = []
        self.labels = []
        self.lattice = None
        self.J = []
        self.all_atoms_added = False
        self.all_couplings_added = False
        self.num_sites = 0
        self.ham = None
        self.interactions = []
    #end def __init__
    
    def _get_interaction_mat0(self):
        """
        J at k == Gamma to calculate the C block matrix in the hamiltonian
        """
        J = self._get_interaction_mat(np.array([[0,0,0]]))[0]
        return J
    #end def _get_interaction_mat0
    
    def _get_interaction_mat(self, kpoints: typing.List[float]):
        """ returns J matrix

        Args:
            kpoints (typing.List[float]): list of k-points. Nx3 list

        Returns:
            np.ndarray: returns J, a nk x ns x ns x 3 x 3 matrix
            nk is the number of k-points 
            ns is the number of sites
        """
        ns = self.num_sites
        I = 1j # imaginary 
        J = np.array([[[np.zeros((3,3)) for _ in range(ns)] for _ in range(ns)] for _ in range(len(kpoints))], dtype=complex)

        one = np.eye(3)
        lattice = self.lattice.matrix
        rlattice = self.lattice.reciprocal_lattice.matrix
        
        for ik, k in enumerate(kpoints):
            for coup in self.interactions:
                i       = coup.i
                j       = coup.j
                site_i = self.sites[i]
                site_j = self.sites[j]
                Jd      = coup.Jij
                d_list  = coup.dij

                Jkij = 0
                fk = []
                
                for d in d_list:
                    fk.append(np.exp(2*np.pi*I*np.dot(k,d)))    #Fourier transform of Jij
                
                Jkij = Jd @ (one * sum(fk))
                J[ik, i, j] += Jkij 
                J[ik, j, i] += Jkij.conjugate().transpose()
        return J
    #end def _get_interaction_mat
    
    def add_atom(self, site:PeriodicSite, s:float):
        """add atom to periodic cell

        Args:
            site (PeriodicSite): atomic site (classical)
            s (float): spin magnitude (not spin-1/2)
        """
        assert(self.all_atoms_added == False)
        # We use pymatgen's "PeriodicSite"" type and add a few new member variables to it starting ".sw_"
        site.S = s
        site.sw_S = abs(s)/2 # convert to spin-1/2
        spin_direction = abs(s)/s # only collinear
        site.sw_R = np.eye(3) 
        site.sw_R[2,2] = spin_direction
        I = 1j
        
        # e1 + e2*I
        site.sw_u = site.sw_R[:,0, None] + spin_direction * I * site.sw_R[:,1, None] 
        # e3
        site.sw_v = site.sw_R[:,2, None]
        self.sites.append(site)

        # Add unique site name for each site added
        new_label = site.species_string + str(self.num_sites + 1)
        if new_label not in self.labels:
            self.labels.append(new_label)
            self.num_sites += 1
    #end def add_atom
    
    def add_coupling(self, 
                     ij_labels : typing.Tuple[int, int], 
                     val : typing.Union[float, np.ndarray],
                     rmax: float = 4.5):
        """adds 2-body pairwise magnetic coupling between two sites

        Args:
            ij_labels (typing.Tuple[int, int]): site indices 
            val (typing.Union[float, np.ndarray]): J magnitude. If float J is isotropic. if np.ndarray (3x1 vector), then anisotropic exchange. Units in meV.
            rmax (float, optional): _description_. Defaults to 4.5.
        """
        assert(self.all_couplings_added == False)
        
        site_i, site_j = ij_labels
        assert(site_i in self.labels)
        assert(site_j in self.labels)
        
        self.all_atoms_added = True # add no more atoms after using addcoupling
        
        self.labels = np.array(self.labels)
        i = np.where(self.labels==site_i)[0][0]
        j = np.where(self.labels==site_j)[0][0]
        val = val/2
        
        if np.isscalar(val):
            val = np.eye(3)*val
            
        if i == j and rmax == 'onsite':
            # On site interaction
            dij = [0,0,0]
            distances = [0]
            coup = Coupling(i, j, val, dij, distances)
            self.interactions.append(coup)
        else:
            # pairwise interactions for ith site
            neighbor_list_i    = self.structure.get_neighbor_list(rmax, sites=[self.sites[i]])
            # centers          = neighbor_list_i[0]
            neighbor_indices_i = neighbor_list_i[1]
            lattice_translations_i = neighbor_list_i[2]
            distances_i = neighbor_list_i[3]
            mask = neighbor_indices_i == j
            
            dij_i = -lattice_translations_i[mask]
            distances_i = distances_i[mask]
            
            coup_i = Coupling(i, j, val, dij_i, distances_i)
            self.interactions.append(coup_i)
            
            if i != j:
                # pairwise interactions for jth site
                # we double count here to get the correct structure factors
                # but then divide by two in the hamiltonian
                neighbor_list_j    = self.structure.get_neighbor_list(rmax, sites=[self.sites[j]])
                # centers          = neighbor_list_j[0]
                neighbor_indices_j = neighbor_list_j[1]
                lattice_translations_j = neighbor_list_j[2]
                distances_j = neighbor_list_j[3]
                mask = neighbor_indices_j == i
                
                dij_j = -lattice_translations_j[mask]
                distances_j = distances_j[mask]
                
                coup_j = Coupling(j, i, val, dij_j, distances_j)
                self.interactions.append(coup_j)
    #end def add_coupling
    
    def band_plot(self,
                  ax : plt.Axes,
                  kpath_labels : typing.List[str] = None,
                  kx : np.ndarray = None,
                  normalize : bool = False):
        """
        After hamiltonian is constructed on a kpath, plot its magnon band structure

        Args:
            ax (plt.Axes): matplotlib axes object
            kpath_labels (typing.List[str], optional): kpath labels. Defaults to None.
            kx (np.ndarray, optional): kpoint spacings. Defaults to None.
            normalize (bool, optional): divides energies with JS, makes unitless energies. Defaults to False.

        Returns:
            _type_: _description_
        """

        nk = self.num_kpoints
        if kx is None:
            x = np.linspace(-1, 1 , nk)
        else:
            x = kx
        
        JS = 1
        if normalize:
            S_tot = sum([ site.sw_S for site in self.sites])/2
            # Use first interaction as defined 
            J = abs(self.interactions[0].Jij[0,0]*2)
            JS = J * S_tot
        
        y= np.sort(self.eik, axis=1).real  / JS  #meV or JS unit 
        
        ax.plot(x,y[:,0], '-b', label='spinw')
        ax.plot(x,y[:,1:], '-b')
        if normalize:
            ax.set_ylabel("Energy (JS)")
        else:
            ax.set_ylabel("Energy (meV)")
            
        max_y = np.max(y)
        pad = 0.05
        ax.set_ylim((-pad*max_y, max_y*(1+pad)))
        ax.set_xlim((min(x), max(x)))
        if kpath_labels is not None:
            for ln, li in enumerate(kpath_labels):
                if li != ' ':
                    plt.axvline(x[ln], ymin=-1000, ymax=1000, linewidth=2, color='k')
                #end if
            #end for
            ax.set_xticks(x, kpath_labels)
            ax.tick_params(axis=u'x', which=u'both',length=0)
        #end if
        
        return ax
    #end def band_plot
    
    def make_ham(self, kpoints : typing.Union[typing.List, np.ndarray] = np.array([[0.,0.,0.], [0.5, 0., 0.]])):
        """Create Spin-wave hamiltonian

        Args:
            kpoints (np.ndarray, optional): list of k-points in reciprocal coordinates. Defaults to np.array([[0.,0.,0.], [0.5, 0., 0.]]).
        """
        # add no more coupling after forming the hamiltonian
        self.all_couplings_added == True 
        assert(kpoints.shape[1] == 3)
        nk = kpoints.shape[0]
        ns = self.num_sites
        self.num_kpoints = nk
        sites = self.sites
        self.ham = np.zeros((nk, ns*2, ns*2), dtype='complex')
        A = np.zeros((nk, ns, ns), dtype='complex')
        B = np.zeros((nk, ns, ns), dtype='complex')
        C = np.zeros((nk, ns, ns), dtype='complex')
        
        self.J = self._get_interaction_mat(kpoints)
        
        J0 = self._get_interaction_mat0()
        
        for ki in range(nk):
            J = self.J[ki, :]
            for i, site_i in enumerate(sites):
                for j, site_j in enumerate(sites):
                    sqrt_Sij = np.sqrt(site_i.sw_S*site_j.sw_S)
                    
                    A[ki, i, j] = 1./2  * sqrt_Sij * (site_i.sw_u.T @ J[i,j] @ site_j.sw_u.conjugate())
                    B[ki, i, j] = 1./2.  * sqrt_Sij * (site_i.sw_u.T @ J[i,j] @ site_j.sw_u)
                    
                    for l, site_l in enumerate(sites):
                        C[ki, i, j] += (i==j) * site_l.sw_S * site_i.sw_v.T @ J0[i,l] @ site_l.sw_v
        
        
        for ki in range(nk):
            self.ham[ki] = np.block([[A[ki]-C[ki], B[ki]], [B[ki].conjugate().transpose(), A[ki].conjugate()-C[ki]]])
    #end def make_ham
    
    def get_eik(self) -> np.ndarray:
        ns = self.num_sites
        nk = self.num_kpoints
        one  = np.eye(ns)
        zero = np.zeros((ns,ns))
        g    = np.block([[one, zero], [zero, -one]])
        
        h_small = 1E-11 * np.block([[one, zero], [zero, one]])
        self.eik = []
        self.modes = []
        self.modes_e = []
        self.evec = []
        for ki in range(nk):
            ham = self.ham[ki]+h_small
            eig_e, eig_v =  np.linalg.eigh(ham)
            
            K = scipy.linalg.cholesky(ham)
            Kdag = K.conjugate().transpose()
            kgk = K @ g @ Kdag

            eig_e, U = np.linalg.eigh(kgk)
            U_dag = U.conjugate().transpose()
            
            L = U_dag @ kgk @ U
            
            E = g@L
            modes_e = np.diag(E)
            T = np.linalg.inv(K) @ U @ scipy.linalg.sqrtm(E)
            
            self.modes.append(T)
            self.modes_e.append(modes_e)
            self.eik.append(eig_e)
        
        self.eik = np.array(self.eik)
        self.evec = np.array(self.evec)
        
        return self.eik
    #end def get_eik
    
    def get_band_structure(self, 
                           ax, 
                           k1=[0,0,0], 
                           k2=[0.5, 0., 0], 
                           labels = None, 
                           num_k=20, 
                           normalize=False, 
                           figname = "band_line.pdf", 
                           use_cholesky = True):

    
        klist = np.linspace(k1, k2, num_k)
        self.make_ham(kpoints=klist)
        self.get_eik()
        if labels is not None:
            kpath_labels = [' '] * klist.shape[0]
            kpath_labels[0] = labels[0]
            kpath_labels[-1] = labels[1]
        else:
            kpath_labels = labels
        self.band_plot(ax, normalize=normalize, kpath_labels=kpath_labels)
        
        with open(self.directory+'/line_kpath_rel.out', 'w') as f:
            np.savetxt(f, klist, fmt='%.6e')
            
        with open(self.directory+'/line_kpath_cart.out', 'w') as f:
            rec_lattice = self.structure.lattice.reciprocal_lattice.matrix
            np.savetxt(f, klist @ rec_lattice, fmt='%.6e')
            
        with open(self.directory+'/line_kpath_eigenvalues.out', 'w') as f:
            eik = self.eik
            np.savetxt(f, eik, fmt='%.6e')
    #end def get_band_structure
    
    def get_full_band_structure(self, 
                                ax, 
                                numk = 20, 
                                figname='complete_band_structure.pdf', 
                                normalize = False, 
                                use_cholesky = True):
        kps = KPathSeek(self.structure)
        kpoints = kps.kpath['kpoints']
        kpaths = kps.kpath['path']
        
        complete_kpath = []
        complete_kpath_labels = []
        complete_kx = []
        first_path = True
        for kpath_num, kpath in enumerate(kpaths):
            numk_in_path = len(kpath)
            
            if kpath_num == 0:
                kx_ref = 0
            else:
                kx_ref = complete_kx[-1]
                
            for ki in range(numk_in_path-1):
                k1 = np.array(kpoints[kpath[ki]])
                k2 = np.array(kpoints[kpath[ki+1]])
                k1_label = kpath[ki]
                k2_label = kpath[ki+1]
                
                if k1_label == 'GAMMA':
                    k1_label = '$G$'
                else:
                    k1_label = '${}$'.format(k1_label)
                if k2_label == 'GAMMA':
                    k2_label = 'G'
                else:
                    k2_label = '${}$'.format(k2_label)
                    
                klist = np.linspace(k1, k2, numk)
                kx    = np.linspace(0, np.sqrt(np.sum(np.power(k2-k1, 2))), numk) + kx_ref
                klabel_list = [' ']*numk
                klabel_list[-1] = k2_label
                
                if ki == 0 and kpath_num == 0:
                    klabel_list[0] = k1_label
                elif kpath_num != 0 and ki == 0:
                    complete_kpath_labels[-1] += '|' + k1_label
                else:
                    klist = klist[1:]
                    klabel_list = klabel_list[1:]
                    kx = kx[1:]
                kx_ref = kx[-1]
                complete_kpath.extend(klist)
                complete_kpath_labels.extend(klabel_list)
                complete_kx.extend(kx)
        
        complete_kpath = np.array(complete_kpath)
        complete_kpath_labels = np.array(complete_kpath_labels)
        complete_kx = np.array(complete_kx)

        with open(self.directory+'/complete_kpath_rel.out', 'w') as f:
            np.savetxt(f, complete_kpath, fmt='%.6e')
            
        with open(self.directory+'/complete_kpath_cart.out', 'w') as f:
            rec_lattice = self.structure.lattice.reciprocal_lattice.matrix
            np.savetxt(f, complete_kpath @ rec_lattice, fmt='%.6e')
            
        # Use relative coordinates for the hamiltonian
        self.make_ham(kpoints=complete_kpath)
        
        self.get_eik(use_cholesky=use_cholesky)
        
        self.band_plot(ax, kpath_labels=complete_kpath_labels, kx=complete_kx, normalize=normalize)
        
        with open(self.directory+'/complete_kpath_eigenvalues.out', 'w') as f:
            eik = self.eik
            np.savetxt(f, eik, fmt='%.6e')
    #end def get_full_band_structure
    
    def magnetization(self,  
                      tmin : int = 5, 
                      tmax : int = 300, 
                      dt : int = 1, 
                      kgrid : typing.List[int] = [1,1,1], 
                      plot : bool = True, 
                      figname : str = "magnetization.pdf"):
        """Generates the magnetization vs temperature curve for 2D structures only
        One-stop shop after reading the crystalline/magnetic structure.
        Hamiltonian is updated inside this routine.

        Args:
            tmin (int, optional): starting temperature (T) in K. Defaults to 5.
            tmax (int, optional): end T in K . Defaults to 300.
            dt (int, optional): infinitesimal T in K. Defaults to 1.
            kgrid (typing.List[int], optional): reciprocal grid density. Defaults to [1,1,1].
            plot (bool, optional): plots the curve. Defaults to True.
            figname (str, optional): figure name. Defaults to "magnetization.pdf".
        """
        print("Producing the magnetization curve for 2D structures")
        print("While producing magnetization curve, division warnings might be produced\n")
        S_tot = sum([ site.sw_S for site in self.sites])
        
        mpgrid = monkhorst_pack(kgrid)        
        self.make_ham(kpoints=mpgrid)
        self.get_eik()
        
        num_branches = int(self.eik.shape[1]/2)
        Ek = self.eik[:, num_branches:]
        
        mu = []
        t  = np.arange(tmax, tmin, -dt)
        lat_k = self.lattice.reciprocal_lattice
        area_k = 2 * lat_k.volume / lat_k.c 
        kb = constants['kb']
        
        for T in t:
            # 2D Mater. 4 (2017) 035002 Eq. 25
            f = lambda m : m - S_tot + area_k * np.mean(1./(np.exp( (Ek*m) / (kb*T*S_tot) ) - 1 ))
            try:
                mu.append(np.real(scipy.optimize.newton(f, S_tot)))
                print(T, mu[-1])
            except:
                mu.append(0)
        
        mu = np.array(mu)
        mu_ind = np.argmin(np.abs(mu-S_tot/2))
        Tc = t[mu_ind]

        if plot:
            fig, ax = plt.subplots()
            plt.plot(t, mu, "-k")
            plt.title("Tc = "+ str(Tc)+ " K")
            plt.xlabel("Temperature (K)")
            plt.ylabel("Total 1/2-Magnetization (mu)")
            if figname is None:
                plt.show()
            else:
                plt.savefig(figname, format='pdf')
                print("\nMagnetization curve is saved to {}\n".format(figname))
        else:
            pass
            # TODO (kayahans): print to file
    #end def magnetization
    
    def modify_spin(self, new_spins : typing.List[int]):
        """Changes the magnetic structure, but keeps the crystalline structure the same.
        After the spins are modified, hamiltonian must be re-formed to yield new results
        Useful utility to experiment with different magnetic motifs of the same structure
        
        Args:
            new_spins (typing.List[int]): updated collinear magnetic structure with integer spins
        """
        print('Modifying spins...')
        print('Before:')
        print('Label\t S(1/2)\t Fractional coordinate')
        for specie, site in zip(self.labels, self.sites):
            print(specie, '\t' ,site.sw_S, '\t' ,site.frac_coords)
        for ind in range(0, len(self.sites)):
            self.sites[ind].sw_S = new_spins[ind]/2
        print('After:')
        print('Label\t S(1/2)\t Fractional coordinate')
        for specie, site in zip(self.labels, self.sites):
            print(specie, '\t' ,site.sw_S, '\t' ,site.frac_coords)
        print('Spins are modified!')
    #end def modify_spin
    
    def read_structure(self, 
                       directory : str ="./",
                       symmetrize : bool = False,
                       symprec : float = 0.1,
                       spins : typing.List[int] = None):
        """Read atomic and magnetic structure 

        Args:
            directory (str, optional): Input files location. Defaults to "./".
            symmetrize (bool, optional): uses the primitive cell of input structure. Defaults to False.
            symprec (float, optional): symmetrization tolerance. Defaults to 0.1.
            spins (typing.List[int], optional): Localized collinear magnetic moments on each site. 
            If OUTCAR is present, "spins" is ignored. Defaults to None.
        """
        self.directory = directory
        
        try:
            # Automated structure/spins read from vasp OUTCAR, POSCAR files
            drone     = VaspDrone(parse_locpot = False,
                              parse_bader  = False,
                              store_volumetric_data = ())
            entry     = drone.assimilate(directory)
            structure = entry['output']['structure']
            structure = Structure.from_dict(structure)
        except:
            # Only Poscar is found in the directory
            # Spins are provided in the function by hand
            if spins is not None:
                try:
                    poscar = Poscar.from_file(directory+'/POSCAR')
                    structure = poscar.structure
                except:
                    print("Error reading POSCAR in the directory")
                    sys.exit(1)

                try:
                    structure.add_site_property('magmom', spins)
                except:
                    print("Error assigning spins to sites in the structure")
                    sys.exit(1)
            else:
                print("Directory should have minimum POSCAR, OUTCAR and vasprun.xml to read structure.")
                print("Either provide these files or enter the spins here.")
                print("In read structure function you should provide spins for all the atoms in the input structure")
                sys.exit(1)
        
        if symmetrize:
            sa = SpacegroupAnalyzer(structure, symprec=symprec)
            ref_structure = sa.get_refined_structure()
            
            self.structure = ref_structure
            ref_structure.add_site_property('magmom', structure.site_properties['magmom'])
            print("Structure is symmetrized")
        else:
            self.structure = structure
        
        # Refine structure to use only the magnetically active sites
        # For example in CrI3, this procedure removes all the I atoms from the structure
        self.lattice = structure.lattice
        mag_str   = CollinearMagneticStructureAnalyzer(structure, round_magmoms=True, make_primitive=False).get_structure_with_only_magnetic_atoms(make_primitive=False)
        magmom    = mag_str.site_properties['magmom']
        
        # update stored structure with the magnetically refined structure
        self.structure = mag_str
        # add all magnetic sites to the Spin-Wave (SW) hamiltonian
        for i_site, site in enumerate(mag_str.sites):
            self.add_atom(site, magmom[i_site])
    #end def read_structure
    
    def report(self):
        """General report of the input system
        """
        print("Start reporting...")
        print('\nLattice parameters of the structure: \n', self.lattice)
        print('\nReciprocal lattice parameters of the structure: \n', self.structure.lattice.reciprocal_lattice.matrix)
        print('\nAtom information:')
        print('Label\t S(1/2)\t Fractional coordinate')
        for specie, site in zip(self.labels, self.sites):
            print(specie, '\t' ,site.sw_S, '\t' ,site.frac_coords)
        print("\nPrinting interactions...")
        if self.interactions != []:
            for coup in self.interactions:
                print('='*80)
                print('Interaction atom indices: ', coup.i, coup.j)
                print('Interaction atom labels: ', self.labels[coup.i], self.labels[coup.j])
                print("Interaction matrix: \n", coup.Jij)
                print("Neighbor distances in Angstrom: \n", coup.distances)
                print("Lattice translations of the neighbors: \n", coup.dij)
                print('='*80)
        print("Printing interactions complete!")
        print("\nReport complete!\n")
    #end def report
    
#end class Spinw

if __name__ == "__main__":
    #
    # An example case for CrI3 magnon band structure and magnetization
    #
    structure = """
    """
    # initialize spinwave object
    sw = Spinw()
    
    # define crystal/magnetic structure
    poscar_str = """Cr2 I6
1.00000000000000
7.0022272135628514   -0.0000830945296044    0.0002539098415054
-3.5011860720286840    6.0642296643781766   -0.0005073112232654
0.0007606168683053   -0.0013155215871961   21.4262704851489616
Cr   I
2     6
Direct
0.5550912950971572  0.7775456080084417  0.0000000178079917
0.8884518592258915  0.4442258981429584  0.0000000397832812
0.2217757846090845  0.4709762947198081  0.9270894200256173
0.2217757590905577  0.7507991542142879  0.0729105273011225
0.5818774346162495  0.1108062278902013  0.9270811745929999
0.5818775550115235  0.4710712413140072  0.0729187886562814
0.8615701222269887  0.1109624027661268  0.0729715974732325
0.8615701901225523  0.7506071729441629  0.9270284343594807
    """
    poscar = Poscar.from_string(poscar_str)
    poscar_filename = "./POSCAR"
    poscar.write_file(poscar_filename)
    sw.read_structure(directory="./", spins=[3,3,0,0,0,0,0,0])
    
    # define magnetic interactions
    
    # Reference values are taken from 2D Mater. 4 (2017) 035002:
    #   * isotropic exchange J is defined as 2.2 meV
    #   * anisotropic exchange lambda is defined as 0.09 meV
    # Negative sign of J denotes that interactions are ferromagnetic.
    # In XXZ hamiltonian, lambda is the anisotropic exchange in z-direction
    # Hence it is the 3rd element of the np.diag([0,0,-0.09]) vector below.
    # For lambda, the convention is to use negative sign for the favored 
    # anisotropy direction and use zero for the unfavorable directions.
    # rmax = 4.5 A means only first nearest neighbors for this example
    
    sw.add_coupling(('Cr1', 'Cr2'), -2.2, rmax=4.5)
    sw.add_coupling(('Cr1', 'Cr2'), np.diag([0,0,-0.09]), rmax=4.5)
    
    # plot magnon band structure
    # compare to 2D Mater. 4 (2017) 035002
    ax = plt.figure().gca()
    sw.get_band_structure(ax, [0.0,0.0,0.0], [0.333333, 0.3333333, 0.0], num_k=100, normalize = True)
    figname = './band_line.pdf'
    plt.xlabel('k*l')
    plt.savefig(figname, format='pdf')
    
    # plot magnetization curve 
    sw.magnetization(kgrid=[20,20,1], tmin = 1, tmax = 100, dt = 1)
    os.remove(poscar_filename)
