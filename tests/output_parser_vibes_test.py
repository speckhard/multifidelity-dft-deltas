""""Output parser tests."""

import pytest
import os
import sys
import ase
import ast
import ase.db
import ase.calculators.aims
import csv

# Append path so that we can access packages that are above this directory.
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import parsing.output_parser_vibes


@pytest.fixture
def atomic_data_csv(tmp_path):
    """Creates a temporary CSV file with the atomic data."""
    csv_content = """Atomic number,Element,EA_half,IP_half,EA_delta,IP_delta,HOMO,LUMO,rs,s index,rp,p index,rd,d index,rf,f index
1,H,-0.69804885,12.42399219,-1.727702621,0,-6.3316859,-6.3316859,0.5521,1,1.3786,3,1.058,7,0,0
6,C,-1.46237959,11.49493763,-1.62667163,11.61299938,-6.05205467,-3.50634355,0.6409,2,0.6314,3,1.8064,11,0,0
73,Ta,-0.7735268,7.09339716,-0.871430425,7.319981845,-4.24775361,-2.54556214,1.4464,35,1.9706,41,0.8411,36,0.2336,28"""
    
    csv_file = tmp_path / "atomic_data.csv"
    csv_file.write_text(csv_content)
    return str(csv_file)


@pytest.fixture
def monomers_data_csv(tmp_path):
    """Creates a temporary CSV file with the monomers data."""
    # We create a dataset that matches the C4Ta4 binary (precision=0, k_dens=8, func=pbe)
    # Note: We add explicit 'pbe' rows to ensure they are picked up.
    csv_content = """chem_formula,category,functional,precision_level,k_point_density,min_atom_num,total_energy,num_atoms,volume,bandstructure_gap,gamma_gap
C_dia,monomers_unrelaxed,pbe,0,8,6,-100.0,1,10.0,1.0,1.2
Ta_bcc,monomers_unrelaxed,pbe,0,8,73,-5000.0,1,20.0,0.0,0.0
C_bad,monomers_unrelaxed,lda,0,8,6,-110.0,1,10.0,1.1,1.3
Ta_bad,monomers_unrelaxed,pbe,1,8,73,-5005.0,1,20.0,0.0,0.0"""
    csv_file = tmp_path / "aims_monomers.csv"
    csv_file.write_text(csv_content)
    return str(csv_file)


def create_mock_output_parser(
        input_paths_txt_file='tests/data/output_parser_vibes/one_path.txt',
        save_directory=None,
        ):  # Added this arg
    """Create mock output parser"""
    parse_obj = parsing.output_parser_vibes.OutputParser(
        input_paths_txt_file, 
        save_directory) # Pass it here
    return parse_obj


def test_get_valence_features(tmpdir):
    """Test extraction of valence electron statistics."""
    # No external CSVs needed for this test as it uses mendeleev
    parse_obj = create_mock_output_parser(save_directory=tmpdir)
    
    # Create C4Ta4. 
    # Carbon (Z=6) has 4 valence electrons.
    # Tantalum (Z=73) has 5 valence electrons (Group 5).
    atoms = ase.Atoms('C4Ta4')
    
    features = parse_obj.get_valence_features(atoms)
    
    # Max: 5 (Ta), Min: 4 (C)
    assert features['max_valence_electrons'] == 5
    assert features['min_valence_electrons'] == 4
    
    # Mean: (4 + 5) / 2 = 4.5
    assert features['mean_valence_electrons'] == 4.5
    
    # MAD: (|4-4.5| + |5-4.5|) / 2 = (0.5 + 0.5) / 2 = 0.5
    assert features['mad_valence_electrons'] == 0.5


def test_get_basis_features(tmpdir):
    """Test extraction of basis function statistics from Pickle."""
    parse_obj = create_mock_output_parser(
        save_directory=tmpdir
    )

    atoms = ase.Atoms('LiF')

    # We use binary_precision=0.
    # Logic in script: Prec 0 maps to -> numerical_setting='light', basis_set_size=0
    # Our fixture has:
    # ('light', 6, 0): 14
    # ('light', 73, 0): 73

    features = parse_obj.get_basis_features(atoms, binary_precision=0)

    assert features['max_basis_functions'] == 8
    assert features['min_basis_functions'] == 2
    assert features['mean_basis_functions'] == (2 + 8) / 2 # 28.0
    assert features['mad_basis_functions'] == 3


def test_get_monomer_features(tmpdir):
    """Test extraction and calculation of monomer features."""
    parse_obj = create_mock_output_parser(
        save_directory=tmpdir
    )
    
    # C4Ta4 -> C (6) and Ta (73)
    atoms = ase.Atoms('C4Ta4')
    
    # We ask for features matching our CSV dummy data: prec=0, k_dens=8
    features = parse_obj.get_monomer_features(
        atoms, precision_level=0, k_point_density=8
    )
    
    # Expected Values:
    # C: E=-100, Vol=10
    # Ta: E=-5000, Vol=20
    print(f'features are {features}')
    expected_max_total_energy_per_atom = -1035.31449675517/4
    expected_min_total_energy_per_atom = -437225.246838012/2

    assert features['max_monomer_total_energy_per_atom'] == expected_max_total_energy_per_atom
    assert features['min_monomer_total_energy_per_atom'] == expected_min_total_energy_per_atom

    # Mean Total Energy = (-100 + -5000) / 2 = -2550
    assert features['mean_monomer_total_energy_per_atom'] == (expected_max_total_energy_per_atom+expected_min_total_energy_per_atom)/2
    
    assert round(features['max_monomer_volume_per_atom'] - 36.6502138178759/2, 7) == 0
    assert round(features['min_monomer_volume_per_atom'] - 46.6564940090217/4, 6) == 0


    # # Test Filtering: 
    # # If we ask for precision=1, we should only get the 'Ta_bad' row (if we had C matching precision 1)
    # # But in our CSV, only Ta has precision 1. C has no match for precision 1.
    # # So stats will be just based on Ta.
    # features_p1 = parse_obj.get_monomer_features(
    #     atoms, precision_level=1, k_point_density=8
    # )


def test_get_atomic_features(tmpdir):
    """Test calculation of atomic features for C4Ta4."""
    # Create the parser with the temp CSV
    parse_obj = create_mock_output_parser(
        save_directory=tmpdir
    )
    
    # Create a dummy ASE Atoms object for C4Ta4
    # Carbon (Z=6), Tantalum (Z=73)
    atoms = ase.Atoms('C4Ta4')
    
    # Calculate features
    features = parse_obj.get_atomic_features(atoms)
    
    # --- Assertions ---
    # Based on:
    # C  EA_half = -1.46237959
    # Ta EA_half = -0.7735268
    
    # Max: -0.7735268
    assert round(features['max_atomic_EA_half'] - (-0.7735268), 6) == 0.0
    
    # Min: -1.46237959
    assert round(features['min_atomic_EA_half'] - (-1.46237959), 6) == 0.0
    
    # Mean: (-1.46237959 + -0.7735268) / 2 = -1.117953195
    assert round(features['mean_atomic_EA_half'] - (-1.1179532), 6) == 0.0
    
    # MAD: (| -1.4623... - -1.1179... | + | -0.7735... - -1.1179... |) / 2
    # MAD = 0.344426395
    assert round(features['mad_atomic_EA_half'] - 0.3444264, 6) == 0.0

    # Check a different property to be sure (IP_half)
    # C  IP = 11.49493763
    # Ta IP = 7.09339716
    assert round(features['max_atomic_IP_half'] - 11.49493763, 6) == 0.0


def test_read_geometry(tmpdir):
    """Test reading the geometry file from a folder."""
    path = (
        'tests/data/test/expansion_1dot05/'
        'pbe/tight/minimal/atomic_zora/4/RhSi/')
    parse_obj = create_mock_output_parser(save_directory=tmpdir)
    atoms_object = parse_obj.read_geometry(path)
    assert atoms_object.get_chemical_formula() == 'RhSi'
    assert atoms_object.get_positions()[0][0] == 0
    assert atoms_object.get_positions()[0][1] == 0
    assert atoms_object.get_positions()[0][2] == 0

    assert round(
        atoms_object.get_cell()[0][0] - 1.0163963568148535, 6) == 0.0
    assert atoms_object.get_cell()[0][1] == 0.0
    assert atoms_object.get_cell()[0][2] == 0.0


def test_read_geometry_relaxed(tmpdir):
    """Test reading the geometry file from a folder."""
    path = (
        'tests/data/test/expansion_1dot05/'
        'pbe/tight/minimal/atomic_zora/4/RhSi/')
    parse_obj = create_mock_output_parser(save_directory=tmpdir)
    atoms_object = parse_obj.read_geometry(path, relaxation=True)
    assert atoms_object.get_chemical_formula() == 'RhSi'
    assert atoms_object.get_positions()[0][0] == 0
    assert atoms_object.get_positions()[0][1] == 0
    assert atoms_object.get_positions()[0][2] == 0

    assert round(
        atoms_object.get_cell()[0][0] - 1.0163963568148535, 6) == 0.0
    assert atoms_object.get_cell()[0][1] == 0.0
    assert atoms_object.get_cell()[0][2] == 0.0


def test_get_ase_params(tmpdir):
    """Test getting ase parameters from parameters.ase file."""

    path = (
        'tests/data/test/expansion_1dot05/' +
        'pbe/tight/minimal/atomic_zora/4/RhSi/')

    parse_obj = create_mock_output_parser(save_directory=tmpdir)
    params_dict = parse_obj.read_ase_params(path)
    assert params_dict['k_grid'][0] == 25
    assert params_dict['k_grid'][1] == 25
    assert params_dict['k_grid'][2] == 25
    assert params_dict['relativistic'][0] == 'atomic_zora'
    assert params_dict['mixer'] == 'pulay'


# def test_write_binaries_input_from_geometry():
#     """Write to ASE db from ASE params and geometry files.

#     This is a test to see if we can write to a new ASE
#     database. We first read into an ASE Atoms object from
#     a geomtry input (aims) and parameters.ase file.
#     We then attach an FHI-aims caluclator. Finally we save
#     the Atoms object and category = 'binaries_input' to the
#     db. We convert the db to a json format for visual inspection.
#     We also read back the data in the db to assert that
#     the data in there is as we expect to be.
#     """
#     # First we have to note where we import the geometry file.
#     path = (
#         '/home/speckhard/Documents/theory/errorbar_project/' +
#         'error_modelling/tests/data/test/expansion_None/pbe/light/' +
#         'minimal/atomic_zora/8/C4Ta4')
#     parse_obj = create_mock_output_parser([path])
#     params_dict = parse_obj.read_ase_params(path)
#     atoms_object = parse_obj.read_geometry(path)
#     # Before we go saving this data to a database let's 
#     # make sure we have the right data. First
#     # check the params.
#     assert params_dict['k_grid'][0] == 12
#     assert params_dict['k_grid'][1] == 12
#     assert params_dict['k_grid'][2] == 12
#     # Now check the chemical formula.
#     assert atoms_object.get_chemical_formula() == 'C4Ta4'
#     # Now check the positions are good.
#     assert round(
#         atoms_object.get_positions()[0][0] - 4.4530000000000003/2, 6) == 0.0
#     assert round(
#         atoms_object.get_positions()[0][1] - 4.4530000000000003/2, 6) == 0.0
#     assert round(
#         atoms_object.get_positions()[0][2] - 4.4530000000000003/2, 6) == 0.0
#     # Now check that we have the right lattice vectors.
#     assert round(
#         atoms_object.get_cell()[0][0] - 4.4530000000000003, 6) == 0.0

#     category = 'binaries_input'
#     calc = ase.calculators.aims.Aims(
#               tier='tier1',
#               label=path+'/',
#               **params_dict)
#     # Attach a calculator to the atoms object.
#     # atoms_object = atoms_object.set_calculator(calc)
#     # Connect to a fake ase db that is located in the same sport    
#     dbcon = ase.db.connect(path + '/' + 'test.db')
#     dbcon.write(
#         atoms_object, attach_calculator=True,
#         category=category, tiers='tier2', name='CTa4',
#         functional='pbe', basis_set='light',
#         relativistic_treatment='atomic_zora',
#         k_point_density=8)
#     dbcon.write(
#         atoms_object, attach_calculator=True,
#         category='binaries_non_input', tiers='tier2', name='CTa4',
#         functional='pbe', basis_set='light',
#         relativistic_treatment='atomic_zora',
#         k_point_density=8)
#     # Now convert the db to a json for visual inspection.
#     os.system(
#         'ase db ' +
#         path + '/test.db' + ' --insert-into ' +
#         path + '/test_db.json'
#         )
#     # Alright now let's read the data from the database and make sure
#     # sure that we can decipher it correctly.
    # rows = dbcon.select(['category='+'binaries_input'])
    # # Let's grab the first row.
    # for row in rows:
    #     assert row['category'] == 'binaries_input'
#     atom_object = row.toatoms()
#     assert atoms_object.get_chemical_formula() == 'C4Ta4'
#     # Now check the positions are good.
#     assert round(
#         atoms_object.get_positions()[0][0] - 4.4530000000000003/2, 6) == 0.0
#     assert round(
#         atoms_object.get_positions()[0][1] - 4.4530000000000003/2, 6) == 0.0
#     assert round(
#         atoms_object.get_positions()[0][2] - 4.4530000000000003/2, 6) == 0.0
#     # Now check that we have the right lattice vectors.
#     assert round(
#         atoms_object.get_cell()[0][0] - 4.4530000000000003, 6) == 0.0    

def test_write_db_and_csv(tmp_path):
    """Test writing to ASE database and CSV file with UID and new features."""
    test_dir = tmp_path / 'test_output'
    test_dir.mkdir()
    
    # Path to a known valid dataset from previous tests
    path = (
        'tests/data/output_parser_vibes/expansion_None/pbe/light/minimal/'
        'atomic_zora/8/C4Ta4_ICSD_234/submission_C4Ta4.sh')
    
    # Create a dummy paths file
    paths_file = test_dir / 'paths.txt'
    with open(paths_file, 'w') as f:
        f.write(path + '\n')
        
    # Initialize parser with this file
    parse_obj = parsing.output_parser_vibes.OutputParser(
        input_paths_txt_file=str(paths_file),
        save_directory=str(test_dir)
    )
    
    # Execute
    parse_obj.write_all_path_data()
    
    # 1. Verify Database
    db_files = list(test_dir.glob('relaxations_*.db'))
    assert len(db_files) == 1
    db_path = db_files[0]

    with ase.db.connect(str(db_path)) as db:
        rows = list(db.select())
        assert len(rows) == 1
        row = rows[0]
        assert row.formula == 'Ta4C4'
        assert row.uid == 1
        
        # Verify that the new Valence/Basis stats are in the key_value_pairs of the DB row
        # Note: Basis will be None/Empty because we didn't pass a valid pickle path in this specific test,
        # but Valence should be calculated via mendeleev.
        assert hasattr(row, 'max_valence_electrons') or 'max_valence_electrons' in row.key_value_pairs
        
    # 2. Verify CSV
    csv_files = list(test_dir.glob('parsed_data_*.csv'))
    assert len(csv_files) == 1
    csv_path = csv_files[0]
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 1
        csv_row = rows[0]
        
        # Standard checks
        assert csv_row['uid'] == '1'
        assert csv_row['compound_name'] == 'C4Ta4'
        
        # CHECK FOR NEW COLUMNS
        assert 'max_valence_electrons' in csv_row
        assert 'min_valence_electrons' in csv_row
        assert 'max_basis_functions' in csv_row 
        assert 'min_basis_functions' in csv_row
        
        # Check Valence Values (C=4, Ta=5 -> Max 5)
        assert float(csv_row['max_valence_electrons']) == 5.0
        assert float(csv_row['min_basis_functions']) == 8.0


def test_parse_detailed_energies(tmp_path):
    """Test parsing of detailed energy components from aims.out."""
    test_dir = tmp_path / 'detailed_energies_test'
    test_dir.mkdir()
    
    # Create the directory structure that OutputParser expects
    # path/relaxation/calculation/aims.out
    sim_path = test_dir / 'sim_path'
    calc_path = sim_path / 'relaxation' / 'calculation'
    calc_path.mkdir(parents=True)
    
    aims_out_content = """
  Total energy components:
  | Sum of eigenvalues            :        -495.58572688 Ha       -13485.57376675 eV
  | XC energy correction          :         -76.30408236 Ha        -2076.33972392 eV
  | XC potential correction       :          98.38402451 Ha         2677.16551970 eV
  | Free-atom electrostatic energy:        -406.23542915 Ha       -11054.22846026 eV
  | Hartree energy correction     :          -1.96131750 Ha          -53.37016451 eV
  | Entropy correction            :          -0.00000375 Ha           -0.00010209 eV
  | ---------------------------
  | Total energy                  :        -881.70253137 Ha       -23992.34659575 eV
  | Total energy, T -> 0          :        -881.70253512 Ha       -23992.34669784 eV  <-- do not rely on this value for anything but (periodic) metals
  | Electronic free energy        :        -881.70253887 Ha       -23992.34679992 eV

  Derived energy quantities:
  | Kinetic energy                :         884.83074915 Ha        24077.46973249 eV
  | Electrostatic energy          :       -1690.22919816 Ha       -45993.47660431 eV
  | Energy correction for multipole
  | error in Hartree potential    :            0.00997186 Ha            0.27134819 eV
  | Sum of eigenvalues per atom                                 :       -2247.59562779 eV
  | Total energy (T->0) per atom                                :       -3998.72444964 eV  <-- do not rely on this value for anything but (periodic) metals
  | Electronic free energy per atom                             :       -3998.72446665 eV
  What follows are estimated values for band gap, HOMO, LUMO, etc.
  | They are estimated on a discrete k-point grid and not necessarily exact.
  | For converged numbers, create a DOS and/or band structure plot on a denser k-grid.

  Highest occupied state (VBM) at      -5.84118675 eV (relative to internal zero)
  | Occupation number:       1.98939287
  | K-point:        1 at     0.000000     0.000000     0.000000 (in units of recip. lattice)

  Lowest unoccupied state (CBM) at      -4.81117215 eV (relative to internal zero)
  | Occupation number:       0.01253503
  | K-point:        1 at     0.000000     0.000000     0.000000 (in units of recip. lattice)

  ESTIMATED overall HOMO-LUMO gap:       1.03001460 eV between HOMO at k-point 1 and LUMO at k-point 1
  | This appears to be a direct band gap.
  The gap value is above 0.2 eV. Unless you are using a very sparse k-point grid,
  this system is most likely an insulator or a semiconductor.

  | Chemical Potential                          :     -5.31778095 eV
    """
    
    with open(calc_path / 'aims.out', 'w') as f:
        f.write(aims_out_content)
        
    # Mock parser
    # input_paths_txt_file is required but not used for this specific method test 
    # if we call the method directly, but __init__ reads it.
    paths_file = test_dir / 'paths.txt'
    with open(paths_file, 'w') as f:
        f.write(str(sim_path) + '\n')

    parse_obj = parsing.output_parser_vibes.OutputParser(
        input_paths_txt_file=str(paths_file),
        save_directory=str(test_dir)
    )
    
    # Call the method directly
    energy_dict = parse_obj.parse_detailed_energies(str(sim_path), relaxation=True)
    
    # Assertions based on the text content
    assert energy_dict['sum_eigenvalues'] == -13485.57376675
    assert energy_dict['xc_energy_correction'] == -2076.33972392
    assert energy_dict['xc_potential_correction'] == 2677.16551970
    assert energy_dict['free_atom_electrostatic_energy'] == -11054.22846026
    assert energy_dict['hartree_energy_correction'] == -53.37016451
    assert energy_dict['entropy_correction'] == -0.00010209
    assert energy_dict['total_energy_T0'] == -23992.34669784
    assert energy_dict['kinetic_energy'] == 24077.46973249
    assert energy_dict['electrostatic_energy'] == -45993.47660431
    assert energy_dict['multipole_correction'] == 0.27134819
    assert energy_dict['sum_eigenvalues_per_atom'] == -2247.59562779
    assert energy_dict['total_energy_T0_per_atom'] == -3998.72444964
    assert energy_dict['electronic_free_energy_per_atom'] == -3998.72446665
    assert energy_dict['vbm'] == -5.84118675
    assert energy_dict['cbm'] == -4.81117215
    assert energy_dict['homo_lumo_gap'] == 1.03001460
    assert energy_dict['chemical_potential'] == -5.31778095



def test_parse_aims_out(tmpdir):
    """Test how we can set the calculator on our atoms object."""
    # Old data from bjorn. I think this path doesn't contain the entire
    # aims.out file which is the reason it fails, let's try a different one.
    # path = ('tests/data/output_parser_vibes/73_Ta')
    path = 'tests/data/output_parser_vibes/O4Si2_ICSD_99/'

    parse_obj = create_mock_output_parser(save_directory=tmpdir)
    # atoms_object = parse_obj.read_geometry(path)
    # print(parse_obj.tier_map)
    aims_dict = parse_obj.parse_aims_out(path)
    print('about to get eigenvalues.')
    total_energy = aims_dict['total_energy']
    print(total_energy)
    assert total_energy == -23992.3465957474
    # # Hartree value times hartree in eV.
    # hartree = 27.21138624598853
    # assert round(eigen[0] + 2673.8175*hartree, 2) == 0.0
    # assert round(eigen[-1] - 0.071241*hartree, 2) == 0.0


def test_parse_aims_out_problem_file(tmpdir):
    """Test parsing aims out file on a file that gave errors in prod."""
    # Old data from bjorn. I think this path doesn't contain the entire
    # aims.out file which is the reason it fails, let's try a different one.
    # path = ('tests/data/output_parser_vibes/73_Ta')
    path = 'tests/data/output_parser_vibes/pbe/light/minimal/atomic_zora/8/SiSr8_ICSD_1234'

    parse_obj = create_mock_output_parser(save_directory=tmpdir)
    # atoms_object = parse_obj.read_geometry(path)
    # print(parse_obj.tier_map)
    aims_dict = parse_obj.parse_aims_out(path)
    print('about to get eigenvalues.')
    print(f'aims_dict: {aims_dict}')
    total_energy = aims_dict['total_energy']
    print(total_energy)
    assert total_energy == -729232.014782314


def test_get_volume(tmpdir):
    """Test getting the volume from a simulation."""
    # Old data from bjorn.
    path = 'tests/data/output_parser_vibes/73_Ta'

    parse_obj = create_mock_output_parser(save_directory=tmpdir)
    atoms_obj = parse_obj.read_geometry(path)
    volume = parse_obj.get_volume(atoms_obj)
    # TODO: @dts, should make sure this value
    # from ASE is correct.
    assert volume != 0.0
    assert isinstance(volume, float)


def test_get_num_atoms(tmpdir):
    """Test Getting the number of atoms from sim."""
    path = 'tests/data/output_parser_vibes/73_Ta'
    parse_obj = create_mock_output_parser(save_directory=tmpdir)
    atoms_obj = parse_obj.read_geometry(path)
    num_atoms = parse_obj.get_num_atoms(atoms_obj)
    assert num_atoms == 8


# def test_get_min_max_atom_name(tmpdir):
#     """Test getting min/max atom name."""
#     parse_obj = create_mock_output_parser(save_directory=tmpdir)
#     compound_name = 'Mg4O4'
#     min_atom, max_atom, num_atoms = parse_obj.get_min_max_atom_name(
#         compound_name)
#     assert min_atom == 12
#     assert max_atom == 8
#     assert num_atoms == 8


# def test_read_bandgap():
#     """Test reading the bandgap."""
#     path = ('tests/data/output_parser_vibes/Li8S4')
#     parse_obj = create_mock_output_parser()
#     calc = parse_obj.set_calculator(path)
#     HOMO_LUMO_gap, gap_bandstructure = parse_obj.get_bandgap_data(calc)
#     assert isinstance(gap_bandstructure, float)
#     assert isinstance(HOMO_LUMO_gap, float)


# def test_get_gamma_gap():
#     """Test ability to get the gamma gap."""
#     path = ('tests/data/output_parser_vibes/73_Ta')
#     parse_obj = create_mock_output_parser()
#     calc = parse_obj.set_calculator(path)
#     gamma_gap = parse_obj.get_gamma_gap(calc)
#     assert isinstance(gamma_gap, float)


def test_parse_path_expansion():
    """Test ability to parse settings used in a calc from pathname."""
    # path = ('tests/data/output_parser/73_Ta')
    # parse_obj = create_mock_output_parser([path])
    path = ('tests/data/test/expansion_1dot05/'
            'pbe/tight/minimal/atomic_zora/4/'
            'RhSi/submission_RhSi.sh')

    settings_dict = parsing.output_parser_vibes.OutputParser.parse_path(
            path, ICSD_number=False, expansion=True)

    assert settings_dict['expansion'] == 'expansion_1dot05'
    assert settings_dict['functional'] == 'pbe'
    assert settings_dict['num_setting'] == 'tight'
    assert settings_dict['basis_size'] == 'minimal'
    assert settings_dict['rel_setting'] == 'atomic_zora'
    assert settings_dict['k_point_density'] == 4
    assert settings_dict['compound_name'] == 'RhSi'


def test_parse_path_ICSD_number():
    """Test ability to parse settings used in a calc from pathname."""
    # path = ('tests/data/output_parser/73_Ta')
    # parse_obj = create_mock_output_parser([path])
    path = ('tests/data/test/'
            'pbe/tight/minimal/atomic_zora/4/'
            'RhSi_ISCD_899/submission_RhSi.sh')

    settings_dict = parsing.output_parser_vibes.OutputParser.parse_path(
            path, ICSD_number=True, expansion=False)

    assert settings_dict['ICSD_number'] == 899
    assert settings_dict['functional'] == 'pbe'
    assert settings_dict['num_setting'] == 'tight'
    assert settings_dict['basis_size'] == 'minimal'
    assert settings_dict['rel_setting'] == 'atomic_zora'
    assert settings_dict['k_point_density'] == 4
    assert settings_dict['compound_name'] == 'RhSi'


def test_gather_all_path_data(tmpdir):
    """Test getting all data from path stored to dict."""
    test_dir = tmpdir / 'test'
    test_dir.mkdir()
    
    path = (
        'tests/data/output_parser_vibes/expansion_None/pbe/light/minimal/'
        'atomic_zora/8/C4Ta4_ICSD_234/submission_C4Ta4.sh')
        
    # Pass the atomic data csv to the parser
    parse_obj = create_mock_output_parser(
        save_directory=test_dir
    )
    
    data_dict, _, relaxed_ase_atoms_obj = parse_obj.gather_all_path_data(path)
    
    assert relaxed_ase_atoms_obj.get_chemical_formula() == 'C4Ta4'
    
    # Existing Energy Assertions
    assert round(data_dict['total_energy'] + 1753052.37929678, 7) == 0.0
    assert round(data_dict['aims_free_energy'] + 1753052.4271399, 7) == 0.0 
    assert round(data_dict['total_energy_per_atom']+219131.5474120975, 7) == 0.0
    assert data_dict['binary_precision'] == 0
    
    # --- NEW: Atomic Feature Assertions ---
    # C4Ta4 contains C (EA=-1.462...) and Ta (EA=-0.773...)
    # Min should be -1.46237959
    assert 'min_atomic_EA_half' in data_dict
    assert round(data_dict['min_atomic_EA_half'] - (-1.46237959), 6) == 0.0
    
    # Max should be -0.7735268
    assert 'max_atomic_EA_half' in data_dict
    assert round(data_dict['max_atomic_EA_half'] - (-0.7735268), 6) == 0.0

    # Ensure file paths were recorded
    assert os.path.isfile(parse_obj.paths_finished_correctly)
    with open(parse_obj.paths_finished_correctly, "r") as txt_file:
        paths = [line.rstrip('\n') for line in txt_file]
        assert path in paths


def test_gather_all_path_data_expired_gen(tmpdir):
    """Test logic for an expired path from gen queue."""
    path = (
        'tests/data/output_parser_vibes/expansion_None/lda/light/minimal/'
        'atomic_zora/8/C4Ta4_ICSD_325/submission_C4Ta4.sh')

    test_dir = tmpdir / 'test'
    test_dir.mkdir()

    parse_obj = create_mock_output_parser(save_directory=test_dir)

    _, _, _ = parse_obj.gather_all_path_data(path)
    # Ensure that the script was written to paths increase charge mixing param.
    print(parse_obj.paths_decrease_charge_mix)
    with open(parse_obj.paths_decrease_charge_mix, 'r') as txt_file:
        paths = [line.rstrip('\n') for line in txt_file]
        print(paths)
        assert path in paths


def test_gather_all_path_data_scf_unconverged(tmpdir):
    """Test logic for an expired path from gen queue."""
    path = (
        'tests/'
        'data/output_parser_vibes/expansion_None/lda/'
        'light/minimal/atomic_zora/8/'
        'C4Ta4_ICSD_325/submission_C4Ta4.sh')
    test_dir = tmpdir / 'test'
    test_dir.mkdir()
    parse_obj = create_mock_output_parser(save_directory=test_dir)
    _, _, _ = parse_obj.gather_all_path_data(path)
    # Ensure that the script was written to paths increase charge mixing param.
    with open(parse_obj.paths_decrease_charge_mix, "r") as txt_file:
        paths = [line.rstrip('\n') for line in txt_file]
        assert path in paths


def test_get_band_file_data(tmpdir):
    """Test getting band file data."""
    path = (
        'tests/data/output_parser_vibes/expansion_None/pbe/light/'
        'minimal/atomic_zora/8/Al16F48_ICSD_999/bandstructure/submission.sh')
    # Real path looks like:
    # /scratch-emmy/projects/bep00098/aflow_binaries_vibes_1000_1999/pbe/light/
    # minimal/atomic_zora/8/Al16F48_ICSD_72174/bandstructure/
    test_dir = tmpdir / 'test'
    test_dir.mkdir()
    # Create a paths list:
    path_list = test_dir / 'path_list.txt'
    with open(test_dir / 'path_list.txt', 'w') as fd:
        fd.write(path)
    parse_obj = parsing.output_parser_vibes.OutputParser(
        input_paths_txt_file=path_list, save_directory=test_dir, bandstructure_calculations=True)
    data_dict, og_ase_atoms_obj, relaxed_ase_atoms_obj = parse_obj.gather_all_path_data(path)
    print(f'data dict is : {data_dict}')
    # Ensure that the script was written to paths increase charge mixing param.
    assert round(data_dict['total_energy']+236399.220803096, 7) == 0


def test_get_path_list(tmpdir):
    """Test translating a txt file of paths to a list."""
    test_dir = tmpdir / 'test'
    test_dir.mkdir()
    paths_txt_file = 'tests/data/output_parser_vibes/three_paths.txt'
    parse_obj = create_mock_output_parser(
        input_paths_txt_file=paths_txt_file, save_directory=test_dir)
    expected_path_list = [
        'tests/data/output_parser_vibes/expansion_None/pbe/light/minimal/'
        'atomic_zora/8/C4Ta4_ICSD_234/submission_C4Ta4.sh',
        'tests/data/output_parser_vibes/expansion_None/pbe/light/minimal/'
        'atomic_zora/8/Ti4O6_ICSD_233/submission_Ti4O6.sh',
        'tests/data/output_parser_vibes/expansion_None/lda/light/minimal/'
        'atomic_zora/8/Ti4O6_ICSD_326/submission_Ti4O6.sh']
    path_list = parse_obj.path_list
    assert len(expected_path_list) == 3
    for i in range(len(expected_path_list)):
        assert expected_path_list[i] == path_list[i]


def test_check_calc_finished():
    """Test whether we sim finished nicely.

    Feed in a file that has have a nice day
    in the last 50 lines. See if the method
    can find return true. Feed another file
    without have a nice day and ensure
    False is returned.
    """
    finished_path = (
        'tests/data/output_parser_vibes/C4Ta4')
    # Use namespace name to call static method.
    calc_finished_bool, calc_expired_bool, out_of_npl_bool, scf_bool = parsing.output_parser_vibes.OutputParser.check_calc_finished(
        finished_path)
    assert calc_finished_bool is True
    assert calc_expired_bool is False
    assert scf_bool is False
    assert out_of_npl_bool is False

    # Now try a file that doesn't have have a nice day.
    unfinished_path = 'tests/data/output_parser_vibes/C4Ta4_not_finished'
    # Use namespace name to call static method.
    calc_finished_bool, calc_expired_bool, out_of_npl_bool, scf_bool = parsing.output_parser_vibes.OutputParser.check_calc_finished(
        unfinished_path)
    assert calc_finished_bool is False
    assert calc_expired_bool is True
    assert scf_bool is False
    assert out_of_npl_bool is False

    # Now try a file where SCF is not converged.
    unfinished_path = 'tests/data/output_parser_vibes/C4Ta4_scf_unconverged'
    # Use namespace name to call static method.
    calc_finished_bool, calc_expired_bool, out_of_npl_bool, scf_bool = parsing.output_parser_vibes.OutputParser.check_calc_finished(
        unfinished_path)
    assert calc_finished_bool is False
    assert calc_expired_bool is False
    assert scf_bool is True
    assert out_of_npl_bool is False

    # Now try a file where calc expired bool not due to being out of SCF.
    unfinished_parent = 'tests/data/output_parser_vibes/C4Ta4_expired_general'
    # Use namespace name to call static method.
    calc_finished_bool, calc_expired_bool, out_of_npl_bool, scf_bool = parsing.output_parser_vibes.OutputParser.check_calc_finished(
        unfinished_parent)
    assert calc_finished_bool is False
    assert calc_expired_bool is True
    assert scf_bool is False
    assert out_of_npl_bool is False
    # Now try a file where calc expired bool not due to being out of SCF.
    unfinished_parent = 'tests/data/output_parser_vibes/C4Ta4_out_of_npl'
    # Use namespace name to call static method.
    calc_finished_bool, calc_expired_bool, out_of_npl_bool, scf_bool = parsing.output_parser_vibes.OutputParser.check_calc_finished(
        unfinished_parent)
    assert calc_finished_bool is False
    assert calc_expired_bool is False
    assert scf_bool is False
    assert out_of_npl_bool is True



def test_get_queue(tmpdir):
    """Test method to get queue name from sub script."""
    test_dir = tmpdir / 'test'
    test_dir.mkdir()
    parse_obj = create_mock_output_parser(save_directory=test_dir)
    unfinished_parent = 'tests/data/output_parser_vibes/C4Ta4_expired_general'
    submission_path_expired_gen = (
        unfinished_parent + '/' + 'submission_C4Ta4.sh')
    queue = parse_obj.get_queue(
        submission_path_expired_gen)
    assert queue == 'general'
    finished_parent = 'tests/data/output_parser_vibes/C4Ta4'
    submission_path_short = finished_parent + '/' + 'submission_C4Ta4.sh'
    queue = parse_obj.get_queue(
            submission_path_short)
    assert queue == 'short'


def test_add_expired_path(tmpdir):
    """Test method to add paths of sims that didn't exit nicely."""
    # Create mock output parse object.
    test_dir = tmpdir / 'test'
    test_dir.mkdir()
    parse_obj = create_mock_output_parser(save_directory=test_dir)
    submission_path = 'test_path/submission_XY.txt'
    # Tests paths to resubmit due to time expiration.
    parse_obj.add_expired_path(submission_path)
    # Check if the file contains the submission path.
    with open(parse_obj.paths_to_resubmit, "r") as txt_file:
        paths = [line.rstrip('\n') for line in txt_file]
        assert submission_path in paths


def test_add_misbehaving_path(tmpdir):
    """Test method to add paths of sims that didn't exit nicely."""
    test_dir = tmpdir / 'test'
    test_dir.mkdir()
    # Create mock output parse object.
    parse_obj = create_mock_output_parser(save_directory=test_dir)
    submission_path = 'test_path/submission_XY.txt'
    # Tests paths to resubmit due to time expiration.
    parse_obj.add_misbehaving_path(submission_path)
    # Check if the file contains the submission path.
    with open(parse_obj.paths_misbehaving, "r") as txt_file:
        paths = [line.rstrip('\n') for line in txt_file]
        assert submission_path in paths


def test_add_increase_charge_mix_path(tmpdir):
    """Test method to add paths of sims that didn't exit nicely."""
    # Create mock output parse object.
    test_dir = tmpdir / 'test'
    test_dir.mkdir()
    parse_obj = create_mock_output_parser(save_directory=test_dir)
    submission_path = 'test_path/submission_XY.txt'
    # Tests paths to resubmit due to time expiration.
    parse_obj.add_increase_charge_mix_path(submission_path)
    # Check if the file contains the submission path.
    with open(parse_obj.paths_increase_charge_mix, "r") as txt_file:
        paths = [line.rstrip('\n') for line in txt_file]
        assert submission_path in paths


def test_check_sim_time_lim_true():
    """Test whether we can find error msg in djob err."""
    expected_bool = True
    expected_out_of_npl_bool = False
    # Now try a file that doesn't have have a nice day.
    unfinished_path = 'tests/data/output_parser_vibes/C4Ta4_not_finished'
    # Use namespace name to call static method.
    expired_time_bool, out_of_npl_bool = parsing.output_parser_vibes.OutputParser.check_sim_time_lim(
        unfinished_path)
    assert expired_time_bool is expected_bool
    assert out_of_npl_bool is expected_out_of_npl_bool


def test_check_sim_time_lim_for_emtpy_dir():
    """Test function when there is no djob.err file since sim didn't run."""
    expected_bool = False
    expected_out_of_npl_bool = False
    # Now try a file that doesn't have have a nice day.
    unfinished_path = 'tests/data/output_parser_vibes/empty_error_files'
    # Use namespace name to call static method.
    expired_time_bool, out_of_npl_bool = parsing.output_parser_vibes.OutputParser.check_sim_time_lim(
        unfinished_path)
    assert expired_time_bool is expected_bool
    assert out_of_npl_bool is expected_out_of_npl_bool


def test_add_decrease_charge_mix_path(tmpdir):
    """Test method to add paths of sims that didn't exit nicely."""
    # Create mock output parse object.
    test_dir = tmpdir / 'test'
    test_dir.mkdir()
    parse_obj = create_mock_output_parser(save_directory=test_dir)
    submission_path = 'test_path/submission_XY.txt'
    # Tests paths to resubmit due to time expiration.
    parse_obj.add_decrease_charge_mix_path(submission_path)
    # Check if the file contains the submission path.
    with open(parse_obj.paths_decrease_charge_mix, "r") as txt_file:
        paths = [line.rstrip('\n') for line in txt_file]
        assert submission_path in paths


def test_check_sim_time_lim_false():
    """Test whether we can find error msg in djob err.
    TODO: we should really combine this and the above
    test with one setup.
    """
    expected_bool = False
    expected_out_of_npl = False
    # Now try a file that doesn't have have a nice day.
    unfinished_path = 'tests/data/output_parser_vibes/C4Ta4'
    # Use namespace name to call static method.
    expired_time_bool, out_of_npl = parsing.output_parser_vibes.OutputParser.check_sim_time_lim(
        unfinished_path)
    assert expired_time_bool is expected_bool
    assert expected_out_of_npl is out_of_npl

    expected_out_of_npl = True
    unfinished_path = 'tests/data/output_parser_vibes/C4Ta4_out_of_npl'
    # Use namespace name to call static method.
    expired_time_bool, out_of_npl = parsing.output_parser_vibes.OutputParser.check_sim_time_lim(
        unfinished_path)
    assert expired_time_bool is expected_bool
    assert expected_out_of_npl is out_of_npl


# TODO: dts, ideally we should add anoter two tests that check
# that the expired time is true and that the test ends up in the
# list to be resubmittted.


def test_check_if_aims_out_exists(tmpdir):
    """Test whether we can find error msg in djob err.
    TODO: we should really combine this and the above
    test with one setup.
    """
    test_dir = tmpdir / 'test'
    test_dir.mkdir()
    expected_bool = True
    # Now try a file that doesn't have have a nice day.
    unfinished_path = 'tests/data/output_parser_vibes/MgO'
    # Use namespace name to call static method.
    parse_obj = create_mock_output_parser(save_directory=test_dir)
    aims_out_bool = parse_obj.check_if_aims_out_exists(
        unfinished_path)
    assert aims_out_bool is expected_bool


def test_get_most_recent_djob_err():
    """Test getting the most recent djob.err file."""
    submission_path = 'tests/data/output_parser_vibes/error_files'
    returned_error_path = parsing.output_parser_vibes.OutputParser.get_most_recent_djob_err(
        submission_path)
    expected_error_path = (
        'djob.err.2342')
    assert returned_error_path == expected_error_path


def test_get_most_recent_djob_err_empty():
    """Test getting the most recent djob.err file when there is none."""
    submission_path = 'tests/data/output_parser_vibes/empty_error_files'
    returned_error_path = parsing.output_parser_vibes.OutputParser.get_most_recent_djob_err(
        submission_path)
    expected_error_path = None
    assert returned_error_path == expected_error_path


def test_check_out_of_npl_when_gathering(tmpdir):
    """Test whether we can find error msg in djob err.
    TODO: we should really combine this and the above
    test with one setup.
    """
    # Now try a file that doesn't have have a nice day.
    test_dir = tmpdir / 'test'
    test_dir.mkdir()
    unfinished_path = '/home/dts/Documents/theory/errorbar_modelling/tests/data/output_parser_vibes/pbe/8/Mgo_ICSD_348/submission.sh'
    # Use namespace name to call static method.
    parse_obj = create_mock_output_parser(save_directory=test_dir)
    parse_obj.gather_all_path_data(
        unfinished_path)
    with open(parse_obj.paths_out_of_npl, "r") as txt_file:
        paths = [line.rstrip('\n') for line in txt_file]
        assert unfinished_path in paths


def test_check_symmetry_issue(tmpdir):
    """Test whether we can find symmetry issue error in error file."""
    symm_issues_dir = tmpdir / 'symm_issues'
    symm_issues_dir.mkdir()
    paths_symmetry_issues = os.path.join(
        str(symm_issues_dir), 'paths_symmetry_issues.txt')
    # Use namespace name to call static method.
    symmetry_issue_file = 'tests/data/output_parser_vibes/pbe/8/LiF_ICSD_233/submission.sh'
    paths_symmetry_issues 
    parse_obj = create_mock_output_parser(save_directory=symm_issues_dir)
    parse_obj.gather_all_path_data(
        symmetry_issue_file)
    with open(parse_obj.paths_symmetry_issues, "r") as txt_file:
        paths = [line.rstrip('\n') for line in txt_file]
        assert symmetry_issue_file in paths


def test_check_for_symmetry_issue(tmpdir):
    """Test finding symmetry issue in path file."""
    parent_path = 'tests/data/output_parser_vibes/error_files'
    issue_path = 'djob.err.2342'
    test_dir = tmpdir / 'test'
    test_dir.mkdir()
    parse_obj = create_mock_output_parser(save_directory=test_dir)
    issue_bool = parse_obj.check_for_symmetry_issue(
        parent_path, issue_path)
    assert issue_bool is True
    issue_path = 'djob.err.2341'
    false_issue_bool = parse_obj.check_for_symmetry_issue(
        parent_path, issue_path)
    assert false_issue_bool is False
