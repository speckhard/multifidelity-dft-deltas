""""Output parser tests."""

from pathlib import Path

import pytest
import os
import sys
import ase
import ast
import csv
import ase.db
import ase.calculators.aims
import pandas as pd

# Append path so that we can access packages that are above this directory.
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import parsing.output_parser_exciting_geo_opt


def create_mock_output_parser(
        test_dir: Path,
        list_of_paths = None,
        paths_txt_file='one_path.txt'):
    """Create mock output parser"""
    # Now create the mock object.
    with open(test_dir / paths_txt_file, 'w') as fd:
        fd.write('\n'.join(list_of_paths))

    parse_obj = parsing.output_parser_exciting_geo_opt.OutputParserExciting(
        input_paths_txt_file=test_dir / paths_txt_file,
        save_directory=test_dir)
    return parse_obj


def test_parse_path_expansion():
    """Test ability to parse settings used in a calc from pathname."""
    # path = ('tests/data/output_parser/73_Ta')
    path = (
        'tests/exciting/sven_regen/old_species_path/GGA_PBE/'
        'precision_0_3/rmt_scaling_0_95/8/Be4S4/submission_Be4S4.sh')

    settings_dict = parsing.output_parser_exciting_geo_opt.OutputParserExciting.parse_path(
            path)

    assert settings_dict['compound_name'] == 'Be4S4'
    assert settings_dict['functional'] == 'GGA_PBE'
    assert settings_dict['APW_precision'] == 0.3
    assert settings_dict['rmt_scaling'] == 0.95
    assert settings_dict['k_point_density'] == 8


def test_gather_all_path_data_no_err_file(tmpdir):
    """Test parsing calc with no error file present in directory.
    
    This file should get added to the out of npl files since this means
    the calc was not run at all.
    """
    # The path here is to an empty directory.
    path = (
        '/home/dts/Documents/theory/errorbar_modelling/tests/data/output_parser_exciting_geo_opt/GGA_PBE/precision_0_4/rmt_scaling_0_95/2/OutOfNpl/submission_O4_Ti2.sh')
    test_dir = tmpdir / 'test'
    test_dir.mkdir()
    parse_obj = create_mock_output_parser(
        test_dir=tmpdir,
        list_of_paths=[path])
    data_dict = parse_obj.gather_all_path_data(path)
    print(data_dict)
    # Currently no energy data is being parsed.
    assert data_dict['APW_precision'] == 0.4
    assert os.path.isfile(parse_obj.paths_out_of_npl)
    print(parse_obj.paths_out_of_npl)
    with open(parse_obj.paths_out_of_npl, 'r') as fo:
        paths_list = fo.readlines()
    assert len(paths_list) == 1
    assert paths_list[0] == path + '\n'

def test_gather_all_path_data_expired_geo_opt_not_started(tmpdir):
    """Calculation cancelled before geo opt step. due to time."""
    path = (
        'tests/data/output_parser_exciting_geo_opt/GGA_PBE/'
        'precision_0_5/rmt_scaling_0_95/8/O36Si18_ICSD_170544/submission.sh')
    # The path here is to an empty directory.
    test_dir = tmpdir / 'test'
    test_dir.mkdir()
    parse_obj = create_mock_output_parser(
        test_dir=tmpdir,
        list_of_paths=[path])
    data_dict = parse_obj.gather_all_path_data(path)
    print(data_dict)
    # Currently no energy data is being parsed.
    assert data_dict['APW_precision'] == 0.5
    assert os.path.isfile(parse_obj.paths_geo_opt_not_started)
    print(parse_obj.paths_geo_opt_not_started)
    with open(parse_obj.paths_geo_opt_not_started, 'r') as fo:
        paths_list = fo.readlines()
    assert len(paths_list) == 1
    assert paths_list[0] == path + '\n'


def test_gather_all_path_data(tmpdir):
    """Test getting all data from path stored to dict."""
    path = (
        '/home/dts/Documents/theory/errorbar_modelling/tests/data/'
        'output_parser_exciting_geo_opt/GGA_PBE/precision_0_4/'
        'rmt_scaling_0_95/2/O8Ti4/submission_O8_Ti4.sh')
    test_dir = tmpdir / 'test'
    test_dir.mkdir()
    parse_obj = create_mock_output_parser(
        test_dir=tmpdir,
        list_of_paths=[path])
    data_dict = parse_obj.gather_all_path_data(path)
    print(data_dict)
    # Currently no energy data is being parsed.
    assert round(
        data_dict['total_comp_time'] - 974.39, 7) == 0
    assert data_dict['total_energy'] == 'not_found' 
    assert round(
        data_dict['k_point_total']-9.0, 7) == 0.0
    assert round(
        data_dict['unit_cell_volume'] - 431.3138326178, 7) == 0.0 
    assert data_dict['num_apw_functions'] == 10
    assert data_dict['total_num_los'] == 188


def test_get_band_file_data():
    """Test getting band file data."""
    pass


def test_create_header(tmpdir):
    """Test creating header for a file.
    We want to ensure that a csv file is created.
    The right header is there.
    """
    path = (
        'tests/data/'
        'output_parser_exciting/old_species_path/GGA_PBE/'
        'precision_0_3/rmt_scaling_0_95/8/H4Li4/submission_h4Li4.sh')
    test_dir = tmpdir / 'test'
    test_dir.mkdir()

    csv_filename = (
        'test_csv_file.csv')

    parse_obj = create_mock_output_parser(
        test_dir=test_dir,
        list_of_paths=[path])

    parse_obj.create_header()
    # Now let's check that the file was created.
    assert os.path.isfile(parse_obj.csv_filename)
    # Now let's open it up and read the header.
    with open(parse_obj.csv_filename, "r") as csv_file:
        header = next(csv.reader(csv_file))
        assert header[0] == 'compound_name'
        assert header[1] == 'APW_precision'
        assert header[2] == 'k_point_density'


def test_check_calc_finished():
    """Test whether we sim finished nicely.

    Feed in a file that has have a nice day
    in the last 50 lines. See if the method
    can find return true. Feed another file
    without have a nice day and ensure
    False is returned.
    """
    finished_path = (
        'tests/data/output_parser_exciting/old_species_path/GGA_PBE/'
        'precision_0_3/rmt_scaling_0_95/8/B2N2/')
    # Use namespace name to call static method.
    (calc_finished_bool, geo_opt_finished_bool, geo_opt_started_bool,
     gs_started, calc_expired_bool, out_of_npl_bool, diag_fail_bool,
     mt_overlap_bool, calc_ongoing_bool, ndirac_limit_bool, oom_bool,
     total_comp_time) = parsing.output_parser_exciting_geo_opt.OutputParserExciting.check_calc_finished(
        finished_path)
    assert calc_finished_bool is True
    assert geo_opt_started_bool is True
    assert gs_started is True
    assert calc_expired_bool is False
    assert calc_ongoing_bool is False
    assert out_of_npl_bool is False
    assert geo_opt_finished_bool is True
    assert total_comp_time == 349.63
    assert diag_fail_bool is False
    assert mt_overlap_bool is False
    assert ndirac_limit_bool is False
    assert oom_bool is False

    # Now try a file that never ran. No INFO OUT no djob out.
    unfinished_path = 'tests/data/output_parser_exciting/old_species_path/GGA_PBE/precision_0_3/rmt_scaling_0_95/8/SiC/'
    # Use namespace name to call static method.
    (
        calc_finished_bool, geo_opt_finished_bool, geo_opt_started_bool,
        gs_started, calc_expired_bool, out_of_npl_bool, diag_fail_bool,
        mt_overlap_bool, calc_ongoing_bool, ndirac_limit_bool, oom_bool,
        total_comp_time) = parsing.output_parser_exciting_geo_opt.OutputParserExciting.check_calc_finished(
            unfinished_path)
    assert calc_finished_bool is False
    assert calc_expired_bool is False
    assert geo_opt_started_bool is False
    assert gs_started is False
    assert calc_ongoing_bool is True
    assert out_of_npl_bool is False
    assert total_comp_time is None
    assert geo_opt_finished_bool is False
    assert diag_fail_bool is False
    assert mt_overlap_bool is False
    assert ndirac_limit_bool is False
    assert oom_bool is False

    # Now try a file where gs didnt finish.
    unfinished_path = 'tests/data/output_parser_exciting/old_species_path/GGA_PBE/precision_0_3/rmt_scaling_0_95/8/O8Po4/'
    # Use namespace name to call static method.
    (calc_finished_bool, geo_opt_finished_bool, geo_opt_started_bool,
     gs_started, calc_expired_bool, out_of_npl_bool, diag_fail_bool,
     mt_overlap_bool, calc_ongoing_bool, ndirac_limit_bool, oom_bool,
     total_comp_time) = parsing.output_parser_exciting_geo_opt.OutputParserExciting.check_calc_finished(
        unfinished_path)
    assert calc_finished_bool is False
    assert calc_expired_bool is False
    assert geo_opt_started_bool is False
    assert gs_started is True
    assert calc_ongoing_bool is False
    assert out_of_npl_bool is False
    assert total_comp_time is None
    assert geo_opt_finished_bool is False
    assert diag_fail_bool is True
    assert mt_overlap_bool is False
    assert ndirac_limit_bool is False
    assert oom_bool is False

    # Now try a file where ndirac limit was reached
    unfinished_path = 'tests/data/output_parser_exciting/old_species_path/GGA_PBE/precision_0_3/rmt_scaling_0_95/8/NaCl/'
    # Use namespace name to call static method.
    (calc_finished_bool, geo_opt_finished_bool, geo_opt_started_bool,
     gs_started, calc_expired_bool, out_of_npl_bool, diag_fail_bool,
     mt_overlap_bool, calc_ongoing_bool, ndirac_limit_bool, oom_bool,
     total_comp_time) = parsing.output_parser_exciting_geo_opt.OutputParserExciting.check_calc_finished(
        unfinished_path)
    assert calc_finished_bool is False
    assert calc_expired_bool is False
    assert geo_opt_started_bool is False
    assert gs_started is True
    assert calc_ongoing_bool is True
    assert out_of_npl_bool is False
    assert total_comp_time is None
    assert geo_opt_finished_bool is False
    assert diag_fail_bool is False
    assert mt_overlap_bool is False
    assert ndirac_limit_bool is True
    assert oom_bool is False

    # Now try a file where out of memory error occured.
    unfinished_path = 'tests/data/output_parser_exciting/old_species_path/GGA_PBE/precision_0_3/rmt_scaling_0_95/8/As16S16/'
    # Use namespace name to call static method.
    (calc_finished_bool, geo_opt_finished_bool, geo_opt_started_bool,
     gs_started, calc_expired_bool, out_of_npl_bool, diag_fail_bool,
     mt_overlap_bool, calc_ongoing_bool, ndirac_limit_bool, oom_bool,
     total_comp_time) = parsing.output_parser_exciting_geo_opt.OutputParserExciting.check_calc_finished(
        unfinished_path)
    assert calc_finished_bool is False
    assert calc_expired_bool is False
    assert geo_opt_started_bool is False
    assert gs_started is True
    assert calc_ongoing_bool is False
    assert out_of_npl_bool is False
    assert total_comp_time is None
    assert geo_opt_finished_bool is False
    assert diag_fail_bool is False
    assert mt_overlap_bool is False
    assert ndirac_limit_bool is False
    assert oom_bool is True


def test_calc_finished_real_example(tmpdir):
    """Test whether we sim finished nicely.

    Feed in a file that has have a nice day
    in the last 50 lines. See if the method
    can find return true. Feed another file
    without have a nice day and ensure
    False is returned.
    """
    unfinished_parent_path = (
        'tests/data/output_parser_exciting_geo_opt/GGA_PBE/precision_0_5/rmt_scaling_0_95/8/O36Si18_ICSD_170544')
    # Use namespace name to call static method.
    (calc_finished_bool, geo_opt_finished_bool, geo_opt_started_bool,
     gs_started, calc_expired_bool, out_of_npl_bool, diag_fail_bool,
     mt_overlap_bool, calc_ongoing_bool, ndirac_limit_bool, oom_bool,
     total_comp_time) = parsing.output_parser_exciting_geo_opt.OutputParserExciting.check_calc_finished(
        unfinished_parent_path)
    assert calc_finished_bool is False
    assert geo_opt_started_bool is False
    assert gs_started is True
    assert calc_expired_bool is True
    assert calc_ongoing_bool is False
    assert out_of_npl_bool is False
    assert geo_opt_finished_bool is False
    assert total_comp_time is None
    assert diag_fail_bool is False
    assert mt_overlap_bool is False
    assert ndirac_limit_bool is False
    assert oom_bool is False
    

def test_gather_all_path_data_with_missing_species_file(tmpdir):
    """Test parsing calc with no error file present in directory.
    
    This file should get added to the out of npl files since this means
    the calc was not run at all.
    """
    # The path here is to a file that doesn't exist, that's not important tho
    # since the containing folder will be used.
    path = (
        'tests/data/output_parser_exciting/old_species_path/GGA_PBE/precision_0_3/rmt_scaling_0_95/8/Fe2/submission_Fe2.sh')
    test_dir = tmpdir / 'test'
    test_dir.mkdir()
    parse_obj = create_mock_output_parser(
        test_dir=tmpdir,
        list_of_paths=[path])
    data_dict = parse_obj.gather_all_path_data(path)
    print(data_dict)
    # Currently no energy data is being parsed.
    assert data_dict['APW_precision'] == 0.3
    assert os.path.isfile(parse_obj.paths_species_xml_missing)
    print(parse_obj.paths_species_xml_missing)
    with open(parse_obj.paths_species_xml_missing, 'r') as fo:
        paths_list = fo.readlines()
    assert len(paths_list) == 1
    assert paths_list[0] == path + '\n'


def test_add_expired_path(tmpdir):
    """Test method to add paths of sims that didn't exit nicely."""
    # Create mock output parse object.
    path = (
        'tests/data/'
        'output_parser_exciting/old_species_path/GGA_PBE/'
        'precision_0_3/rmt_scaling_0_95/8/H4Li4/submission_h4Li4.sh')

    test_dir = tmpdir / 'test'
    test_dir.mkdir()

    paths_to_resubmit = 'resub_paths.txt'

    parse_obj = create_mock_output_parser(
        test_dir=test_dir,
        list_of_paths=[path])

    submission_path = 'test_path/submission_XY.txt'
    # Tests paths to resubmit due to time expiration.
    parse_obj.add_expired_path(submission_path)

    print(parse_obj.paths_to_resubmit)
    # Check if the file contains the submission path.
    with open(parse_obj.paths_to_resubmit, "r") as txt_file:
        paths = [line.rstrip('\n') for line in txt_file]
        assert submission_path in paths


def test_add_misbehaving_path(tmpdir):
    """Test method to add paths of sims that didn't exit nicely."""
    # Create mock output parse object.
    path = (
        'tests/data/'
        'output_parser_exciting/old_species_path/GGA_PBE/'
        'precision_0_3/rmt_scaling_0_95/8/H4Li4/submission_h4Li4.sh')

    test_dir = tmpdir / 'test'
    test_dir.mkdir()

    paths_misbehaving = 'misbehave_paths.txt'

    parse_obj = create_mock_output_parser(
        test_dir=test_dir,
        list_of_paths=[path])

    submission_path = 'test_path/submission_XY.txt'
    # Tests paths to resubmit due to time expiration.
    parse_obj.add_misbehaving_path(submission_path)
    # Check if the file contains the submission path.
    with open(parse_obj.paths_misbehaving, "r") as txt_file:
        paths = [line.rstrip('\n') for line in txt_file]
        assert submission_path in paths


def test_check_out_file_fail(tmpdir):

    fail_path = 'tests/data/output_parser_exciting/old_species_path/GGA_PBE/precision_0_3/rmt_scaling_0_95/8/O8Po4'

    diag_fail_bool, mt_overlap_bool, calc_ongoing_bool, ndirac_limit_bool = parsing.output_parser_exciting_geo_opt.OutputParserExciting.check_out_file_fail(
        fail_path)
    assert diag_fail_bool is True
    assert mt_overlap_bool is False
    assert calc_ongoing_bool is False
    assert ndirac_limit_bool is False


    # Now try a file that ran
    pass_path = 'tests/data/output_parser_exciting/old_species_path/GGA_PBE/precision_0_3/rmt_scaling_0_95/8/TiO/'
    diag_fail_bool, mt_overlap_bool, calc_ongoing_bool, ndirac_limit_bool = parsing.output_parser_exciting_geo_opt.OutputParserExciting.check_out_file_fail(
        pass_path)
    assert diag_fail_bool is False
    assert mt_overlap_bool is False
    assert calc_ongoing_bool is False
    assert ndirac_limit_bool is False

    # Now try a file that never ran. I don't this situation would be called in the code,
    # since this method should only be called if an INFO out exists and if that is the case
    # then a djob out should exist.
    pass_path = 'tests/data/output_parser_exciting/old_species_path/GGA_PBE/precision_0_3/rmt_scaling_0_95/8/SiC/'
    diag_fail_bool, mt_overlap_bool, calc_ongoing_bool, ndirac_limit_bool = parsing.output_parser_exciting_geo_opt.OutputParserExciting.check_out_file_fail(
        pass_path)
    assert mt_overlap_bool is False
    assert diag_fail_bool is False
    assert calc_ongoing_bool is True
    assert ndirac_limit_bool is False



    # Now try a file where the mt overlapped.
    pass_path = 'tests/data/output_parser_exciting_geo_opt/GGA_PBE/precision_0_4/rmt_scaling_0_95/2/Li4O4/'
    diag_fail_bool, mt_overlap_bool, calc_ongoing_bool, ndirac_limit_bool = parsing.output_parser_exciting_geo_opt.OutputParserExciting.check_out_file_fail(
        pass_path)
    assert mt_overlap_bool is True
    assert diag_fail_bool is False
    assert calc_ongoing_bool is False
    assert ndirac_limit_bool is False

    # Now try a file where the ndirac limit was reached
    pass_path = 'tests/data/output_parser_exciting/old_species_path/GGA_PBE/precision_0_3/rmt_scaling_0_95/8/NaCl/'
    diag_fail_bool, mt_overlap_bool, calc_ongoing_bool, ndirac_limit_bool= parsing.output_parser_exciting_geo_opt.OutputParserExciting.check_out_file_fail(
        pass_path)
    assert mt_overlap_bool is False
    assert diag_fail_bool is False
    assert calc_ongoing_bool is True
    assert ndirac_limit_bool is True


def test_check_no_info_out_err_file_species_xml_missing(tmpdir):

    fail_parent_path = 'tests/data/output_parser_exciting/old_species_path/GGA_PBE/precision_0_3/rmt_scaling_0_95/8/Fe2'
    species_xml_missing_bool, scalar_to_integer_bool = parsing.output_parser_exciting_geo_opt.OutputParserExciting.check_no_info_out_err_file(fail_parent_path)
    assert species_xml_missing_bool is True
    assert scalar_to_integer_bool is False

    # Now try a file that never ran.
    pass_path = 'tests/data/output_parser_exciting/old_species_path/GGA_PBE/precision_0_3/rmt_scaling_0_95/8/SiC/'
    species_xml_missing_bool, scalar_to_integer_bool = parsing.output_parser_exciting_geo_opt.OutputParserExciting.check_no_info_out_err_file(pass_path)

    assert species_xml_missing_bool is False
    assert scalar_to_integer_bool is False


def test_check_state_out_read_issue(tmpdir):
    fail_parent_path = 'tests/data/output_parser_exciting/old_species_path/GGA_PBE/precision_0_3/rmt_scaling_0_95/8/H2O'
    state_out_bool = parsing.output_parser_exciting_geo_opt.OutputParserExciting.check_state_out_read_issue(fail_parent_path)
    assert state_out_bool is True
    # Now try a file that never ran.
    pass_path = 'tests/data/output_parser_exciting/old_species_path/GGA_PBE/precision_0_3/rmt_scaling_0_95/8/SiC/'
    state_out_bool = parsing.output_parser_exciting_geo_opt.OutputParserExciting.check_state_out_read_issue(pass_path)
    assert state_out_bool is False


def test_check_no_info_out_err_file_scalar_to_integer(tmpdir):

    fail_parent_path = 'tests/data/output_parser_exciting/old_species_path/GGA_PBE/precision_0_3/rmt_scaling_0_95/8/PbS'
    species_xml_missing_bool, scalar_to_integer_bool = parsing.output_parser_exciting_geo_opt.OutputParserExciting.check_no_info_out_err_file(fail_parent_path)
    assert species_xml_missing_bool is False
    assert scalar_to_integer_bool is True
    # Now try a file that never ran.
    pass_path = 'tests/data/output_parser_exciting/old_species_path/GGA_PBE/precision_0_3/rmt_scaling_0_95/8/SiC/'
    species_xml_missing_bool, scalar_to_integer_bool = parsing.output_parser_exciting_geo_opt.OutputParserExciting.check_no_info_out_err_file(pass_path)
    assert scalar_to_integer_bool is False
    assert species_xml_missing_bool is False


def test_check_calc_finished_max_force_target_already_reached(tmpdir):

    parent_path = 'tests/data/output_parser_exciting_geo_opt/precision_0_8/rmt_scaling_0_95/8/Ge6N8/'
    (calc_finished_bool, geo_opt_finished_bool, geo_opt_started_bool,
     gs_started,calc_expired_bool, out_of_npl_bool, diag_fail_bool, mt_overlap_bool,
     calc_ongoing_bool, ndirac_limit_bool, oom_bool, total_comp_time) = parsing.output_parser_exciting_geo_opt.OutputParserExciting.check_calc_finished(
        parent_path)
    assert calc_finished_bool is True
    assert calc_expired_bool is False
    assert geo_opt_started_bool is True
    assert gs_started is True
    assert calc_ongoing_bool is False
    assert out_of_npl_bool is False
    assert total_comp_time == 3025.47
    assert geo_opt_finished_bool is True
    assert diag_fail_bool is False
    assert mt_overlap_bool is False
    assert ndirac_limit_bool is False
    assert oom_bool is False


def test_gather_data_max_force_target_already_reached(tmpdir):

    parent_path = 'tests/data/output_parser_exciting_geo_opt/precision_0_8/rmt_scaling_0_95/8/Ge6N8/'
    parse_obj = create_mock_output_parser(
        test_dir=tmpdir,
        list_of_paths=[parent_path])
    data_dict = parse_obj.gather_all_path_data(parent_path)
    print(data_dict)
    # Currently no energy data is being parsed.
    assert data_dict['APW_precision'] == 0.8
    assert data_dict['total_comp_time'] == 3025.47

    # Why is this job being added to the paths misbehaving?
    assert not os.path.isfile(parse_obj.paths_misbehaving)
    assert data_dict['unit_cell_volume'] == 969.0178568398
    assert data_dict['status'] == 'calc_finished_after_parsing'


def test_check_if_exciting_info_out_does_not_exist(tmpdir):
    """Test whether we can find error msg in djob err.
    TODO: we should really combine this and the above
    test with one setup.
    """
    path = 'bogus.txt'
    test_dir = tmpdir / 'test'
    test_dir.mkdir()
    expected_bool = True
    # Now try a file that doesn't have have a nice day.
    unfinished_path = 'tests/data/output_parser_exciting/MgO'
    # Use namespace name to call static method.
    parse_obj = create_mock_output_parser(test_dir, list_of_paths=[path])
    exciting_out_bool = parse_obj.check_if_exciting_info_out_does_not_exist(
        unfinished_path)
    assert exciting_out_bool is expected_bool


def test_get_most_recent_djob_err():
    """Test getting the most recent djob.err file."""
    submission_path = 'tests/data/output_parser_vibes/error_files'
    returned_error_path = parsing.output_parser_exciting_geo_opt.OutputParserExciting.get_most_recent_djob_err(
        submission_path)
    expected_error_path = (
        'djob.err.2342')
    assert returned_error_path == expected_error_path


def test_get_most_recent_djob_err_empty():
    """Test getting the most recent djob.err file when there is none."""
    submission_path = 'tests/data/output_parser_vibes/empty_error_files'
    returned_error_path = parsing.output_parser_exciting_geo_opt.OutputParserExciting.get_most_recent_djob_err(
        submission_path)
    expected_error_path = None
    assert returned_error_path == expected_error_path


def test_connect_to_csv(tmpdir):
    paths = [
        ('/home/dts/Documents/theory/errorbar_modelling/tests/data/output_parser_exciting_geo_opt/GGA_PBE/precision_0_4/rmt_scaling_0_95/2/O4Ti2/submission.sh'),
        ('/home/dts/Documents/theory/errorbar_modelling/tests/data/output_parser_exciting_geo_opt/GGA_PBE/precision_0_4/rmt_scaling_0_95/2/MgO/submission.sh')]

    test_dir = tmpdir / 'test'
    test_dir.mkdir()

    parse_obj = create_mock_output_parser(
        test_dir=test_dir,
        list_of_paths=paths)
    parse_obj.connect_to_csv()
    df = pd.read_csv(parse_obj.csv_filename)
    print(df)

    with open(parse_obj.csv_filename, "r") as csv_file:
        header = next(csv.reader(csv_file))
        assert header[0] == 'compound_name'
        assert header[1] == 'APW_precision'
        assert header[2] == 'k_point_density'
        # assert header[7] == 'total_energy'
        assert header[-3] == 'relaxed_alpha_angle'
        assert header[-2] == 'relaxed_beta_angle'
        assert header[-1] == 'relaxed_gamma_angle'

        first_row = next(csv.reader(csv_file))
        assert first_row[0] == 'O4Ti2'
        assert float(first_row[1]) == 0.4
        assert first_row[-2] == '90.0'

        # second_row = next(csv.reader(csv_file))
        # assert second_row[0] == 'MgO'
        # assert second_row[1] == 0.4