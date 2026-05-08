"""Add material descriptors to dataframes using csv containing atomic data."""

import logging
import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


class AddDescriptors:
    """ This class adds descriptors to dataframe based on atomic data in csv format."""
    def __init__(self, dataframe, functional, csv_filename=None):
        self.logger = logging.getLogger()
        self.dataframe = dataframe
        self.functional = functional
        print("HEres the directory inside add descrip")
        print(os. getcwd())
        if csv_filename is None:
            self.get_csv_filename()
        else: 
            self.csv_filename = csv_filename
        self.load_csv_file()

    def get_csv_filename(self):
        if self.functional == "pbe":
            self.csv_filename = "modelling/data/descriptor_aims_data/really_tight_full_cut20_pbesol.csv"
        elif self.functional == "lda":
            self.csv_filename = "modelling/data/descriptor_aims_data/really_tight_full_cut20_pwlda.csv"
        else:
            print("functional: %s not recognized as either lda or pbe" % self.functional)

    def load_csv_file(self):
        """Load the csv file containing atomic descriptors into a df."""
        self.csv_data = pd.read_csv(self.csv_filename)

    def add_descriptors(self):
        """Add descriptors from loaded dataframe into the main dataframe."""

        self.dataframe['A EA Half'] = [
            float(self.csv_data[self.csv_data['Atomic number'] == x][
                'EA_half']) for x in self.dataframe['A Atom Number']]
        self.dataframe['B EA Half'] = [
            float(self.csv_data[self.csv_data['Atomic number'] == x][
                'EA_half']) for x in self.dataframe['B Atom Number']]
        self.dataframe['A IP Half'] = [
            float(self.csv_data[self.csv_data['Atomic number'] == x][
                'IP_half']) for x in self.dataframe['A Atom Number']]
        self.dataframe['B IP Half'] = [
            float(self.csv_data[self.csv_data['Atomic number'] == x][
                'IP_half']) for x in self.dataframe['B Atom Number']]
        self.dataframe['A EA Delta'] = [
            float(self.csv_data[self.csv_data['Atomic number'] == x][
                'EA_delta']) for x in self.dataframe['A Atom Number']]
        self.dataframe['B EA Delta'] = [
            float(self.csv_data[self.csv_data['Atomic number'] == x][
                'EA_delta']) for x in self.dataframe['B Atom Number']]
        self.dataframe['A IP Delta'] = [
            float(self.csv_data[self.csv_data['Atomic number'] == x][
                'IP_delta']) for x in self.dataframe['A Atom Number']]
        self.dataframe['B IP Delta'] = [
            float(self.csv_data[self.csv_data['Atomic number'] == x][
                'IP_delta']) for x in self.dataframe['B Atom Number']]
                
        self.dataframe['A HOMO'] = [
            float(self.csv_data[self.csv_data['Atomic number'] == x][
                'HOMO']) for x in self.dataframe['A Atom Number']]
        self.dataframe['B HOMO'] = [
            float(self.csv_data[self.csv_data['Atomic number'] == x][
                'HOMO']) for x in self.dataframe['B Atom Number']]
        
        self.dataframe['A LUMO'] = [
            float(self.csv_data[self.csv_data['Atomic number'] == x][
                'IP_half']) for x in self.dataframe['A Atom Number']]
        self.dataframe['B LUMO'] = [
            float(self.csv_data[self.csv_data['Atomic number'] == x][
                'IP_half']) for x in self.dataframe['B Atom Number']]
        
        self.dataframe['A rs'] = [
            float(self.csv_data[self.csv_data['Atomic number'] == x][
                'rs']) for x in self.dataframe['A Atom Number']]
        self.dataframe['B rs'] = [
            float(self.csv_data[self.csv_data['Atomic number'] == x][
                'rs']) for x in self.dataframe['B Atom Number']]
        self.dataframe['A s index'] = [
            float(self.csv_data[self.csv_data['Atomic number'] == x][
                's index']) for x in self.dataframe['A Atom Number']]
        self.dataframe['B s index'] = [
            float(self.csv_data[self.csv_data['Atomic number'] == x][
                's index']) for x in self.dataframe['B Atom Number']]      

        self.dataframe['A rp'] = [
            float(self.csv_data[self.csv_data['Atomic number'] == x][
                'rp']) for x in self.dataframe['A Atom Number']]
        self.dataframe['B rp'] = [
            float(self.csv_data[self.csv_data['Atomic number'] == x][
                'rp']) for x in self.dataframe['B Atom Number']]
        self.dataframe['A p index'] = [
            float(self.csv_data[self.csv_data['Atomic number'] == x][
                'p index']) for x in self.dataframe['A Atom Number']]
        self.dataframe['B p index'] = [
            float(self.csv_data[
                self.csv_data['Atomic number'] == x][
                    'p index']) for x in self.dataframe['B Atom Number']]      

        self.dataframe['A rd'] = [
            float(self.csv_data[self.csv_data['Atomic number'] == x][
                'rd']) for x in self.dataframe['A Atom Number']]
        self.dataframe['B rd'] = [
            float(self.csv_data[self.csv_data['Atomic number'] == x][
                'rd']) for x in self.dataframe['B Atom Number']]
        self.dataframe['A d index'] = [
            float(self.csv_data[self.csv_data['Atomic number'] == x][
                'd index']) for x in self.dataframe['A Atom Number']]
        self.dataframe['B d index'] = [
            float(self.csv_data[self.csv_data['Atomic number'] == x][
                'd index']) for x in self.dataframe['B Atom Number']]      

        self.dataframe['A rf'] = [
            float(self.csv_data[self.csv_data['Atomic number'] == x][
                'rf']) for x in self.dataframe['A Atom Number']]
        self.dataframe['B rf'] = [
            float(self.csv_data[self.csv_data['Atomic number'] == x][
                'rf']) for x in self.dataframe['B Atom Number']]
        self.dataframe['A f index'] = [
            float(self.csv_data[self.csv_data['Atomic number'] == x][
                'f index']) for x in self.dataframe['A Atom Number']]
        self.dataframe['B f index'] = [
            float(self.csv_data[self.csv_data['Atomic number'] == x][
                'f index']) for x in self.dataframe['B Atom Number']] 

    def add_descriptors_sisso(self):
        """Add descriptors from loaded dataframe into the main dataframe."""
        
        self.dataframe['min_ea_half'] = [
            float(self.csv_data[self.csv_data['Atomic number'] == x]['EA_half']) for x in self.dataframe['min_atom_number']]
        self.dataframe['max_ea_half'] = [
            float(self.csv_data[self.csv_data['Atomic number'] == x]['EA_half']) for x in self.dataframe['max_atom_number']]
        self.dataframe['min_ip_half'] = [
            float(self.csv_data[self.csv_data['Atomic number'] == x]['IP_half']) for x in self.dataframe['min_atom_number']]
        self.dataframe['max_ip_half'] = [
            float(self.csv_data[self.csv_data['Atomic number'] == x]['IP_half']) for x in self.dataframe['max_atom_number']]
        self.dataframe['min_ea_delta'] = [
            float(self.csv_data[self.csv_data['Atomic number'] == x]['EA_delta']) for x in self.dataframe['min_atom_number']]
        self.dataframe['max_ea_delta'] = [
            float(self.csv_data[self.csv_data['Atomic number'] == x]['EA_delta']) for x in self.dataframe['max_atom_number']]
        self.dataframe['min_ip_delta'] = [
            float(self.csv_data[self.csv_data['Atomic number'] == x]['IP_delta']) for x in self.dataframe['min_atom_number']]
        self.dataframe['max_ip_delta'] = [
            float(self.csv_data[self.csv_data['Atomic number'] == x]['IP_delta']) for x in self.dataframe['max_atom_number']]
                
        self.dataframe['min_homo'] = [
            float(self.csv_data[self.csv_data['Atomic number'] == x]['HOMO']) for x in self.dataframe['min_atom_number']]
        self.dataframe['max_homo'] = [
            float(self.csv_data[self.csv_data['Atomic number'] == x]['HOMO']) for x in self.dataframe['max_atom_number']]
        
        self.dataframe['min_lumo'] = [
            float(self.csv_data[self.csv_data['Atomic number'] == x]['LUMO']) for x in self.dataframe['min_atom_number']]
        self.dataframe['max_lumo'] = [
            float(self.csv_data[self.csv_data['Atomic number'] == x]['LUMO']) for x in self.dataframe['max_atom_number']]
        
        self.dataframe['min_rs'] = [
            float(self.csv_data[self.csv_data['Atomic number'] == x]['rs']) for x in self.dataframe['min_atom_number']]
        self.dataframe['max_rs'] = [
            float(self.csv_data[self.csv_data['Atomic number'] == x]['rs']) for x in self.dataframe['max_atom_number']]
        self.dataframe['min_s_index'] = [
            float(self.csv_data[self.csv_data['Atomic number'] == x]['s index']) for x in self.dataframe['min_atom_number']]
        self.dataframe['max_s_index'] = [
            float(self.csv_data[self.csv_data['Atomic number'] == x]['s index']) for x in self.dataframe['max_atom_number']]      

        self.dataframe['min_rp'] = [
            float(self.csv_data[self.csv_data['Atomic number'] == x]['rp']) for x in self.dataframe['min_atom_number']]
        self.dataframe['max_rp'] = [
            float(self.csv_data[self.csv_data['Atomic number'] == x]['rp']) for x in self.dataframe['max_atom_number']]
        self.dataframe['min_p_index'] = [
            float(self.csv_data[self.csv_data['Atomic number'] == x]['p index']) for x in self.dataframe['min_atom_number']]
        self.dataframe['max_p_index'] = [
            float(self.csv_data[self.csv_data['Atomic number'] == x]['p index']) for x in self.dataframe['max_atom_number']]      

        self.dataframe['min_rd'] = [
            float(self.csv_data[self.csv_data['Atomic number'] == x]['rd']) for x in self.dataframe['min_atom_number']]
        self.dataframe['max_rd'] = [
            float(self.csv_data[self.csv_data['Atomic number'] == x]['rd']) for x in self.dataframe['max_atom_number']]
        self.dataframe['min_d_index'] = [
            float(self.csv_data[self.csv_data['Atomic number'] == x]['d index']) for x in self.dataframe['min_atom_number']]
        self.dataframe['max_d_index'] = [
            float(self.csv_data[self.csv_data['Atomic number'] == x]['d index']) for x in self.dataframe['max_atom_number']]      

        self.dataframe['min_rf'] = [
            float(self.csv_data[self.csv_data['Atomic number'] == x]['rf']) for x in self.dataframe['min_atom_number']]
        self.dataframe['max_rf'] = [
            float(self.csv_data[self.csv_data['Atomic number'] == x]['rf']) for x in self.dataframe['max_atom_number']]
        self.dataframe['min_f_index'] = [
            float(self.csv_data[self.csv_data['Atomic number'] == x]['f index']) for x in self.dataframe['min_atom_number']]
        self.dataframe['max_f_index'] = [
            float(self.csv_data[self.csv_data['Atomic number'] == x]['f index']) for x in self.dataframe['max_atom_number']] 