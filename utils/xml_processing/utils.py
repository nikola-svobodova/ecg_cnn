import os
import xml
import pandas as pd
import xmltodict

from utils.xml_processing.musexmlex import MuseXmlParser


# Quick and dirty fuction to get df from the data
def load_df_from_xml(path):
    def start_element(name, attrs):
        g_parser.start_element(name, attrs)

    def end_element(name):
        g_parser.end_element(name)

    def char_data(data):
        g_parser.char_data(data)

    g_parser = MuseXmlParser()

    p = xml.parsers.expat.ParserCreate()

    p.StartElementHandler = start_element
    p.EndElementHandler = end_element
    p.CharacterDataHandler = char_data

    with open(path, 'rb') as f:
        p.ParseFile(f)

    # convert the data into a ZCG buffer
    g_parser.makeZcg()
    g_parser.write_csv('temp.csv')

    df = pd.read_csv('temp.csv')

    os.remove('temp.csv')

    return df


def get_patient_dict(path_to_xml):
    with open(path_to_xml, 'rb') as fd:
        return xmltodict.parse(fd.read().decode('utf8'))


def get_diagnosis(patient_dict):
    patient_diagnosises = patient_dict['RestingECG']['Diagnosis']['DiagnosisStatement']
    diag_text = ''
    newline = ''
    for diag in patient_diagnosises:
        diag_flag = diag.get('StmtFlag', None)
        if diag_flag:
            newline = ';'
        diag_text += f"{diag.get('StmtText', None)}{newline}"
    return diag_text
