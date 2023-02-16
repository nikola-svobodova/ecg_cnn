import os
import pandas as pd
import numpy as np
from utils.constants import DATA_FOLDER
from scipy import signal
from sklearn.model_selection import train_test_split
import pickle
from utils.constants import PICKLE_DATA
from sklearn.utils import shuffle

from utils.xml_processing.utils import get_patient_dict, get_diagnosis, load_df_from_xml


def get_annotations_df(filename, sheetname):
    """
    Loads doctors annotations

    :param filename: str
        filename
    :param sheetname: str
        sheetname
    :return: pd.DataFrame
        annotations
    """
    return pd.read_excel(filename, sheetname)


def load_muse(folder_path):
    """
    Loads .xml exported MUSE files

    :param folder_path: str
        folder path
    :return: list[str]
        MUSE files paths
    """
    muse_data_files_path = []
    for path, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".xml"):
                muse_data_files_path.append(os.path.join(path, file))

    return muse_data_files_path


def prepare_data_without_ecg(muse_files_paths):
    diagnosises = []
    ids = []
    dates = []
    times = []

    for muse_data_file in muse_files_paths:
        patient_dict = get_patient_dict(muse_data_file)
        diagnosis = get_diagnosis(patient_dict)
        patient_id = patient_dict['RestingECG']['PatientDemographics']['PatientID'].replace('/', '')
        date = patient_dict['RestingECG']['TestDemographics']['AcquisitionDate']
        time = patient_dict['RestingECG']['TestDemographics']['AcquisitionTime']

        diagnosises.append(diagnosis)
        ids.append(patient_id)
        dates.append(date)
        times.append(time)

    data_without_ekg = (pd.DataFrame(data={"id": ids, "date": dates, "time": times, "diagnose": diagnosises,
                                           "file_path": muse_files_paths}).sort_values(["id"]))
    return data_without_ekg


def prepare_data_without_ecg_cardio(muse_files_paths):
    diagnosises = []
    ids = []
    dates = []

    for muse_data_file in muse_files_paths:
        patient_dict = get_patient_dict(muse_data_file)
        diagnosis = get_diagnosis(patient_dict)
        patient_id = patient_dict['RestingECG']['PatientDemographics']['PatientID']
        date = patient_dict['RestingECG']['TestDemographics']['AcquisitionDate']
        time = patient_dict['RestingECG']['TestDemographics']['AcquisitionTime']

        diagnosises.append(diagnosis)
        ids.append(patient_id)
        dates.append(f'{date} {time}')

    data_without_ekg = (
        pd.DataFrame(data={"id": ids, "datetime": dates, "diagnose": diagnosises, "file_path": muse_files_paths})
        .assign(datetime=lambda df: pd.to_datetime(df["datetime"]))
        .sort_values(["id", "datetime"])
        .drop_duplicates(subset=["id", "datetime"])
        .loc[lambda df: ~df.diagnose.str.contains("paced")]
    )
    return data_without_ekg


def add_annotations_and_ecg(annotations, data_without_ekg):
    sorted_annotations = annotations[['patientId', 'date', 'time', 'rhythm']].sort_values(by=['patientId'])
    sorted_annotations['patientId'] = sorted_annotations['patientId'].str.replace('/', '')
    sorted_annotations['date'] = pd.to_datetime(sorted_annotations['date'], errors='coerce')
    sorted_annotations['date'] = sorted_annotations['date'].dt.strftime('%m-%d-%Y')
    sorted_annotations.rename(columns={'patientId': 'id'}, inplace=True)

    _matched_df = sorted_annotations.merge(data_without_ekg, on=['id', 'date', 'time'], how='left')

    print(_matched_df[_matched_df['file_path'].isna()])

    return _matched_df


def add_annotations_and_ecg_cardioversion(annotations, data_without_ekg):
    joined = annotations.join(data_without_ekg.set_index('id'), on='patient_id', lsuffix='_annotations')
    joined['delay'] = joined.datetime - joined.datetime_annotations
    joined['delay_minutes'] = joined.delay.dt.total_seconds() / 60
    joined = joined[joined.delay_minutes.abs() < 600]
    grouped_info = joined.groupby(['ID']).delay_minutes.agg(['min', 'max', 'count']).rename(
        columns={'min': 'first_ekg_delay', 'max': 'last_ekg_delay', 'count': 'ekg_count'})
    joined = joined.join(grouped_info, on='ID')

    joined = joined.assign(delay_quotient=(joined.delay_minutes - joined.first_ekg_delay) / (
            joined.last_ekg_delay - joined.first_ekg_delay))
    labeled_data = joined.assign(rhythm='unknown')
    labeled_data['rhythm'] = np.where(labeled_data.delay_quotient < 0.1, labeled_data.input_rhythm,
                                      labeled_data['rhythm'])
    labeled_data['rhythm'] = np.where(labeled_data.delay_quotient > 0.9, labeled_data.output_rhythm,
                                      labeled_data['rhythm'])
    return labeled_data


def add_annotations_and_ecg_most_recent(annotations, data_without_ekg):
    sorted_annotations = annotations[['bid', 'ekg_date', 'ekg_time', 'rytmus']].sort_values(by=['bid'])
    sorted_annotations['bid'] = sorted_annotations['bid'].str.replace('/', '')
    sorted_annotations['ekg_date'] = pd.to_datetime(sorted_annotations['ekg_date'], errors='coerce')
    sorted_annotations['ekg_date'] = sorted_annotations['ekg_date'].dt.strftime('%m-%d-%Y')
    sorted_annotations.rename(columns={'bid': 'id'}, inplace=True)
    sorted_annotations.rename(columns={'ekg_date': 'date'}, inplace=True)
    sorted_annotations.rename(columns={'ekg_time': 'time'}, inplace=True)
    sorted_annotations.rename(columns={'rytmus': 'rhythm'}, inplace=True)

    _matched_df = sorted_annotations.merge(data_without_ekg, on=['id', 'date'], how='left')

    print(_matched_df[_matched_df['file_path'].isna()])

    return _matched_df


def get_data_including_ecg(annotations, muse_files_path, cardioversion=False):
    muse_data_files = load_muse(muse_files_path)

    if cardioversion:
        without_ecg = prepare_data_without_ecg_cardio(muse_data_files)
        labeled = add_annotations_and_ecg_cardioversion(annotations, without_ecg)
    else:
        without_ecg = prepare_data_without_ecg(muse_data_files)
        labeled = add_annotations_and_ecg(annotations, without_ecg)
    data_with_ecg = labeled.assign(ekg=lambda df: df.file_path.apply(lambda data_file_path: load_df_from_xml(data_file_path)))

    return data_with_ecg


def get_final_dataset():
#    labeled_data = pd.read_csv(DATA_FOLDER + 'super_testovaci.csv')
    labeled_data = pd.read_csv(DATA_FOLDER + 'final_dataset.csv')

    muse_files = load_muse(DATA_FOLDER + 'old_annotation_matches')
    muse_files.extend(load_muse(DATA_FOLDER + 'annotation_matches'))
#     muse_files.extend(load_muse(DATA_FOLDER + 'Export-ze-seznamu'))
#     muse_files = load_muse(DATA_FOLDER + 'Export-ze-seznamu')

    data_with_ekg = (
        labeled_data
        .assign(ecg=lambda df: df.file_path.apply(lambda data_file_path: load_df_from_xml(data_file_path)))
    )
    return data_with_ekg


def downsampling_and_normalization(original_lead, num_of_samples=4096):
    """
    Resamples and normalizes data

    :param original_lead: ndarray
        original ECG lead
    :param num_of_samples: int
        number of samples
    :return: ndarray
        downsampled and normalized ECG lead
    """
    modified = signal.resample(original_lead, num_of_samples)
    modified = modified / 1000
    return modified


def label_condition(rhythm):
    if rhythm == 'AF':
        return 1
    else:
        return 0


def prepare_data_from_final_dataset():
    data_from_final_dataset_file = get_final_dataset()
    assert (len(data_from_final_dataset_file) == 5345)

    df = pd.DataFrame()
    df['label'] = data_from_final_dataset_file['rhythm'].apply(label_condition)
    df['data'] = data_from_final_dataset_file.apply(
        lambda row: downsampling_and_normalization(row.ecg.iloc[:, 0:12].values), axis=1)
    assert(df.loc[0, 'data'].shape == (4096, 12))
    df['muse'] = data_from_final_dataset_file['muse_diagnose']
    df['rhythm'] = data_from_final_dataset_file['rhythm']
    df = shuffle(df)
    return df


def pickle_data(X, y, filename):
    """
    Saves dataset (pickle)

    :param X: ndarray
        data values
    :param y: ndarray
        data labels
    :param filename: str
        filename
    """
    X = np.stack(X)
    pickle_out = open(PICKLE_DATA + "X_" + filename + ".pickle", "wb")
    pickle.dump(X, pickle_out)
    pickle_out.close()

    pickle_out = open(PICKLE_DATA + "y_" + filename + ".pickle", "wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()


def split_data(data):
    """
    Splits and saves data

    :param data: pd.DataFrame
    """
    X_train, X_test, y_train, y_test = train_test_split(data.data.values, data.label.values,
                                                        test_size=0.1, shuffle=True, random_state=37, stratify=data.label.values)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=0.1111, random_state=37, stratify=y_train)

    pickle_data(X_train, y_train, "train")
    pickle_data(X_val, y_val, "val")
    pickle_data(X_test, y_test, "test")


def load_data(dataset_type=''):
    """
    Loads pickle dataset

    :param dataset_type: str
        TRAIN / TEST  / VAL / SUPER_TEST
    :return: tuple[ndarray, ndarray]
    """
    pickle_in = open(PICKLE_DATA + "X" + dataset_type + ".pickle", "rb")
    X = pickle.load(pickle_in)

    pickle_in = open(PICKLE_DATA + "y" + dataset_type + ".pickle", "rb")
    y = pickle.load(pickle_in)
    return X, y

