import numpy as np
import pandas as pd
import simpy
from utils import date_time
from collections import namedtuple


class Triage_Bay(object):
    def __init__(self):
        pass

    @staticmethod
    def resources(env, count):
        T_Bay = namedtuple('T_Bay', 'number')
        # t1 = T_Bay(1)
        # t2 = T_Bay(2)
        # t3 = T_Bay(3)
        T_Bay_Store = simpy.FilterStore(env, capacity=count)
        # T_Bay_Store.items = [t1, t2, t3]
        T_Bay_Store.items = [T_Bay(i + 1) for i in range(count)]
        return T_Bay_Store


class Inpatient_Bed(object):
    def __init__(self, params):
        self.df = pd.read_csv(params['inpatient']['details'])
        self.floor_bed_dict = {}
        for _, g in self.df.groupby('floor'):
            self.floor_bed_dict[_] = len(g)
        self.total_beds = sum([val for val in self.floor_bed_dict.values()])

    def resources(self, env):
        Bed_Factory = simpy.FilterStore(env, capacity=self.total_beds)
        Bed_ = namedtuple('Bed', 'floor, number')
        Bed_Factory.items = []
        for floor in self.floor_bed_dict.keys():
            bed_count = self.floor_bed_dict[floor]  # gets max count
            Bed_Factory.items += [Bed_(floor=floor, number=i + 1) for i in range(bed_count)]
        return Bed_Factory


class Staff(object):
    def __init__(self):
        self.df_schedule = None
        self.df_count = None
        self.prev_shift = None
        self.prev_count = None
        self.tuple_ = None
        self.proba_list = [0.3, 0.4, 0.3]

    def get_shift(self, day, hour):
        shift = self.df_schedule.loc[day, str(hour)]
        return shift

    def get_shift_count(self, shift, random=True):
        count = self.df_count.loc[shift, 'count']
        if not random:
            return count
        elif shift == self.prev_shift:
            return self.prev_count
        else:
            count_m2 = max(count - 2, 1)
            count_m1 = max(count - 1, 1)
            count = np.random.choice([count_m2, count_m1, count], p=self.proba_list)
            self.prev_shift = shift
            self.prev_count = count
            return count

    def resources(self, env):
        total_count = self.df_count['count'].sum()
        Staff_Factory = simpy.FilterStore(env, capacity=total_count)
        Staff_ = self.tuple_
        Staff_Factory.items = []
        for shift in self.df_count.index:
            count = self.get_shift_count(shift, random=False)  # gets max count
            Staff_Factory.items += [Staff_(shift=shift, number=i + 1) for i in range(count)]
        return Staff_Factory


class Nurse(Staff):
    def __init__(self, params):
        super().__init__()
        self.df_schedule = pd.read_csv(params['nurse']['shift_schedule'], index_col=0)
        self.df_count = pd.read_csv(params['nurse']['shift_count'], index_col=0)
        self.tuple_ = namedtuple('Nurse', 'shift, number')
        self.proba_list = params['nurse']['shift_count_proba']


class MD(Staff):
    def __init__(self, params):
        super().__init__()
        self.df_schedule = pd.read_csv(params['md']['shift_schedule'], index_col=0)
        self.df_count = pd.read_csv(params['md']['shift_count'], index_col=0)
        self.tuple_ = namedtuple('MD', 'shift, number')
        self.proba_list = params['md']['shift_count_proba']


class APP(Staff):
    def __init__(self, params):
        super().__init__()
        self.df_schedule = pd.read_csv(params['app']['shift_schedule'], index_col=0)
        self.df_count = pd.read_csv(params['app']['shift_count'], index_col=0)
        self.tuple_ = namedtuple('APP', 'shift, number')
        self.proba_list = params['app']['shift_count_proba']


class CaseManagers(Staff):
    def __init__(self, params):
        super().__init__()
        self.df_schedule = pd.read_csv(params['cm']['shift_schedule'], index_col=0)
        self.df_count = pd.read_csv(params['cm']['shift_count'], index_col=0)
        self.tuple_ = namedtuple('CM', 'shift, number')
        self.proba_list = [0., 0., 1]


class Pharmacist(Staff):
    def __init__(self, params):
        super().__init__()
        self.df_schedule = pd.read_csv(params['pharmacist']['shift_schedule'], index_col=0)
        self.df_count = pd.read_csv(params['pharmacist']['shift_count'], index_col=0)
        self.tuple_ = namedtuple('Pharmacist', 'shift, number')
        self.proba_list = [0., 0., 1]


class BedStaff(Staff):
    def __init__(self, params):
        super().__init__()
        self.df_schedule = pd.read_csv(params['bed_staff']['shift_schedule'], index_col=0)
        self.df_count = pd.read_csv(params['bed_staff']['shift_count'], index_col=0)
        self.tuple_ = namedtuple('BedStaff', 'shift, number')
        self.proba_list = [0., 0., 1]


class Inpatient_Hospital(object):

    def __init__(self, env, params):
        self.params = params
        # initialize factories
        self.nurse_factory = Nurse(params)
        self.md_factory = MD(params)
        self.app_factory = APP(params)
        self.cm_factory = CaseManagers(params)
        self.pharmacist_factory = Pharmacist(params)
        self.bed_factory = Inpatient_Bed(params)
        self.bed_staff_factory = BedStaff(params)
        self.today = None

        # resources
        self.env = env
        self.visit = simpy.Resource(env, capacity=1000)
        self.register_desk = simpy.Resource(env, 1)
        self.nurse = self.nurse_factory.resources(env)
        self.md = self.md_factory.resources(env)
        self.app = self.app_factory.resources(env)
        self.cm = self.cm_factory.resources(env)
        self.bed = self.bed_factory.resources(env)
        self.occupied_beds = []
        self.bed_staff = self.bed_staff_factory.resources(env)
        self.pharmacist = self.pharmacist_factory.resources(env)
        self.rounding_patient_timeout = params['rounding_duration']
        self.rounding_floor_timeout = params['floor_change_duration']
        self.discharge_order_timeout = params['discharge_order_duration']
        self.discharge_process_timeout = params['discharge_process_duration']
        self.medication_timeout = params['medication_duration']
        self.case_management_timeout = params['case_management_duration']
        self.equipment_timeout = params['equipment_duration']
        self.transport_timeout = params['transport_duration']
        self.bed_clean_timeout = params['bed_clean_duration']
        self.triage_bay = Triage_Bay.resources(env, 3)
        self.treatment_room = simpy.Resource(env, 5)
        self.triage_timeout_nurse = params['nurse']['triage_duration']
        self.triage_timeout_nurse_longer = params['nurse']['longer_triage_duration']
        self.triage_timeout_app = params['app']['triage_duration']
        self.triage_timeout_app_longer = params['app']['longer_triage_duration']
        self.triage_timeout_md = params['md']['triage_duration']
        self.triage_timeout_md_longer = params['md']['longer_triage_duration']

    def registration(self, patient):
        timeout = self.params['registration']['duration']
        yield self.env.timeout(abs(np.random.normal(timeout * 60, 1 * 10)))

    def triage(self, patient, by='nurse', longer=False):
        if longer:
            triage_timeout_ = 'longer_triage_duration'
        else:
            triage_timeout_ = 'triage_duration'
        timeout = self.params[by][triage_timeout_]
        yield self.env.timeout(abs(np.random.normal(timeout * 60, 1 * 60)))

    def examination(self, patient):
        yield self.env.timeout(abs(np.random.normal(30 * 60, 5 * 60)))

    def treatment(self, patient):
        yield self.env.timeout(abs(np.random.normal(30 * 60, 5 * 60)))


class Log(object):

    def __init__(self):
        self.log = {}

    def add_patient(self, env, admit_info):
        # get patient info
        patient_info = admit_info['patient_info']
        mrn = patient_info['mrn']
        admit_id = patient_info['admit_id']
        service = admit_info['service']
        admit_date = admit_info['date']
        patient_id = str(int(mrn)) + ' ' + str(int(admit_id))
        arrival_time = date_time.get_datetime(env.now).strftime("%Y-%m-%d %H:%M:%S")
        self.log[patient_id] = {}
        self.log[patient_id]['mrn'] = str(mrn)
        self.log[patient_id]['admit_id'] = admit_id
        self.log[patient_id]['admit_date'] = admit_date
        self.log[patient_id]['service'] = service
        self.log[patient_id]['arrival_time'] = arrival_time

    def add_admit_info(self, env, admit_info):
        # get patient info
        patient_info = admit_info['patient_info']
        mrn = patient_info['mrn']
        admit_id = patient_info['admit_id']
        patient_id = str(int(mrn)) + ' ' + str(int(admit_id))
        bed_assign_time = date_time.get_datetime(env.now).strftime("%Y-%m-%d %H:%M:%S")
        self.log[patient_id]['bed'] = admit_info['bed']
        self.log[patient_id]['floor'] = admit_info['floor']
        self.log[patient_id]['bed_assign_time'] = bed_assign_time

    def add_discharge_info(self, env, discharge_dict):
        mrn = discharge_dict['patient_info']['mrn']
        admit_id = discharge_dict['patient_info']['admit_id']
        discharge_date = discharge_dict['date']
        patient_id = str(int(mrn)) + ' ' + str(int(admit_id))
        self.log[patient_id]['discharge_date'] = discharge_date
        self.log[patient_id]['rounding team'] = discharge_dict['rounding_team']
        self.log[patient_id]['rounding MD'] = discharge_dict['discharge_md']
        self.log[patient_id]['rounding APP'] = discharge_dict['discharge_app']
        self.log[patient_id]['rounding_floor_adder'] = discharge_dict['floor_adder']
        self.log[patient_id]['rounding_queue'] = discharge_dict['round_queue']

    def add_rounding(self, start, end, patient_info):
        start_time = date_time.get_datetime(start).strftime("%Y-%m-%d %H:%M:%S")
        end_time = date_time.get_datetime(end).strftime("%Y-%m-%d %H:%M:%S")
        patient_id = patient_info[0]
        self.log[patient_id]['rounding_begin_time'] = start_time
        self.log[patient_id]['rounding_end_time'] = end_time

    def add_discharge_order(self, start, end, patient_info, app_info):
        start_time = date_time.get_datetime(start).strftime("%Y-%m-%d %H:%M:%S")
        end_time = date_time.get_datetime(end).strftime("%Y-%m-%d %H:%M:%S")
        patient_id = patient_info[0]
        staff = 'APP_' + str(app_info[0]) + '_' + str(app_info[1])
        self.log[patient_id]['discharge_process_begin_time'] = start_time
        self.log[patient_id]['discharge_process_end_time'] = end_time
        self.log[patient_id]['discharge_process_app'] = staff

    def add_discharge_process(self, start, end, patient_info, nurse_info):
        start_time = date_time.get_datetime(start).strftime("%Y-%m-%d %H:%M:%S")
        end_time = date_time.get_datetime(end).strftime("%Y-%m-%d %H:%M:%S")
        patient_id = patient_info[0]
        staff = 'Nurse_' + str(nurse_info[0]) + '_' + str(nurse_info[1])
        self.log[patient_id]['discharge_process_begin_time'] = start_time
        self.log[patient_id]['discharge_process_end_time'] = end_time
        self.log[patient_id]['discharge_process_nurse'] = staff

    def add_case_management(self, start, end, patient_info, cm_info):
        start_time = date_time.get_datetime(start).strftime("%Y-%m-%d %H:%M:%S")
        end_time = date_time.get_datetime(end).strftime("%Y-%m-%d %H:%M:%S")
        patient_id = patient_info[0]
        staff = 'CM_' + str(cm_info[0]) + '_' + str(cm_info[1])
        self.log[patient_id]['case_management_begin_time'] = start_time
        self.log[patient_id]['case_management_end_time'] = end_time
        self.log[patient_id]['case_management_staff'] = staff

    def add_transport(self, start, end, patient_info, cm_info):
        start_time = date_time.get_datetime(start).strftime("%Y-%m-%d %H:%M:%S")
        end_time = date_time.get_datetime(end).strftime("%Y-%m-%d %H:%M:%S")
        patient_id = patient_info[0]
        staff = 'CM_' + str(cm_info[0]) + '_' + str(cm_info[1])
        self.log[patient_id]['transport_begin_time'] = start_time
        self.log[patient_id]['transport_end_time'] = end_time
        self.log[patient_id]['transport_staff'] = staff

    def add_equipment(self, start, end, patient_info, cm_info):
        start_time = date_time.get_datetime(start).strftime("%Y-%m-%d %H:%M:%S")
        end_time = date_time.get_datetime(end).strftime("%Y-%m-%d %H:%M:%S")
        patient_id = patient_info[0]
        staff = 'CM_' + str(cm_info[0]) + '_' + str(cm_info[1])
        self.log[patient_id]['equipment_begin_time'] = start_time
        self.log[patient_id]['equipment_end_time'] = end_time
        self.log[patient_id]['equipment_staff'] = staff

    def add_medication(self, start, end, patient_info, cm_info):
        start_time = date_time.get_datetime(start).strftime("%Y-%m-%d %H:%M:%S")
        end_time = date_time.get_datetime(end).strftime("%Y-%m-%d %H:%M:%S")
        patient_id = patient_info[0]
        staff = 'Pharmacist_' + str(cm_info[0]) + '_' + str(cm_info[1])
        self.log[patient_id]['pharmacist_begin_time'] = start_time
        self.log[patient_id]['pharmacist_end_time'] = end_time
        self.log[patient_id]['pharmacist_staff'] = staff

    def add_bed_cleanup(self, start, end, patient_info, cm_info):
        start_time = date_time.get_datetime(start).strftime("%Y-%m-%d %H:%M:%S")
        end_time = date_time.get_datetime(end).strftime("%Y-%m-%d %H:%M:%S")
        patient_id = patient_info[0]
        staff = 'BedStaff_' + str(cm_info[0]) + '_' + str(cm_info[1])
        self.log[patient_id]['bedstaff_begin_time'] = start_time
        self.log[patient_id]['bedstaff_end_time'] = end_time
        self.log[patient_id]['bedstaff_staff'] = staff

    def end_discharge(self, discharge_time, patient_info):
        discharge_time = date_time.get_datetime(discharge_time).strftime("%Y-%m-%d %H:%M:%S")
        patient_id = patient_info[0]
        self.log[patient_id]['final_discharge_time'] = discharge_time

    def add_begin_registration(self, env, patient):
        _ = date_time.get_datetime(env.now).strftime("%Y-%m-%d %H:%M:%S")
        self.log[patient]['registration_begin_time'] = _

    def add_end_registration(self, env, patient):
        _ = date_time.get_datetime(env.now).strftime("%Y-%m-%d %H:%M:%S")
        self.log[patient]['registration_end_time'] = _

    def add_lobby_queue(self, env, patient, count):
        self.log[patient]['lobby_queue'] = count

    def add_begin_triage(self, env, patient):
        _ = date_time.get_datetime(env.now).strftime("%Y-%m-%d %H:%M:%S")
        self.log[patient]['first_triage_begin_time'] = _

    def add_end_triage(self, env, patient):
        _ = date_time.get_datetime(env.now).strftime("%Y-%m-%d %H:%M:%S")
        self.log[patient]['first_triage_end_time'] = _

    def add_triage_staff(self, env, patient, staff, type='Nurse'):
        self.log[patient]['triage_' + type.lower()] = type + '_' + staff.shift + '_' + str(staff.number)

    def add_triage_staff_count(self, env, patient, shift, count, type='Nurse'):
        self.log[patient]['triage_' + type.lower() + '_shift'] = shift
        self.log[patient]['triage_' + type.lower() + '_count'] = count

    def add_begin_examination(self, env, patient):
        _ = date_time.get_datetime(env.now).strftime("%Y-%m-%d %H:%M:%S")
        self.log[patient]['examination_begin_time'] = _

    def add_end_examination(self, env, patient):
        _ = date_time.get_datetime(env.now).strftime("%Y-%m-%d %H:%M:%S")
        self.log[patient]['examination_end_time'] = _

    def add_examination_staff(self, env, patient, staff, type='APP'):
        self.log[patient]['examination_' + type.lower()] = type + '_' + staff.shift + '_' + str(staff.number)

    def add_examination_staff_count(self, env, patient, shift, count, type='APP'):
        self.log[patient]['examination_' + type.lower() + '_shift'] = shift
        self.log[patient]['examination_' + type.lower() + '_count'] = count

    def add_begin_treatment(self, env, patient):
        _ = date_time.get_datetime(env.now).strftime("%Y-%m-%d %H:%M:%S")
        self.log[patient]['treatment_begin_time'] = _

    def add_end_treatment(self, env, patient):
        _ = date_time.get_datetime(env.now).strftime("%Y-%m-%d %H:%M:%S")
        self.log[patient]['treatment_end_time'] = _

    def add_treatment_staff(self, env, patient, staff, type='APP'):
        self.log[patient]['treatment_' + type.lower()] = type + '_' + staff.shift + '_' + str(staff.number)

    def add_treatment_staff_count(self, env, patient, shift, count, type='APP'):
        self.log[patient]['treatment_' + type.lower() + '_shift'] = shift
        self.log[patient]['treatment_' + type.lower() + '_count'] = count

    def publish_log(self, filepath="log.csv"):
        df = pd.DataFrame.from_dict(self.log, orient='index')
        if df.empty:
            return
        df = df[df.mrn != '1000']
        df['medication'] = ~df.pharmacist_staff.isna()
        df['cm'] = ~df.case_management_staff.isna()
        df['transport'] = ~df.transport_staff.isna()
        df['equipment'] = ~df.equipment_staff.isna()
        time_cols = [x for x in df.columns if x.endswith('_time')]
        for col in time_cols:
            df.loc[:, col] = pd.to_datetime(df[col])
        df['discharge_month'] = df['final_discharge_time'].dt.month
        df['discharge_day'] = df['final_discharge_time'].dt.day
        df['discharge_hour'] = df['final_discharge_time'].dt.hour
        df['discharge_dayofweek'] = df['final_discharge_time'].dt.dayofweek
        df['arrival_month'] = df['arrival_time'].dt.month
        df['arrival_day'] = df['arrival_time'].dt.day
        df['arrival_hour'] = df['arrival_time'].dt.hour
        df['arrival_dayofweek'] = df['arrival_time'].dt.dayofweek
        # df['registration_duration'] = (df.registration_end_time - df.registration_begin_time).astype('timedelta64[s]') / 60
        # df['wait_lobby_duration'] = (df.first_triage_begin_time - df.registration_end_time).astype('timedelta64[s]') / 60
        # df['first_triage_duration'] = (df.first_triage_end_time - df.first_triage_begin_time).astype('timedelta64[s]') / 60
        # if 'treatment_end_time' in df.columns:
        df['bed_wait_mins'] = (df.bed_assign_time - df.arrival_time).astype('timedelta64[s]') / 60
        df['LOS_hours'] = (df.final_discharge_time - df.bed_assign_time).astype('timedelta64[s]') / 60 / 60
        df.index.name = 'id'
        print(df.discharge_hour.describe())
        print(df.groupby('discharge_dayofweek').agg(discharge_time=('discharge_hour', 'median')))
        df.to_csv(filepath)

        # print(df.groupby(['service', 'medication', 'cm', 'equipment', 'transport']).agg(
        #     discharge_hour = ('discharge_hour', np.median),
        #     discharge_count = ('discharge_hour', 'count')
        # ))

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(5, 10))
        df.arrival_hour.hist(ax=ax[0], cumulative=True, density=True)
        ax[0].set_title('Arrival Distribution')
        df.discharge_hour.hist(ax=ax[1], cumulative=True, density=True)
        ax[1].set_title('Discharge Distribution')
        plt.show()

        fig, ax = plt.subplots(nrows=2, ncols=1, sharex=False, figsize=(5, 10))
        df.bed_wait_mins.hist(ax=ax[0])
        ax[0].set_title('Bed Wait (min) Distribution')
        df.LOS_hours.hist(ax=ax[1])
        ax[1].set_title('LOS (hours) Distribution')
        plt.show()

        plt.plot(df.arrival_hour, df.bed_wait_mins, 'ro')
        plt.ylim(0, 500)
        plt.show()
