import numpy as np
import pandas as pd
import simpy
import argparse
from model import Inpatient_Hospital, Log
from process import *
from utils import parameters, date_time


def start_admission(env, params, hosp, log_data):
    # today's date time
    day, hour, minute = date_time.get_day_hour_min(env.now)
    date = date_time.get_datetime(env.now).strftime("%m-%d-%Y")

    # create patient profile
    mrn = np.random.randint(low=1e6, high=1e7 - 1)
    admit_id = np.random.randint(low=1e6, high=1e7 - 1)
    patient_id = str(mrn) + ' ' + str(admit_id)
    service = np.random.choice(['A', 'B'], p=[0.6, 0.4])
    print(f"{date_time.get_datetime(env.now)} Patient {patient_id} service {service} arrived")
    admit_info = {
        'date': date,
        'patient_info': {'mrn': mrn, 'admit_id': admit_id},
        'service': service
    }
    log_data.add_patient(env, admit_info)

    # add admission process
    # log_data.add_patient(env, admit_info)
    bed = yield hosp.bed.get()
    hosp.occupied_beds.append(bed)
    inpatient_df = pd.read_csv(params['inpatient']['details'])
    inpatient_df.loc[(inpatient_df.bed == bed.number) & (inpatient_df.floor == bed.floor), 'mrn'] = mrn
    inpatient_df.loc[(inpatient_df.bed == bed.number) & (inpatient_df.floor == bed.floor), 'admit_id'] = admit_id
    inpatient_df.loc[(inpatient_df.bed == bed.number) & (inpatient_df.floor == bed.floor), 'service'] = service
    inpatient_df.loc[(inpatient_df.bed == bed.number) & (inpatient_df.floor == bed.floor), 'ready_to_discharge'] = 'no'
    inpatient_df.to_csv(params['inpatient']['details'], index=False)
    admit_info['bed'] = bed.number
    admit_info['floor'] = bed.floor
    print(
        f"{date_time.get_datetime(env.now)} Patient {patient_id} service {service} assigned bed {bed.number} floor {bed.floor}")
    log_data.add_admit_info(env, admit_info)

    # LOS
    yield env.timeout(15 * 60 * 60)

    inpatient_df = pd.read_csv(params['inpatient']['details'])
    inpatient_df.loc[(inpatient_df.bed == bed.number) & (inpatient_df.floor == bed.floor), 'ready_to_discharge'] = 'yes'
    inpatient_df.to_csv(params['inpatient']['details'], index=False)


def start_discharge(env, params, hosp, log_data):
    # today's date time
    day, hour, minute = date_time.get_day_hour_min(env.now)
    date = date_time.get_datetime(env.now).strftime("%m-%d-%Y")
    # print(date, day, hour, minute)

    # Make today's rounding teams
    # Fetch MDs, APPs available for rounding
    apps, mds = [], []
    factory_ = {'md': hosp.md_factory, 'app': hosp.app_factory}
    factory = {'md': hosp.md, 'app': hosp.app}
    for staff_type, staff_list in zip(['md', 'app'], [mds, apps]):
        shift = factory_[staff_type].get_shift(day, hour)
        shift_count = factory_[staff_type].get_shift_count(shift)
        for i in range(shift_count):
            staff = yield factory[staff_type].get(lambda _: (_.shift == shift) & (_.number <= shift_count))
            staff_list.append(staff)
            # print(f"{date_time.get_datetime(env.now)} {staff_type} {staff.shift, staff.number} is ready")
    n_mds = len(mds)
    n_apps = len(apps)
    # print(f"Number of MDs , APPs available for rounding {n_mds}, {n_apps}")

    # Build Rounding Teams
    n_teams = n_mds
    n_apps_per_team = n_apps // n_teams

    rounding_teams = {i: {'mds': [i], 'apps': []} for i in range(n_mds)}
    for team in rounding_teams.keys():
        rounding_teams[team]['n_patients'] = 0
        rounding_teams[team]['n_floors'] = 0
        rounding_teams[team]['patients'] = {'mrn': [], 'bed': [], 'floor': [],
                                            'service': [], 'floor_adder': [],
                                            'patient_adder': []}
        low_idx = team * n_apps_per_team
        high_idx = min(n_apps, (team + 1) * n_apps_per_team)
        high_idx = n_apps if team == n_mds - 1 else high_idx
        rounding_teams[team]['apps'] = list(range(n_apps)[low_idx:high_idx])
    rounding_teams['date'] = date

    # Free Staff after building Rounding Teams
    for staff_type, staff_list in zip(['md', 'app'], [mds, apps]):
        for staff in staff_list:
            yield factory[staff_type].put(staff)
            # print(f"{date_time.get_datetime(env.now)} {staff_type} {staff.shift, staff.number} is done")

    # Read Today's Patient Data
    discharge_patient_df = get_discharge_list(params)

    # Kick off discharge for patients
    for _, row in discharge_patient_df.iterrows():

        # get patient details
        mrn = row['mrn']
        admit_id = row['admit_id']
        bed_number = row['bed']
        floor = row['floor']
        service = row['service']

        # get bed
        bed = None
        for Bed in hosp.occupied_beds:
            if Bed.floor == floor and Bed.number == bed_number:
                bed = Bed
                print(f"{date_time.get_datetime(env.now)} Bed {bed} is now granted")

        # assign patient to rounding team
        team_idx = _ % n_teams

        # get rounding team's queue
        if rounding_teams[team_idx]['patients']['mrn'] == []:
            prev_floor = None
            floor_adder = 0
            patient_adder = 0
        else:
            prev_floor = rounding_teams[team_idx]['patients']['floor'][-1]
            floor_adder = rounding_teams[team_idx]['patients']['floor_adder'][-1]
            patient_adder = rounding_teams[team_idx]['patients']['patient_adder'][-1]

        # update floor adder
        floor_adder = floor_adder + np.abs(floor - prev_floor) if prev_floor is not None else floor_adder
        rounding_teams[team_idx]['patients']['floor_adder'].append(floor_adder)

        # update patient_adder
        patient_adder += 1
        rounding_teams[team_idx]['patients']['patient_adder'].append(patient_adder)

        # add patient to rounding team's list
        rounding_teams[team_idx]['patients']['mrn'].append(mrn)
        rounding_teams[team_idx]['patients']['bed'].append(bed_number)
        rounding_teams[team_idx]['patients']['floor'].append(floor)
        rounding_teams[team_idx]['patients']['service'].append(service)

        # allot patient to APP
        n_apps_team = len(rounding_teams[team_idx]['apps'])
        discharge_info = {
            'date': date,
            'patient_info': {'mrn': mrn, 'admit_id': admit_id, 'bed': bed, 'floor': floor, 'service': service},
            'discharge_md': rounding_teams[team_idx]['mds'][0] + 1,
            'discharge_app': rounding_teams[team_idx]['apps'][patient_adder % n_apps_team - 1] + 1,
            'round_queue': patient_adder,
            'floor_adder': floor_adder,
            'rounding_team': team_idx
        }
        log_data.add_discharge_info(env, discharge_info)
        env.process(discharge_patient(env, params, discharge_info, hosp, log_data, verbose=False))
    # yield env.timeout((24 * 60 * 60) - env.now % (24 * 60 * 60))


def run_simulation(env, params, log_data):
    # instantiate model classes
    hosp = Inpatient_Hospital(env, params)

    # arrival info
    arrival_df = pd.read_csv(params['inpatient']['arrival'], index_col=0)
    closing_admission_hour = 20

    # start with occupied beds
    # env.process(add_inpatients(env, params, hosp, log_data, count=5))
    inpatient_df = pd.read_csv(params['inpatient']['details'])
    inpatient_df.loc[:, 'mrn'] = pd.NA
    inpatient_df.loc[:, 'admit_id'] = pd.NA
    inpatient_df.loc[:, 'service'] = pd.NA
    inpatient_df.loc[:, 'ready_to_discharge'] = pd.NA
    for floor in inpatient_df.floor.unique():
        for bed_number in np.random.choice(inpatient_df.loc[inpatient_df.floor == floor, 'bed'].values, 5,
                                           replace=False):
            mrn = 1000
            admit_id = np.random.randint(low=1e6, high=1e7 - 1)
            service = np.random.choice(['A', 'B'], p=[0.5, 0.5])
            date = date_time.get_datetime(env.now).strftime("%m-%d-%Y")
            inpatient_df.loc[(inpatient_df.floor == floor) & (inpatient_df.bed == bed_number), 'mrn'] = mrn
            inpatient_df.loc[(inpatient_df.floor == floor) & (inpatient_df.bed == bed_number), 'admit_id'] = admit_id
            inpatient_df.loc[(inpatient_df.floor == floor) & (inpatient_df.bed == bed_number), 'service'] = service
            inpatient_df.loc[
                (inpatient_df.floor == floor) & (inpatient_df.bed == bed_number), 'ready_to_discharge'] = 'yes'
            bed = yield hosp.bed.get(lambda Bed: (Bed.floor == floor) & (Bed.number == bed_number))
            hosp.occupied_beds.append(bed)
            admit_info = {
                'date': date,
                'patient_info': {'mrn': mrn, 'admit_id': admit_id},
                'service': service,
                'bed': bed.number,
                'floor': bed.floor
            }
            log_data.add_patient(env, admit_info)
            log_data.add_admit_info(env, admit_info)
    inpatient_df.to_csv(params['inpatient']['details'], index=False)

    while True:
        day, hour = date_time.get_day_hour(env.now)
        arrival_rate = arrival_df.loc[day, str(hour)] + 1e-2
        rounding_start = params['rounding_start']['mean']
        rounding_start_spread = np.random.choice([-2, -1, 0, 1, 2], p=[0.1, 0.2, 0.4, 0.2, 0.1]) * \
                                params['rounding_start']['std']
        rounding_complete = False
        while hour < closing_admission_hour:
            time_to_rounding = rounding_start * 60 * 60 + rounding_start_spread * 60 - env.now % (24 * 60 * 60)
            rounding_timeout = time_to_rounding if time_to_rounding >= 0 and not rounding_complete else 24 * 60 * 60
            next_arrival_timeout = 1 / arrival_rate
            t1 = env.timeout(rounding_timeout, value='rounding')
            t2 = env.timeout(next_arrival_timeout * 60 * 60, value='repeat')
            t3 = env.timeout((1 * 60 * 60) - env.now % (1 * 60 * 60))
            res = yield t1 | t2 | t3
            if t1 in res.keys():
                print(date_time.get_datetime(env.now), 'discharge')
                env.process(start_discharge(env, params, hosp, log_data))
                rounding_complete = True
                day, hour = date_time.get_day_hour(env.now)
                # update arrival
                arrival_rate = arrival_df.loc[day, str(hour)] + 1e-2
            elif t2 in res.keys():
                env.process(start_admission(env, params, hosp, log_data))
                day, hour = date_time.get_day_hour(env.now)
                arrival_rate = arrival_df.loc[day, str(hour)] + 1e-2
            else:
                day, hour = date_time.get_day_hour(env.now)
                arrival_rate = arrival_df.loc[day, str(hour)] + 1e-2

        yield env.timeout((24 * 60 * 60) - env.now % (24 * 60 * 60))


def main():
    # parameters
    parser = parameters.arg_parser()
    config = parser.parse_args().params
    params = parameters.get_params(config)

    # start logger
    log_data = Log()

    # run simulation
    env = simpy.Environment()
    env.process(run_simulation(env, params, log_data))
    env.run(until=60 * 60 * 24 * 365 * 10)  # 60 secs * 60 mins * 24 hours * 1 day
    log_data.publish_log()


if __name__ == '__main__':
    main()
