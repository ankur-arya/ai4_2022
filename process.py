from utils import date_time
import numpy as np
import pandas as pd


def get_discharge_list(params):
    inpatient_df = pd.read_csv(params['inpatient']['details'])
    discharge_patient_df = inpatient_df[inpatient_df.ready_to_discharge == 'yes']
    return discharge_patient_df


def assign_bed(env, hosp, log_data, admit_info, params, verbose=False):
    inpatient_df = pd.read_csv(params['inpatient']['details'])
    bed = yield hosp.bed.get()
    print(admit_info, bed)
    yield hosp.bed.put(bed)


def remove_inpatient(env, params, hosp, bed, floor):
    bed_number = bed.number
    yield hosp.bed.put(bed)
    print(f"{date_time.get_datetime(env.now)} Bed {bed} is now free")
    hosp.occupied_beds = [Bed for Bed in hosp.occupied_beds if Bed != bed]

    inpatient_df = pd.read_csv(params['inpatient']['details'])
    inpatient_df.loc[(inpatient_df.bed == bed_number) & (inpatient_df.floor == floor), 'mrn'] = pd.NA
    inpatient_df.loc[(inpatient_df.bed == bed_number) & (inpatient_df.floor == floor), 'admit_id'] = pd.NA
    inpatient_df.loc[(inpatient_df.bed == bed_number) & (inpatient_df.floor == floor), 'service'] = pd.NA
    inpatient_df.loc[(inpatient_df.bed == bed_number) & (inpatient_df.floor == floor), 'ready_to_discharge'] = pd.NA
    inpatient_df.to_csv(params['inpatient']['details'], index=False)


def discharge_rounds(env, hosp, log_data, patient_id, round_queue, floor_adder, verbose=False):
    start = env.now
    yield env.timeout(hosp.rounding_patient_timeout * round_queue * 60 +
                      hosp.rounding_floor_timeout * floor_adder * 60 + np.maximum(np.random.normal(0, 10 * 60, 1), 0))
    log_data.add_rounding(start, env.now, patient_id)
    if verbose:
        print(f"{date_time.get_datetime(env.now)} Patient {patient_id}'s rounding just completed")


def discharge_notes(env, hosp, log_data, discharge_app, patient_id, verbose=False):
    day, hour = date_time.get_day_hour(env.now)
    shift = hosp.app_factory.get_shift(day, hour)
    app = yield hosp.app.get(lambda app: (app.shift == shift) & (app.number == discharge_app))
    if verbose:
        print(
            f"{date_time.get_datetime(env.now)} Patient {patient_id} has been assigned APP {app.shift, app.number} for discharge order")
    start = env.now
    if verbose:
        print(
            f"{date_time.get_datetime(env.now)} Patient {patient_id} begin discharge order by APP {app.shift, app.number}")

    mean = hosp.discharge_order_timeout['mean']
    std = hosp.discharge_order_timeout['std']
    yield env.timeout(np.maximum(np.random.normal(loc=mean, scale=std) * 60, 10 * 60))

    if verbose:
        print(
            f"{date_time.get_datetime(env.now)} Patient {patient_id} end discharge order by APP {app.shift, app.number}")
    yield hosp.app.put(app)
    log_data.add_discharge_order(start, env.now, patient_id, (app.shift, app.number))
    if verbose:
        print(f"{date_time.get_datetime(env.now)} Patient {patient_id} is now done with APP {app.shift, app.number}")


def discharge_process(env, hosp, log_data, patient_id, home=True, verbose=False):
    day, hour = date_time.get_day_hour(env.now)
    shift = hosp.nurse_factory.get_shift(day, hour)
    shift_count = hosp.nurse_factory.get_shift_count(shift)
    nurse = yield hosp.nurse.get(lambda nurse: (nurse.shift == shift) & (nurse.number <= shift_count))
    start = env.now
    if verbose:
        print(
            f"{date_time.get_datetime(env.now)} Patient {patient_id} is assigned to Nurse {nurse.shift, nurse.number} for discharge work")
    if verbose:
        print(
            f"{date_time.get_datetime(env.now)} Patient {patient_id} begin discharge work items by Nurse {nurse.shift, nurse.number}")
    mean = hosp.discharge_process_timeout['mean'][0] if home else hosp.discharge_process_timeout['mean'][1]
    std = hosp.discharge_process_timeout['std'][0] if home else hosp.discharge_process_timeout['std'][1]
    yield env.timeout(np.maximum(np.random.normal(loc=mean, scale=std) * 60, 10 * 60))
    if verbose:
        print(
            f"{date_time.get_datetime(env.now)} Patient {patient_id} end discharge order items by Nurse {nurse.shift, nurse.number}")
    yield hosp.nurse.put(nurse)
    log_data.add_discharge_process(start, env.now, patient_id, (nurse.shift, nurse.number))
    if verbose:
        print(
            f"{date_time.get_datetime(env.now)} Patient {patient_id} is now done with Nurse {nurse.shift, nurse.number}")


def case_management(env, hosp, log_data, patient_id, verbose=False):
    day, hour = date_time.get_day_hour(env.now)
    shift = hosp.cm_factory.get_shift(day, hour)
    shift_count = hosp.cm_factory.get_shift_count(shift)
    cm = yield hosp.cm.get(lambda cm: (cm.shift == shift) & (cm.number <= shift_count))
    if verbose:
        print(
            f"{date_time.get_datetime(env.now)} Patient {patient_id} is now assigned to CM {cm.number} for Case Management")
    start = env.now
    if verbose:
        print(
            f"{date_time.get_datetime(env.now)} Patient {patient_id} begin case management items by CM {cm.shift, cm.number}")
    mean = hosp.case_management_timeout['mean']
    std = hosp.case_management_timeout['std']
    yield env.timeout(np.maximum(np.random.normal(loc=mean, scale=std) * 60, 10 * 60))
    if verbose:
        print(
            f"{date_time.get_datetime(env.now)} Patient {patient_id} end case management items by CM {cm.shift, cm.number}")
    yield hosp.cm.put(cm)
    log_data.add_case_management(start, env.now, patient_id, (cm.shift, cm.number))
    if verbose:
        print(f"{date_time.get_datetime(env.now)} Patient {patient_id} is now done with CM {cm.shift, cm.number}")


def transport(env, hosp, log_data, patient_id, verbose=False):
    day, hour = date_time.get_day_hour(env.now)
    shift = hosp.cm_factory.get_shift(day, hour)
    shift_count = hosp.cm_factory.get_shift_count(shift)
    cm = yield hosp.cm.get(lambda cm: (cm.shift == shift) & (cm.number <= shift_count))
    start = env.now
    if verbose:
        print(f"{date_time.get_datetime(env.now)} Patient {patient_id} is now assigned to CM {cm.number} for Transport")
    if verbose:
        print(f"{date_time.get_datetime(env.now)} Patient {patient_id} begin Transport by CM {cm.shift, cm.number}")
    mean = hosp.transport_timeout['mean']
    std = hosp.transport_timeout['std']
    yield env.timeout(np.maximum(np.random.normal(loc=mean, scale=std) * 60, 10 * 60))
    if verbose:
        print(f"{date_time.get_datetime(env.now)} Patient {patient_id} end Transport by CM {cm.shift, cm.number}")
    yield hosp.cm.put(cm)
    log_data.add_transport(start, env.now, patient_id, (cm.shift, cm.number))
    if verbose:
        print(f"{date_time.get_datetime(env.now)} Patient {patient_id} is now done with CM {cm.shift, cm.number}")


def equipment(env, hosp, log_data, patient_id, verbose=False):
    day, hour = date_time.get_day_hour(env.now)
    shift = hosp.cm_factory.get_shift(day, hour)
    shift_count = hosp.cm_factory.get_shift_count(shift)
    cm = yield hosp.cm.get(lambda cm: (cm.shift == shift) & (cm.number <= shift_count))
    start = env.now
    if verbose:
        print(f"{date_time.get_datetime(env.now)} Patient {patient_id} is now assigned to CM {cm.number} for Equipment")
    if verbose:
        print(f"{date_time.get_datetime(env.now)} Patient {patient_id} begin Equipment by CM {cm.shift, cm.number}")
    mean = hosp.equipment_timeout['mean']
    std = hosp.equipment_timeout['std']
    yield env.timeout(np.maximum(np.random.normal(loc=mean, scale=std) * 60, 10 * 60))
    if verbose:
        print(f"{date_time.get_datetime(env.now)} Patient {patient_id} end Equipment by CM {cm.shift, cm.number}")
    yield hosp.cm.put(cm)
    log_data.add_equipment(start, env.now, patient_id, (cm.shift, cm.number))
    if verbose:
        print(f"{date_time.get_datetime(env.now)} Patient {patient_id} is now done with CM {cm.shift, cm.number}")


def medication(env, hosp, log_data, patient_id, verbose=False):
    day, hour = date_time.get_day_hour(env.now)
    shift = hosp.pharmacist_factory.get_shift(day, hour)
    shift_count = hosp.pharmacist_factory.get_shift_count(shift)
    pharmacist = yield hosp.pharmacist.get(
        lambda pharmacist: (pharmacist.shift == shift) & (pharmacist.number <= shift_count))
    start = env.now
    if verbose:
        print(
            f"{date_time.get_datetime(env.now)} Patient {patient_id} is now assigned to pharmacist {pharmacist.number} for Medication")
    if verbose:
        print(
            f"{date_time.get_datetime(env.now)} Patient {patient_id} begin Medication by pharmacist {pharmacist.shift, pharmacist.number}")
    mean = hosp.medication_timeout['mean']
    std = hosp.medication_timeout['std']
    yield env.timeout(np.maximum(np.random.normal(loc=mean, scale=std) * 60, 10 * 60))
    if verbose:
        print(
            f"{date_time.get_datetime(env.now)} Patient {patient_id} end Medication by pharmacist {pharmacist.shift, pharmacist.number}")
    yield hosp.pharmacist.put(pharmacist)
    log_data.add_medication(start, env.now, patient_id, (pharmacist.shift, pharmacist.number))
    if verbose:
        print(
            f"{date_time.get_datetime(env.now)} Patient {patient_id} is now done with pharmacist {pharmacist.shift, pharmacist.number}")


def bed_cleanup(env, hosp, log_data, patient_id, verbose=False):
    day, hour = date_time.get_day_hour(env.now)
    shift = hosp.bed_staff_factory.get_shift(day, hour)
    shift_count = hosp.bed_staff_factory.get_shift_count(shift)
    bed_staff = yield hosp.bed_staff.get(
        lambda bed_staff: (bed_staff.shift == shift) & (bed_staff.number <= shift_count))
    start = env.now
    if verbose:
        print(
            f"{date_time.get_datetime(env.now)} Patient {patient_id} is now assigned to bed_staff {bed_staff.number} for Bed Clean Up")
    if verbose:
        print(
            f"{date_time.get_datetime(env.now)} Patient {patient_id} begin Bed Clean by bed_staff {bed_staff.shift, bed_staff.number}")
    mean = hosp.bed_clean_timeout['mean']
    std = hosp.bed_clean_timeout['std']
    yield env.timeout(np.maximum(np.random.normal(loc=mean, scale=std) * 60, 10 * 60))
    if verbose:
        print(
            f"{date_time.get_datetime(env.now)} Patient {patient_id} end Bed Clean up by bed_staff {bed_staff.shift, bed_staff.number}")
    yield hosp.bed_staff.put(bed_staff)
    log_data.add_bed_cleanup(start, env.now, patient_id, (bed_staff.shift, bed_staff.number))
    if verbose:
        print(
            f"{date_time.get_datetime(env.now)} Patient {patient_id} is now done with bed_staff {bed_staff.shift, bed_staff.number}")


def admit_patient(env, params, admit_info, hosp, log_data, verbose=True):
    # get patient info
    patient_info = admit_info['patient_info']
    mrn = patient_info['mrn']
    admit_id = patient_info['admit_id']
    service = admit_info['service']
    date = admit_info['date']
    if verbose:
        print(
            f"{date_time.get_datetime(env.now)} Patient {mrn} {admit_id} is now admitted")
    # yield env.timeout(30*60*60)
    # yield env.process(assign_bed(env, hosp, log_data, admit_info, params, verbose=False))

    # get nurse triage
    # get app triage
    # get md admit
    # get bed


def discharge_patient(env, params, discharge_info, hosp, log_data, verbose=False):
    # get patient info
    patient_info = discharge_info['patient_info']
    mrn = patient_info['mrn']
    admit_id = patient_info['admit_id']
    service = patient_info['service']
    floor = patient_info['floor']
    bed = patient_info['bed']
    date = discharge_info['date']
    patient_id = [str(int(mrn)) + ' ' + str(int(admit_id))]

    # home floor
    home = (floor == 0 and service == 'A') or (floor == 1 and service == 'B')

    # discharge info
    discharge_md = discharge_info['discharge_md']
    discharge_app = discharge_info['discharge_app']
    round_queue = discharge_info['round_queue']
    floor_adder = discharge_info['floor_adder']

    # discharge rounding
    yield env.process(discharge_rounds(env, hosp, log_data, patient_id, round_queue, floor_adder, verbose=False))

    # discharge notes by APP
    yield env.process(discharge_notes(env, hosp, log_data, discharge_app, patient_id, verbose=False))

    # discharge process by nurse
    yield env.process(discharge_process(env, hosp, log_data, patient_id, home, verbose=False))

    # roll dice
    cm_prob = np.random.choice([0, 1], p=[0.8, 0.2])

    # case_management
    if cm_prob:
        yield env.process(case_management(env, hosp, log_data, patient_id, verbose=False))

        # roll dice
        transport_prob = np.random.choice([0, 1], p=[0.5, 0.5])

        # transport
        if transport_prob:
            yield env.process(transport(env, hosp, log_data, patient_id, verbose=False))

        # roll dice
        equipment_prob = np.random.choice([0, 1], p=[0.5, 0.5])

        # equipment
        if equipment_prob:
            yield env.process(equipment(env, hosp, log_data, patient_id, verbose=False))

    # roll dice
    med_prob = np.random.choice([0, 1], p=[0.5, 0.5])

    # case_management
    if med_prob:
        yield env.process(medication(env, hosp, log_data, patient_id, verbose=False))

    # clear bed
    if verbose:
        print(f"{date_time.get_datetime(env.now)} Patient {patient_id} is now discharged and bed is being cleaned up")
    log_data.end_discharge(env.now, patient_id)
    yield env.process(bed_cleanup(env, hosp, log_data, patient_id, verbose=False))
    yield env.process(remove_inpatient(env, params, hosp, bed, floor))


def patient_walk_in(env, patient_df):
    day, hour = date_time.get_day_hour(env.now)
    count_per_hour = patient_df.loc[day, str(hour)]
    arrivals_per_sec = count_per_hour / 3600
    return arrivals_per_sec


def patient_arrival(env, patient, ed, log_data, arrival_mode, esi):
    # emergency severity index
    esi_label = ['ESI 1', 'ESI 2', 'ESI 3', 'ESI 4', 'ESI 5']

    if esi in [0, 1]:
        log_data.add_patient(env, patient, arrival_mode, esi_label[esi])
        env.process(high_emergency(env, patient, ed, log_data))

    elif esi in [2]:
        log_data.add_patient(env, patient, arrival_mode, esi_label[esi])
        env.process(mid_emergency(env, patient, ed, log_data))
        env.process(examination_treatment(env, patient, ed, log_data))

    else:
        log_data.add_patient(env, patient, arrival_mode, esi_label[esi])
        env.process(low_emergency(env, patient, ed, log_data))


def low_emergency(env, patient, ed, log_data):
    with ed.visit.request() as visit_request:
        yield visit_request

        print(f"{date_time.get_datetime(env.now)} Patient {patient} arrives at ED")

        # register at desk
        with ed.register_desk.request() as request:
            yield request
            print(f"{date_time.get_datetime(env.now)} Patient {patient} begins registration")
            start = env.now
            log_data.add_begin_registration(env, patient)
            yield env.process(ed.registration(patient))
            end = env.now
            print(
                f"{date_time.get_datetime(env.now)} Patient {patient} finished registration in {date_time.get_timedelta(start, end, units='min')} min")
            log_data.add_end_registration(env, patient)
        start = env.now
        print(f"{date_time.get_datetime(env.now)} Patient {patient} waiting in lobby")
        print(f"{date_time.get_datetime(env.now)} {len(ed.triage_bay.get_queue)} patients waiting in lobby")
        log_data.add_lobby_queue(env, patient, len(ed.triage_bay.get_queue))

        # triage bay
        triage_bay = yield ed.triage_bay.get(lambda bay: bay.number > 0)
        print(f"{date_time.get_datetime(env.now)} Triage bay {triage_bay.number} is available")

        # triage
        # nurse
        day, hour = date_time.get_day_hour(env.now)
        shift = ed.nurse_factory.get_shift(day, hour)
        shift_count = ed.nurse_factory.get_shift_count(shift)
        nurse = yield ed.nurse.get(lambda nurse: (nurse.shift == shift) & (nurse.number <= shift_count))
        print(f"{date_time.get_datetime(env.now)} Nurse {nurse.shift, nurse.number} is ready for triage")
        end = env.now
        print(
            f"{date_time.get_datetime(env.now)} Patient {patient} waited in lobby for {date_time.get_timedelta(start, end, units='min')} min")
        start = env.now
        log_data.add_begin_triage(env, patient)
        log_data.add_triage_staff(env, patient, nurse, type='Nurse')
        log_data.add_triage_staff_count(env, patient, shift, shift_count, type='Nurse')
        print(f"{date_time.get_datetime(env.now)} Patient {patient} begins triage by Nurse {nurse.shift, nurse.number}")
        triage_nurse_shift, triage_shift_number = nurse.shift, nurse.number
        yield env.process(ed.triage(patient, by='nurse'))
        print(
            f"{date_time.get_datetime(env.now)} Patient {patient} finishes triage by Nurse {nurse.shift, nurse.number}")
        end = env.now
        print(
            f"{date_time.get_datetime(env.now)} Patient {patient} Nurse {nurse.shift, nurse.number} finished triage in {date_time.get_timedelta(start, end, units='min')} min")
        yield ed.nurse.put(nurse)
        print(f"{date_time.get_datetime(env.now)} Nurse {nurse.shift, nurse.number} is now available")

        # app
        day, hour = date_time.get_day_hour(env.now)
        shift = ed.app_factory.get_shift(day, hour)
        shift_count = ed.app_factory.get_shift_count(shift)
        app = yield ed.app.get(lambda app: (app.shift == shift) & (app.number <= shift_count))
        print(f"{date_time.get_datetime(env.now)} APP {app.shift, app.number} is ready for triage")
        start = env.now
        print(f"{date_time.get_datetime(env.now)} Patient {patient} begins triage by APP {app.shift, app.number}")
        log_data.add_triage_staff(env, patient, app, type='APP')
        log_data.add_triage_staff_count(env, patient, shift, shift_count, type='APP')

        if app.number == triage_shift_number == 3:
            yield env.process(ed.triage(patient, by='app', longer=True))
        else:
            yield env.process(ed.triage(patient, by='app'))

        print(f"{date_time.get_datetime(env.now)} Patient {patient} finishes triage by APP {app.shift, app.number}")
        yield ed.app.put(app)
        print(f"{date_time.get_datetime(env.now)} APP {app.shift, app.number} is now available")
        end = env.now
        log_data.add_end_triage(env, patient)
        print(
            f"{date_time.get_datetime(env.now)} Patient {patient} finished triage by APP {app.shift, app.number} in {date_time.get_timedelta(start, end, units='min')} min")

        yield ed.triage_bay.put(triage_bay)
        print(f"{date_time.get_datetime(env.now)} Triage {triage_bay.number} is now free")


def mid_emergency(env, patient, ed, log_data):
    with ed.visit.request() as visit_request:
        yield visit_request

        print(f"{date_time.get_datetime(env.now)} Patient {patient} arrives at ED")

        # register at desk
        with ed.register_desk.request() as request:
            yield request
            print(f"{date_time.get_datetime(env.now)} Patient {patient} begins registration")
            start = env.now
            log_data.add_begin_registration(env, patient)
            yield env.process(ed.registration(patient))
            end = env.now
            print(
                f"{date_time.get_datetime(env.now)} Patient {patient} finished registration in {date_time.get_timedelta(start, end, units='min')} min")
            log_data.add_end_registration(env, patient)
        start = env.now
        print(f"{date_time.get_datetime(env.now)} Patient {patient} waiting in lobby")
        print(f"{date_time.get_datetime(env.now)} {len(ed.triage_bay.get_queue)} patients waiting in lobby")
        log_data.add_lobby_queue(env, patient, len(ed.triage_bay.get_queue))

        # triage bay
        triage_bay = yield ed.triage_bay.get(lambda bay: bay.number > 0)
        print(f"{date_time.get_datetime(env.now)} Triage bay {triage_bay.number} is available")

        # triage
        # nurse
        day, hour = date_time.get_day_hour(env.now)
        shift = ed.nurse_factory.get_shift(day, hour)
        shift_count = ed.nurse_factory.get_shift_count(shift)
        nurse = yield ed.nurse.get(lambda nurse: (nurse.shift == shift) & (nurse.number <= shift_count))
        print(f"{date_time.get_datetime(env.now)} Nurse {nurse.shift, nurse.number} is ready for triage")
        end = env.now
        print(
            f"{date_time.get_datetime(env.now)} Patient {patient} waited in lobby for {date_time.get_timedelta(start, end, units='min')} min")
        start = env.now
        log_data.add_begin_triage(env, patient)
        log_data.add_triage_staff(env, patient, nurse, type='Nurse')
        log_data.add_triage_staff_count(env, patient, shift, shift_count, type='Nurse')
        print(f"{date_time.get_datetime(env.now)} Patient {patient} begins triage by Nurse {nurse.shift, nurse.number}")
        triage_nurse_shift, triage_nurse_number = nurse.shift, nurse.number
        yield env.process(ed.triage(patient, by='nurse'))
        print(
            f"{date_time.get_datetime(env.now)} Patient {patient} finishes triage by Nurse {nurse.shift, nurse.number}")
        end = env.now
        print(
            f"{date_time.get_datetime(env.now)} Patient {patient} Nurse {nurse.shift, nurse.number} finished triage in {date_time.get_timedelta(start, end, units='min')} min")
        yield ed.nurse.put(nurse)
        print(f"{date_time.get_datetime(env.now)} Nurse {nurse.shift, nurse.number} is now available")

        # md
        day, hour = date_time.get_day_hour(env.now)
        shift = ed.md_factory.get_shift(day, hour)
        shift_count = ed.md_factory.get_shift_count(shift)
        md = yield ed.md.get(lambda md: (md.shift == shift) & (md.number <= shift_count))
        print(f"{date_time.get_datetime(env.now)} MD {md.shift, md.number} is ready for triage")
        start = env.now
        print(f"{date_time.get_datetime(env.now)} Patient {patient} begins triage by MD {md.shift, md.number}")
        if md.number == triage_nurse_number == 2:
            yield env.process(ed.triage(patient, by='md', longer=True))
        else:
            yield env.process(ed.triage(patient, by='md'))
        log_data.add_triage_staff(env, patient, md, type='MD')
        log_data.add_triage_staff_count(env, patient, shift, shift_count, type='MD')
        print(f"{date_time.get_datetime(env.now)} Patient {patient} finishes triage by MD {md.shift, md.number}")
        yield ed.md.put(md)
        print(f"{date_time.get_datetime(env.now)} MD {md.shift, md.number} is now available")
        end = env.now
        log_data.add_end_triage(env, patient)
        print(
            f"{date_time.get_datetime(env.now)} Patient {patient} finished triage by MD {md.shift, md.number} in {date_time.get_timedelta(start, end, units='min')} min")

        yield ed.triage_bay.put(triage_bay)
        print(f"{date_time.get_datetime(env.now)} Triage {triage_bay.number} is now free")


def high_emergency(env, patient, ed, log_data):
    with ed.visit.request() as visit_request:
        yield visit_request

        print(f"{date_time.get_datetime(env.now)} Patient {patient} arrives at ED")

        # register at desk
        with ed.register_desk.request() as request:
            yield request
            print(f"{date_time.get_datetime(env.now)} Patient {patient} begins registration")
            start = env.now
            log_data.add_begin_registration(env, patient)
            yield env.process(ed.registration(patient))
            end = env.now
            print(
                f"{date_time.get_datetime(env.now)} Patient {patient} finished registration in {date_time.get_timedelta(start, end, units='min')} min")
            log_data.add_end_registration(env, patient)
        start = env.now
        print(f"{date_time.get_datetime(env.now)} Patient {patient} waiting in lobby")
        print(f"{date_time.get_datetime(env.now)} {len(ed.triage_bay.get_queue)} patients waiting in lobby")
        log_data.add_lobby_queue(env, patient, len(ed.triage_bay.get_queue))

        # triage bay
        triage_bay = yield ed.triage_bay.get(lambda bay: bay.number > 0)
        print(f"{date_time.get_datetime(env.now)} Triage bay {triage_bay.number} is available")

        # triage
        # nurse
        day, hour = date_time.get_day_hour(env.now)
        shift = ed.nurse_factory.get_shift(day, hour)
        shift_count = ed.nurse_factory.get_shift_count(shift)
        nurse = yield ed.nurse.get(lambda nurse: (nurse.shift == shift) & (nurse.number <= shift_count))
        print(f"{date_time.get_datetime(env.now)} Nurse {nurse.shift, nurse.number} is ready for triage")
        end = env.now
        print(
            f"{date_time.get_datetime(env.now)} Patient {patient} waited in lobby for {date_time.get_timedelta(start, end, units='min')} min")
        start = env.now
        log_data.add_begin_triage(env, patient)
        log_data.add_triage_staff(env, patient, nurse, type='Nurse')
        log_data.add_triage_staff_count(env, patient, shift, shift_count, type='Nurse')
        print(f"{date_time.get_datetime(env.now)} Patient {patient} begins triage by Nurse {nurse.shift, nurse.number}")
        triage_nurse_shift, triage_nurse_number = nurse.shift, nurse.number
        yield env.process(ed.triage(patient, by='nurse'))
        print(
            f"{date_time.get_datetime(env.now)} Patient {patient} finishes triage by Nurse {nurse.shift, nurse.number}")
        end = env.now
        print(
            f"{date_time.get_datetime(env.now)} Patient {patient} Nurse {nurse.shift, nurse.number} finished triage in {date_time.get_timedelta(start, end, units='min')} min")
        yield ed.nurse.put(nurse)
        print(f"{date_time.get_datetime(env.now)} Nurse {nurse.shift, nurse.number} is now available")

        # md
        day, hour = date_time.get_day_hour(env.now)
        shift = ed.md_factory.get_shift(day, hour)
        shift_count = ed.md_factory.get_shift_count(shift)
        md = yield ed.md.get(lambda md: (md.shift == shift) & (md.number <= shift_count))
        print(f"{date_time.get_datetime(env.now)} MD {md.shift, md.number} is ready for triage")
        start = env.now
        print(f"{date_time.get_datetime(env.now)} Patient {patient} begins triage by MD {md.shift, md.number}")

        if md.number == triage_nurse_number == 2:
            yield env.process(ed.triage(patient, by='md', longer=True))
        else:
            yield env.process(ed.triage(patient, by='md'))

        log_data.add_triage_staff(env, patient, md, type='MD')
        log_data.add_triage_staff_count(env, patient, shift, shift_count, type='MD')
        print(f"{date_time.get_datetime(env.now)} Patient {patient} finishes triage by MD {md.shift, md.number}")
        yield ed.md.put(md)
        print(f"{date_time.get_datetime(env.now)} MD {md.shift, md.number} is now available")
        end = env.now
        log_data.add_end_triage(env, patient)
        print(
            f"{date_time.get_datetime(env.now)} Patient {patient} finished triage by MD {md.shift, md.number} in {date_time.get_timedelta(start, end, units='min')} min")

        yield ed.triage_bay.put(triage_bay)
        print(f"{date_time.get_datetime(env.now)} Triage {triage_bay.number} is now free")


def examination_treatment(env, patient, ed, log_data):
    with ed.treatment_room.request() as request:
        yield request
        print(f"{date_time.get_datetime(env.now)} Treatment room is available")

        # examination
        # app
        day, hour = date_time.get_day_hour(env.now)
        shift = ed.app_factory.get_shift(day, hour)
        shift_count = ed.app_factory.get_shift_count(shift)
        app = yield ed.app.get(lambda app: (app.shift == shift) & (app.number <= shift_count))
        print(f"{date_time.get_datetime(env.now)} APP {app.shift, app.number} is ready")
        start = env.now
        print(f"{date_time.get_datetime(env.now)} Patient {patient} begins examination by APP {app.shift, app.number}")
        log_data.add_begin_examination(env, patient)
        log_data.add_examination_staff(env, patient, app, type='APP')
        log_data.add_examination_staff_count(env, patient, shift, shift_count, type='APP')
        yield env.process(ed.examination(patient))
        print(
            f"{date_time.get_datetime(env.now)} Patient {patient} finishes examination by APP {app.shift, app.number}")
        yield ed.app.put(app)
        print(f"{date_time.get_datetime(env.now)} APP {app.shift, app.number} is now available")
        end = env.now
        log_data.add_end_examination(env, patient)
        print(
            f"{date_time.get_datetime(env.now)} Patient {patient} finished examination by APP {app.shift, app.number} in {date_time.get_timedelta(start, end, units='min')} min")

        # tests and Scans
        yield env.timeout(abs(np.random.normal(30 * 60, 5 * 60)))

        # treatment
        # app
        # TODO: Same APP who examined the patient
        day, hour = date_time.get_day_hour(env.now)
        shift = ed.app_factory.get_shift(day, hour)
        shift_count = ed.app_factory.get_shift_count(shift)
        app = yield ed.app.get(lambda app: (app.shift == shift) & (app.number <= shift_count))
        print(f"{date_time.get_datetime(env.now)} APP {app.shift, app.number} is ready")
        start = env.now
        print(f"{date_time.get_datetime(env.now)} Patient {patient} begins treatment by APP {app.shift, app.number}")
        log_data.add_begin_treatment(env, patient)
        log_data.add_treatment_staff(env, patient, app, type='APP')
        log_data.add_treatment_staff_count(env, patient, shift, shift_count, type='APP')
        yield env.process(ed.treatment(patient))
        print(f"{date_time.get_datetime(env.now)} Patient {patient} finishes treatment by APP {app.shift, app.number}")
        yield ed.app.put(app)
        print(f"{date_time.get_datetime(env.now)} APP {app.shift, app.number} is now available")
        end = env.now
        log_data.add_end_treatment(env, patient)
        print(
            f"{date_time.get_datetime(env.now)} Patient {patient} finished treatment by APP {app.shift, app.number} in {date_time.get_timedelta(start, end, units='min')} min")
