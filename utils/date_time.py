import datetime


def get_datetime(secs, start_time=datetime.datetime(2022, 1, 1, 0, 0, 0)):
    if not isinstance(secs, (int, float)):
        try:
            secs = int(secs)
        except:
            secs = None
    if secs is not None:
        return start_time + datetime.timedelta(seconds=round(secs))


def get_timedelta(start, end, units='sec'):
    if not isinstance(start, (int, float)):
        try:
            start = int(start[0])
        except:
            return
    if not isinstance(end, (int, float)):
        try:
            end = int(end[0])
        except:
            return
    if units == 'min':
        div = 60
    elif units == 'hour':
        div = 3600
    else:
        div = 1
    return round((end - start) / div, 0)


def get_day_hour(secs):
    dtime = get_datetime(secs)
    day = dtime.weekday()
    hour = dtime.hour
    day_dict = {0: 'Sun', 1: 'Mon', 2: 'Tue', 3: 'Wed', 4: 'Thu', 5: 'Fri', 6: 'Sat'}
    day = day_dict.get(day)
    return day, hour


def get_day_hour_min(secs):
    dtime = get_datetime(secs)
    day = dtime.weekday()
    hour = dtime.hour
    day_dict = {0: 'Sun', 1: 'Mon', 2: 'Tue', 3: 'Wed', 4: 'Thu', 5: 'Fri', 6: 'Sat'}
    day = day_dict.get(day)
    minute = dtime.minute
    return day, hour, minute
