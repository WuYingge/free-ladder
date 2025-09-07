import datetime
from interval_utils import intervals

def after_close():
    set_date = datetime.datetime.now()
    if set_date.hour > 15:
        run_time_date = set_date.date() + datetime.timedelta(days=1)
    else:
        run_time_date = set_date.date()
    run_time = datetime.datetime(
        year=run_time_date.year, 
        month=run_time_date.month, 
        day=run_time_date.day, 
        hour=15, 
        minute=30
    )
    print(f"run time is {run_time}")
    def condition():
        now = datetime.datetime.now()
        return now > run_time
    return condition

def when_to_run(when, fun, /, *args, **kwargs):
    while not when():
        intervals(600)
    print(f"start to run in {datetime.datetime.now()}")
    fun(*args, **kwargs)
