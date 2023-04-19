from fextractor import *


class PrepareParser(SimpleParser):
    pass


def prepare_dayofweek(args: PrepareParser):
    args.keycol = 'day_of_year'
    args.target = 'day_of_week'
    dbconn = DBConnector()
    sql = """
    select DISTINCT toDayOfYear(pickup_datetime) as day_of_year, toDayOfWeek(pickup_datetime) as day_of_week
    from trips
    order by day_of_year
    """
    df = dbconn.execute(sql)

    reqs = df[[args.keycol]]
    labels = df[[args.keycol, args.target]]

    save_features(reqs, args.task_dir, 'requests.csv')
    save_features(labels, args.task_dir, 'labels.csv')


def prepare_is_weekend(args: PrepareParser):
    args.keycol = 'day_of_year'
    args.target = 'is_weekend'

    dbconn = DBConnector()
    sql = """
    select DISTINCT toDayOfYear(pickup_datetime) as day_of_year, 
    case
        when toDayOfWeek(pickup_datetime) in (0, 6) then 1
        else 0
    end as is_weekend
    from trips
    order by day_of_year
    """
    df = dbconn.execute(sql)

    reqs = df[[args.keycol]]
    labels = df[[args.keycol, args.target]]

    save_features(reqs, args.task_dir, 'requests.csv')
    save_features(labels, args.task_dir, 'labels.csv')


def prepare_hourofday(args: PrepareParser):
    args.keycol = 'hourstamp'
    args.target = 'hour_of_day'

    dbconn = DBConnector()
    sql = """
    select DISTINCT (toDayOfYear(pickup_datetime) * 24 + toHour(pickup_datetime)) as hourstamp, toHour(pickup_datetime) as hour_of_day 
    from trips
    order by (toDayOfYear(pickup_datetime) * 24 + toHour(pickup_datetime))
    """
    df = dbconn.execute(sql)

    reqs = df[[args.keycol]]
    labels = df[[args.keycol, args.target]]

    save_features(reqs, args.task_dir, 'requests.csv')
    save_features(labels, args.task_dir, 'labels.csv')


def prepare_is_night(args: PrepareParser):
    args.keycol = 'hourstamp'
    args.target = 'is_night'

    dbconn = DBConnector()
    sql = """
    select DISTINCT (toDayOfYear(pickup_datetime) * 24 + toHour(pickup_datetime)) as hourstamp, 
    case
        when toHour(pickup_datetime) >= 22 or toHour(pickup_datetime) < 6 then 1
        else 0
    end as is_night
    from trips
    order by (toDayOfYear(pickup_datetime) * 24 + toHour(pickup_datetime))
    """
    df = dbconn.execute(sql)

    reqs = df[[args.keycol]]
    labels = df[[args.keycol, args.target]]

    save_features(reqs, args.task_dir, 'requests.csv')
    save_features(labels, args.task_dir, 'labels.csv')


if __name__ == "__main__":
    args = PrepareParser().parse_args()
    if args.task == 'day_of_week':
        prepare_dayofweek(args)
    elif args.task == 'is_weekend':
        prepare_is_weekend(args)
    elif args.task == 'hour_of_day':
        prepare_hourofday(args)
    elif args.task == 'is_night':
        prepare_is_night(args)
    else:
        print('Unknown task')
