from fextractor import *


class PrepareParser(SimpleParser):
    fare_prediction_from: str = '2015-08-01 00:00:00'
    fare_prediction_to: str = '2015-08-15 00:00:00'
    fare_prediction_sample: int = 10000
    pass


def prepare_fare_prediction(args: PrepareParser):
    args.keycol = 'trip_id'
    args.target = 'fare_amount'
    fare_prediction_from = args.fare_prediction_from
    fare_prediction_to = args.fare_prediction_to
    num_samples = args.fare_prediction_sample
    from_date = fare_prediction_from.split(' ')[0]
    to_date = fare_prediction_to.split(' ')[0]
    args.task_dir += f'_{from_date}_{to_date}_{num_samples}'

    fcols = ['passenger_count', 'trip_distance', 'pickup_datetime',
             'pickup_longitude', 'pickup_latitude',
             'dropoff_longitude', 'dropoff_latitude',
             'pickup_ntaname', 'dropoff_ntaname']

    dbconn = DBConnector()
    # if target is nan or null, drop it
    sql = """
    select {keycol},
    {fcols},
    {target}
    from trips
    where pickup_datetime >= '{fare_prediction_from}'
        and pickup_datetime < '{fare_prediction_to}'
        and {target} is not null
    order by pickup_datetime, {keycol}
    """.format(keycol=args.keycol, target=args.target,
               fcols=",".join(fcols),
               fare_prediction_from=fare_prediction_from, fare_prediction_to=fare_prediction_to)
    df = dbconn.execute(sql)
    df = df.sample(num_samples, random_state=0)

    reqs = df[[args.keycol] + fcols]
    labels = df[[args.keycol, args.target]]

    save_features(reqs, args.task_dir, 'requests.csv')
    save_features(labels, args.task_dir, 'labels.csv')


def prepare_dayofweek(args: PrepareParser):
    args.keycol = 'day_of_year'
    args.target = 'day_of_week'
    dbconn = DBConnector()
    # note that we want our label start from 0. so we need to minus 1
    sql = """
    select DISTINCT toDayOfYear(pickup_datetime) as {keycol}, (toDayOfWeek(pickup_datetime) - 1) as {target}
    from trips
    where {target} is not null
    order by {keycol}
    """.format(keycol=args.keycol, target=args.target)
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
    select DISTINCT toDayOfYear(pickup_datetime) as {keycol}, 
    case
        when toDayOfWeek(pickup_datetime) in (6, 7) then 1
        else 0
    end as {target}
    from trips
    where {target} is not null
    order by {keycol}
    """.format(keycol=args.keycol, target=args.target)
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
    select DISTINCT (toDayOfYear(pickup_datetime) * 24 + toHour(pickup_datetime)) as {keycol}, toHour(pickup_datetime) as {target} 
    from trips
    where {target} is not null
    order by {keycol}
    """.format(keycol=args.keycol, target=args.target)
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
    select DISTINCT (toDayOfYear(pickup_datetime) * 24 + toHour(pickup_datetime)) as {keycol}, 
    case
        when toHour(pickup_datetime) >= 22 or toHour(pickup_datetime) < 6 then 1
        else 0
    end as {target}
    from trips
    where {target} is not null
    order by {keycol}
    """.format(keycol=args.keycol, target=args.target)
    df = dbconn.execute(sql)

    reqs = df[[args.keycol]]
    labels = df[[args.keycol, args.target]]

    save_features(reqs, args.task_dir, 'requests.csv')
    save_features(labels, args.task_dir, 'labels.csv')


if __name__ == "__main__":
    args = PrepareParser().parse_args()
    if args.task == 'fare_prediction':
        prepare_fare_prediction(args)
    elif args.task == 'day_of_week':
        prepare_dayofweek(args)
    elif args.task == 'is_weekend':
        prepare_is_weekend(args)
    elif args.task == 'hour_of_day':
        prepare_hourofday(args)
    elif args.task == 'is_night':
        prepare_is_night(args)
    else:
        print('Unknown task')
