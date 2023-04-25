from fextractor import extract
from pipeline import *


def allocate_qsamples(avg_sample: float, qimps: list[float]):
    # allocate sampling rate according to query importance
    # count number of non-zero element in qimps
    sum_sample = avg_sample * len([qimp for qimp in qimps if qimp > 0])
    qsamples = sum_sample * np.array(qimps) / np.sum(qimps)
    qsamples = [np.round(qsample * 100) / 100 for qsample in qsamples]
    return qsamples


def get_avg_sample(sample: Union[float, str]) -> float:
    if isinstance(sample, float):
        avg_sample = sample
    else:
        avg_sample = float(sample.split(':')[1])
    return avg_sample


def load_apx_features(args: SimpleParser, fcols: list[str]) -> pd.DataFrame:
    """ load approximate features from csv files, extract if not exists
    1 compute query importance = sum(importance of valid agg features from this query)
    2 allocate sampling rate according to query importance
    3 transform the original query templates to approximate query templates
    4 execute the queries to extract features, and save it to csv
    5 load the features from csv and process nan values
    6 return apx features
    """
    feature_dir = args.feature_dir
    filename = f'{args.ffile_prefix}.csv'
    fpath = os.path.join(feature_dir, filename)
    if not os.path.exists(fpath):
        # extract the required features
        print(f'{fpath} does not exist, extract them')

        sql_templates = args.sql_templates
        # rewrite the sql template, such only valid feas (fname in qcols) will be returned
        qcols = compute_valid_qcols(sql_templates, fcols)
        sql_templates = [aggfname_rewrite(template, qcols[i])
                         for i, template in enumerate(sql_templates)]

        # compute query sample rate
        if isinstance(args.sample, float):
            qsamples = [args.sample] * len(sql_templates)
        else:
            avg_sample = get_avg_sample(args.sample)
            fimportance = load_from_csv(
                args.pipelines_dir, 'feature_importance.csv')
            fimportance = fimportance[fimportance['fname'].isin(fcols)]
            fcols = fimportance['fname'].values.tolist()
            fimps = fimportance['importance'].values.tolist()

            qagg_imps = compuet_query_agg_importance(
                sql_templates, fcols, fimps)
            print(
                f'qimps={compuet_query_importance(sql_templates, fcols, fimps)}')
            print(f'qagg_imps={qagg_imps}')
            qsamples = allocate_qsamples(avg_sample, qagg_imps)
        print(f'qsamples={qsamples}')

        # rewrite the sql template to be approximate templates
        sql_templates = [approximation_rewrite(
            template, qsamples[i]) for i, template in enumerate(sql_templates)]

        # extract features
        feature_dir = args.feature_dir
        extract(args.task_dir, feature_dir,
                args.ffile_prefix, args.keycol, sql_templates)

    print(f'load features from {fpath}')
    apx_features = load_from_csv(feature_dir, f'{args.ffile_prefix}.csv')
    apx_features = nan_processing(apx_features, dropna=False)
    apx_features = datetime_processing(apx_features, method='drop')

    return apx_features


def run(args: SimpleParser):
    """ run pipeline with automatically sampled features.
    We assume 
        exact pipeline is pre-built
        feature importances on exact pipeline are available
        test workload is available
        apx pipeline built with avg_sample is available as baseline
    If sampled features are not available, extract it directly.
    args.sample with be set to 'auto:{avg_sample}'
    Workflow:
    1. load pipeline
    2. load apx pipeline built with avg_sample
    3. load test_X, test_y, test_kids
    4. load apx_features
        4.1 compute query importance = sum(importance of agg features from this query)
        4.2 allocate sampling rate according to query importance
        4.3 transform the original query templates to approximate query templates
        4.4 execute the queries to extract features, and save it to csv
        4.5 load the features from csv and return apx features
    5. merge test_kids and test_apx_features
    6. evaluate the pipeline with apx features
    """

    # load pipelines
    pipe = load_pipeline(args.pipelines_dir, 'pipeline.pkl')
    apx_pipe = load_apx_pipeline(
        args.pipelines_dir, get_avg_sample(args.sample))

    # load test workload
    test_X = pd.read_csv(os.path.join(args.pipelines_dir, 'test_X.csv'))
    test_y = pd.read_csv(os.path.join(args.pipelines_dir, 'test_y.csv'))
    test_kids = pd.read_csv(os.path.join(args.pipelines_dir, 'test_kids.csv'))

    fnames = pipe.feature_names_in_.tolist()
    assert sorted(fnames) == sorted(apx_pipe.feature_names_in_.tolist())

    # load apx features
    apx_features = load_apx_features(args, fnames)

    # merge test_kids and test_apx_features
    apx_X = pd.merge(test_kids, apx_features, how='left', on=args.keycol)

    aggfnames, _ = feature_ctype_inference(fnames, args.keycol, args.target)
    exp_X = baseline_expected_default(test_X, test_X, aggfnames)

    evals = []
    test_pred = pipe.predict(test_X)
    evals.append(evaluate_pipeline(
        args, pipe, test_X, test_y, 'extP-extF-acc'))
    evals.append(evaluate_pipeline(
        args, pipe, exp_X, test_y, 'extP-bs0F-acc'))
    evals.append(evaluate_pipeline(
        args, pipe, exp_X, test_pred, 'extP-bs0F-sim'))
    evals.append(evaluate_pipeline(
        args, apx_pipe, apx_X, test_y, 'apxP-apxF-acc'))
    evals.append(evaluate_pipeline(
        args, apx_pipe, apx_X, test_pred, 'apxP-apxF-sim'))
    evals.append(evaluate_pipeline(
        args, apx_pipe, test_X, test_y, 'apxP-extF-acc'))
    evals.append(evaluate_pipeline(
        args, apx_pipe, test_X, test_pred, 'apxP-extF-sim'))
    evals.append(evaluate_pipeline(
        args, pipe, apx_X, test_y, 'extP-apxF-acc'))
    evals.append(evaluate_pipeline(
        args, pipe, apx_X, test_pred, 'extP-apxF-sim'))

    # show evals as pandas dataframe
    evals_df = pd.DataFrame(evals)
    print(evals_df)
    save_to_csv(evals_df, args.evals_dir, 'evals.csv')
    plot_hist_and_save(args, pipe.predict(apx_X), os.path.join(
        args.evals_dir, 'apx_y_test_pred.png'), 'apx_y_test_pred', 'y', 'count')
    plot_hist_and_save(args, pipe.predict(exp_X), os.path.join(
        args.evals_dir, 'bsl_y_test_pred.png'), 'bsl_y_test_pred', 'y', 'count')


if __name__ == "__main__":
    args = SimpleParser().parse_args()
    print(args)
    assert args.apx_training == False, 'apx_training must be False'
    run(args)
