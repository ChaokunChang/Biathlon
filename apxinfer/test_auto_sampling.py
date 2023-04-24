from fextractor import extract
from pipeline import *


def load_apx_pipeline(pipelines_dir, sample):
    if sample is not None:
        apx_pipe_dir = os.path.join(
            pipelines_dir, f'sample_{sample}')
    else:
        apx_pipe_dir = pipelines_dir
    apx_pipe = load_pipeline(apx_pipe_dir, 'pipeline.pkl')
    return apx_pipe


def allocate_sampling_rate(args: SimpleParser):
    avg_sample: str = args.sample
    fcols: list = args.fcols
    agg_features, nonagg_features = feature_ctype_inference(
        fcols, args.keycol, args.target)
    sampling_imps = args.fimps
    # if the feature is not aggregation feature, set importance to 0
    for i in range(len(fcols)):
        if fcols[i] not in agg_features:
            sampling_imps[i] = 0
    assert np.sum(sampling_imps) > 0, 'no aggregation feature is selected'

    avg_sample = float(avg_sample[len('auto'):])
    sample_rates = avg_sample * np.array(sampling_imps) / np.sum(sampling_imps)
    # sample_rates must be in 0.01, 0.02, ... 0.09, 0.10, 0.11, ... 0.99, 1.00
    sample_rates = np.round(sample_rates * 100) / 100
    assert len(sample_rates) == len(fcols)
    return sample_rates


def aggfname_rewrite(template: str, qcols: list) -> str:
    """ rewrite the sql template, such only fname in qcols will be returned
    the sql template will be like select fop as fname, ... from table
    we only keep the col that is in qcols
    """
    if len(qcols) == 0:
        return ""

    assert 'select ' in template or 'SELECT ' in template
    template = template.replace('select ', 'SELECT ')
    template = template.replace('from ', 'FROM ')
    select_ = template.split('SELECT ')[1].split('FROM ')[0]
    if (len(template.split('SELECT ')[1].split('FROM ')) > 1):
        from_ = " FROM " + template.split('SELECT ')[1].split('FROM ')[1]
    else:
        from_ = ""
    # print(f'select_={select_}')
    # print(f'from_={from_}')
    fop_as_fnames = [fop_as_fname.strip()
                     for fop_as_fname in select_.split(',')]
    # print(f'fop_as_fnames={fop_as_fnames}')

    new_fop_as_fnames = []
    for i, fop_as_fname in enumerate(fop_as_fnames):
        fop, fname = fop_as_fname.split(' as ')
        fop, fname = fop.strip(), fname.strip()
        if fname in qcols:
            new_fop_as_fnames.append(fop_as_fname)

    new_select = 'SELECT ' + ', '.join(new_fop_as_fnames)
    new_template = new_select + from_

    return new_template.strip()


def load_auto_apx_features(args: SimpleParser) -> pd.DataFrame:
    """ load approximate features from csv files, extract if not exists
    1 compute query importance = sum(importance of agg features from this query)
    2 allocate sampling rate according to query importance
    3 transform the original query templates to approximate query templates
    4 execute the queries to extract features, and save it to csv
    5 load the features from csv and process nan values
    6 return apx features
    """
    avg_sample = float(args.sample[len('auto'):])
    fcols = args.fcols
    fimps = args.fimps
    sql_templates = args.templator.templates
    q2f_map, f2q_map = get_query_feature_map(sql_templates)
    qimps = np.zeros(len(sql_templates))
    qcols = [[] for _ in range(len(sql_templates))]
    for fcol, fimp in zip(fcols, fimps):
        qid = f2q_map[fcol]
        qcols[qid].append(fcol)
        if is_agg_feature(fcol):
            qimps[qid] += fimp

    # allocate sampling rate according to query importance
    # count number of non-zero element in qimps
    sum_sample = avg_sample * np.sum(qimps > 0)
    qsamples = sum_sample * np.array(qimps) / np.sum(qimps)
    # qsamples = [np.round(qsample * 1000) / 1000 for qsample in qsamples]
    print(f'qsamples={qsamples}')

    # rewrite the sql template to be approximate templates
    sql_templates = [approximation_rewrite(
        template, qsamples[i]) for i, template in enumerate(sql_templates)]

    # rewrite the sql template, such only fname in qcols will be returned
    sql_templates = [aggfname_rewrite(template, qcols[i])
                     for i, template in enumerate(sql_templates)]

    # extract features
    feature_dir = args.feature_dir
    extract(args.task_dir, feature_dir,
            args.ffile_prefix, args.keycol, sql_templates)
    apx_features = load_from_csv(feature_dir, f'{args.ffile_prefix}.csv')
    apx_features = nan_processing(apx_features, dropna=False)
    apx_features = datetime_processing(apx_features, method='drop')
    return apx_features


def run(args: SimpleParser):
    """ run pipeline with automatically sampled features.
    If sampled features are not available, extract it directly.
    args.sample with be set to 'auto{avg_sample}'
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
    avg_sample = float(args.sample[len('auto'):])
    apx_pipe = load_apx_pipeline(args.pipelines_dir, avg_sample)

    # load test data
    test_X = pd.read_csv(os.path.join(args.pipelines_dir, 'test_X.csv'))
    test_y = pd.read_csv(os.path.join(args.pipelines_dir, 'test_y.csv'))
    test_kids = pd.read_csv(os.path.join(args.pipelines_dir, 'test_kids.csv'))

    # load apx features
    apx_features = load_auto_apx_features(args)

    # merge test_kids and test_apx_features
    apx_X = pd.merge(test_kids, apx_features, on=args.keycol, how='left')

    typed_fnames = feature_type_inference(apx_X, args.keycol, args.target)
    exp_X = baseline_expected_default(
        test_X, test_X, typed_fnames['agg_features'])

    evals = []
    test_pred = pipe.predict(test_X)
    evals.append(evaluate_pipeline(args, pipe, test_X, test_y, 'ext'))
    evals.append(evaluate_pipeline(args, pipe, exp_X, test_y, 'b0a'))
    evals.append(evaluate_pipeline(args, pipe, exp_X, test_pred, 'b0s'))
    evals.append(evaluate_pipeline(args, apx_pipe, apx_X, test_y, 'b1a'))
    evals.append(evaluate_pipeline(args, apx_pipe, apx_X, test_pred, 'b1s'))
    evals.append(evaluate_pipeline(args, pipe, apx_X, test_y, 'apx'))
    evals.append(evaluate_pipeline(args, pipe, apx_X, test_pred, 'sim'))

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
    assert isinstance(args.sample, str) and args.sample.startswith('auto')
    run(args)
