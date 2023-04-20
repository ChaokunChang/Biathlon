from pipeline import *

if __name__ == "__main__":
    args = SimpleParser().parse_args()
    assert args.sample >= 0 and args.sample <= 1, 'sample rate must be in [0, 1]'

    pipe = load_pipeline(args.pipeline_fpath)
    # print(pipe)

    test_X = pd.read_csv(os.path.join(args.outdir_base, 'test_X.csv'))
    test_y = pd.read_csv(os.path.join(args.outdir_base, 'test_y.csv'))
    test_kids = pd.read_csv(os.path.join(
        args.outdir_base, 'test_kids.csv')).values.flatten().tolist()

    apx_features = load_features(args, sort_by=args.sort_by, kids=test_kids)
    apx_features = nan_processing(apx_features, dropna=False)
    typed_fnames = feature_type_inference(
        apx_features, args.keycol, target=args.target)
    labels = load_labels(args, apx_features[args.keycol].values.tolist())
    apx_Xy = pd.merge(apx_features, labels, on=args.keycol).sort_values(
        by=args.sort_by).drop(columns=typed_fnames['dt_features'])
    apx_Xy = apx_value_estimation(
        apx_Xy, typed_fnames['agg_features'], args.sample)
    apx_X_test, apx_y_test = apx_Xy.drop(
        columns=[args.target]), apx_Xy[args.target]
    exp_X_test = baseline_expected_default(
        test_X, test_X, typed_fnames['agg_features'])

    evals = []
    evals.append(evaluate_pipeline(args, pipe, test_X, test_y, 'ext'))
    evals.append(evaluate_pipeline(args, pipe, exp_X_test, apx_y_test, 'bsl'))
    evals.append(evaluate_pipeline(
        args, pipe, exp_X_test, pipe.predict(test_X), 'bsm'))
    evals.append(evaluate_pipeline(args, pipe, apx_X_test, apx_y_test, 'apx'))
    evals.append(evaluate_pipeline(
        args, pipe, apx_X_test, pipe.predict(test_X), 'sim'))

    # show evals as pandas dataframe
    evals_df = pd.DataFrame(
        evals, columns=['tag', 'mse', 'mae', 'r2', 'expv', 'maxe'])
    print(evals_df)

    plot_hist_and_save(args, pipe.predict(apx_X_test),
                       'apx_y_test_pred.png', 'apx_y_test_pred', 'y', 'count')
    plot_hist_and_save(args, pipe.predict(exp_X_test),
                       'bsl_y_test_pred.png', 'bsl_y_test_pred', 'y', 'count')
