from pipeline import *


if __name__ == "__main__":
    args = SimpleParser().parse_args()
    assert args.apx_training == False, 'apx_training must be False'

    # load pipelines
    pipe = load_pipeline(args.pipelines_dir, 'pipeline.pkl')
    apx_pipe = load_apx_pipeline(args.pipelines_dir, args.sample)

    # load test workload
    test_X = pd.read_csv(os.path.join(args.pipelines_dir, 'test_X.csv'))
    test_y = pd.read_csv(os.path.join(args.pipelines_dir, 'test_y.csv'))
    test_kids = pd.read_csv(os.path.join(args.pipelines_dir, 'test_kids.csv'))

    fnames = pipe.feature_names_in_.tolist()
    assert sorted(fnames) == sorted(apx_pipe.feature_names_in_.tolist())

    # load apx feature
    apx_features = load_features(
        args, dropna=False, kids=test_kids.values.flatten().tolist(), cols=fnames)

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
