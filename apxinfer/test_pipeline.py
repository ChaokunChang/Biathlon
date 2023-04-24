from pipeline import *


def load_apx_pipeline(args: SimpleParser):
    if args.sample is not None:
        apx_pipe_dir = os.path.join(
            args.pipelines_dir, f'sample_{args.sample}')
    else:
        apx_pipe_dir = args.pipelines_dir
    apx_pipe = load_pipeline(apx_pipe_dir, 'pipeline.pkl')
    return apx_pipe


if __name__ == "__main__":
    args = SimpleParser().parse_args()
    # assert args.sample >= 0 and args.sample <= 1, 'sample rate must be in [0, 1]'
    assert args.apx_training == False, 'apx_training must be False'

    pipe = load_pipeline(args.pipelines_dir, 'pipeline.pkl')
    apx_pipe = load_apx_pipeline(args)

    test_X = pd.read_csv(os.path.join(args.pipelines_dir, 'test_X.csv'))
    test_y = pd.read_csv(os.path.join(args.pipelines_dir, 'test_y.csv'))
    test_kids = pd.read_csv(os.path.join(args.pipelines_dir, 'test_kids.csv'))

    apx_features = load_features(
        args, dropna=False, kids=test_kids.values.flatten().tolist(), cols=args.fcols)
    typed_fnames = feature_type_inference(
        apx_features, args.keycol, target=args.target)

    apx_X = pd.merge(test_kids, apx_features, how='left', on=args.keycol).drop(
        columns=typed_fnames['dt_features'])
    apx_X = apx_value_estimation(
        apx_X, typed_fnames['agg_features'], args.sample)
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
