from pipeline import *


def feature_selection_by_importance(pipelines_dir: str, sql_templates_file: str, topk: Union[float, int] = 10, must_include: list[str] = None):
    # load pipelines
    pipe = load_pipeline(pipelines_dir, 'pipeline.pkl')

    # load test workload
    test_X = pd.read_csv(os.path.join(pipelines_dir, 'test_X.csv'))
    test_y = pd.read_csv(os.path.join(pipelines_dir, 'test_y.csv'))
    test_kids = pd.read_csv(os.path.join(pipelines_dir, 'test_kids.csv'))

    # load feature names
    fnames = pipe.feature_names_in_.tolist()

    # load feature importances
    feature_importance = load_from_csv(pipelines_dir, 'feature_importance.csv')
    candidates_df = feature_importance[feature_importance['fname'].isin(fnames)][[
        'fname', 'importance']]

    # select topk features
    if topk < 1:
        topk = int(topk * len(fnames))
    topk = min(int(topk), len(fnames))
    candidates_df = candidates_df.sort_values(by='importance', ascending=False)
    candidates_df = candidates_df.iloc[:topk]
    fcols = candidates_df['fname'].values.tolist()

    if must_include is not None:
        for col in must_include:
            if col is not None:
                fcols.append(col)
    print(f'fcols={fcols}')
    # improve the sql template files according to fcols
    templates_dir = os.path.dirname(sql_templates_file)
    templates_filename = os.path.basename(sql_templates_file).split('.')[0]
    new_template_file = os.path.join(
        templates_dir, f'{templates_filename}_selected_top{topk}.sql')

    templator = SQLTemplates().from_file(sql_templates_file)
    # rewrite the sql template, such only valid feas (fname in qcols) will be returned
    sql_templates = [select_reduction_rewrite(template, fcols)
                     for i, template in enumerate(templator.templates)]
    # save the new template file
    with open(new_template_file, 'w') as f:
        f.write(';\n'.join(sql_templates))

    return new_template_file


if __name__ == "__main__":
    args = SimpleParser().parse_args()
    feature_selection_by_importance(args.pipelines_dir,
                                    args.sql_templates_file, args.topk_features,
                                    must_include=[args.sort_by])
