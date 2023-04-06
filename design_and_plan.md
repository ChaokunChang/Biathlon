
# TODO
1. Design and Implement DBEngine.
2. Intergrate ClickHouse as Backend.
3. Load trips data into ClickHouse.

# Workflow
1. Load or Generate Data. Save to Database or CSV
2. Load or Generate Query. Save to Text or Json
3. Load and Run Query. Save results to Text or Json, including query, results, time
   1. Note that not all features will be used in model building.
4. Compose workload. Composed by (Data, Request, Queries, Features, Label)
5. Split Workload into Train, Val, and Test. Key by Data, split by Request.
6. Train a machine learning model for online prediction with TrainSet's (Features, Label)
7. Evaluate model using validation set, and config the pipeline with validation set
8. Test Pipeline with test set.
   1. During Test, we should run related queries again. (features used in model is subset of all features)


# Integration
## Clickhouse
```
[WITH expr_list|(subquery)]
SELECT [DISTINCT [ON (column1, column2, ...)]] expr_list
[FROM [db.]table | (subquery) | table_function] [FINAL]
[SAMPLE sample_coeff]
[ARRAY JOIN ...]
[GLOBAL] [ANY|ALL|ASOF] [INNER|LEFT|RIGHT|FULL|CROSS] [OUTER|SEMI|ANTI] JOIN (subquery)|table (ON <expr_list>)|(USING <column_list>)
[PREWHERE expr]
[WHERE expr]
[GROUP BY expr_list] [WITH ROLLUP|WITH CUBE] [WITH TOTALS]
[HAVING expr]
[ORDER BY expr_list] [WITH FILL] [FROM expr] [TO expr] [STEP expr] [INTERPOLATE [(expr_list)]]
[LIMIT [offset_value, ]n BY columns]
[LIMIT [n, ]m] [WITH TIES]
[SETTINGS ...]
[UNION  ...]
[INTO OUTFILE filename [COMPRESSION type [LEVEL level]] ]
[FORMAT format]
```