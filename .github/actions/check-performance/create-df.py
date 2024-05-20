import sys
import os
import pandas as pd

############### Define input variables ##########################
FRAMEWORK = sys.argv[1] 
WORKLOAD = sys.argv[2]
SCRIPT = sys.argv[3]
SCRIPT_NAME = SCRIPT[:-3]
PRECISION = sys.argv[4]
BENCHMARK_TYPE = sys.argv[5]
TOLERANCE = sys.argv[6]
THRESHOLD = sys.argv[7]
RESULT = sys.argv[8]
PASS = sys.argv[9]
STEP_URL = sys.argv[10]
OUTPUT_FOLDER = sys.argv[11]

############### Add parameters to table ##########################
if os.path.isfile(f'{OUTPUT_FOLDER}/performances/result-table_{WORKLOAD}_{SCRIPT_NAME}.csv') == False:
    result_df = pd.DataFrame(columns=['Framework', 'Workload', 'Script', 'Precision', 'Benchmark Type', 'Tolerance', 'Threshold', 'Result', 'Pass/Fail', 'URL'])
    result_df.loc[len(result_df.index)] = [FRAMEWORK, WORKLOAD, SCRIPT, PRECISION, BENCHMARK_TYPE, TOLERANCE, THRESHOLD, RESULT, PASS, STEP_URL]
    result_df.to_csv(f'{OUTPUT_FOLDER}/performances/result-table_{WORKLOAD}_{SCRIPT_NAME}.csv', index=False)
else:
    result_df = pd.read_csv(f'{OUTPUT_FOLDER}/performances/result-table_{WORKLOAD}_{SCRIPT_NAME}.csv')
    result_df.loc[len(result_df.index)] = [FRAMEWORK, WORKLOAD, SCRIPT, PRECISION, BENCHMARK_TYPE, TOLERANCE, THRESHOLD, RESULT, PASS, STEP_URL]
    result_df.to_csv(f'{OUTPUT_FOLDER}/performances/result-table_{WORKLOAD}_{SCRIPT_NAME}.csv', index=False)
