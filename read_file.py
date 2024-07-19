import json
import ast

filename1 = '2023_06_filtered_with_predictions_J48'
filename2 = '2023_06_filtered_with_predictions_LinearRegression'
filename3 = '2023_06_filtered_with_predictions_RandomForest'

output_file1 = '2023_06_J48.json'
output_file2 = '2023_06_LinearRegression.json'
output_file3 = '2023_06_RandomForest.json'
data = []
with open(filename3, 'r', encoding="ISO-8859-1") as f:
    for line in f:
        data.append(ast.literal_eval(line.strip()))


jobs = []
for item in data:
    data = {
        'id': item['jobid'],
        'submit': item['@submit'],
        'elapsed': item['elapsed'],
        'start': item['@start'],
        'end': item['@end'],
        'elapsed_predicted': item['elapsed_predicted'],

    }
    jobs.append(data)

json.dump(jobs, open(output_file3, 'w'))
