import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import argparse
def load_json(json_file):
    jobs = []
    if not json_file.exists():
        print(f"File {json_file} not found")
        raise FileNotFoundError
    with open(json_file, 'r') as file:
        items = json.load(file)

    for item in items:
        data = {
            'id': item['id'],
            'submit': item['submit'],
            'elapsed': item['elapsed'],
            'start': item['start'],
            'end': item['end'],
            'predicted_minimum_duration_in_seconds': item['predicted_minimum_duration_in_seconds'],
            'predicted_maximum_duration_in_seconds': item['predicted_maximum_duration_in_seconds']
        }
        jobs.append(data)

    return jobs

def get_jobs(df):

    init_date = datetime.strptime('2023-06-12T18:38:10', "%Y-%m-%dT%H:%M:%S")
    end_date = datetime.strptime('2023-06-12T21:32:18',"%Y-%m-%dT%H:%M:%S")
    #df['submit'] = pd.to_datetime(df['submit'], format='%Y-%m-%dT%H:%M:%S')
    df = df[ ( df['submit'] >= init_date ) & ( df['submit'] <= end_date ) ]
    return df

def simulate_execution_slurm(df, n_nodes):
    num_cores = n_nodes

    df['waiting_time'] = 0
    current_time = df['normalized_submit'].min()

    node_occupancy = [current_time] * num_cores

    for index, row in df.iterrows():
        # Encontra o nó (core) com o menor tempo de ocupação
        min_index = np.argmin(node_occupancy)
        node_start_time = node_occupancy[min_index]

        # Atualiza o tempo de início do job no nó
        if row['normalized_submit'] > node_start_time:
            start_time = row['normalized_submit']
        else:
            start_time = node_start_time


        df.loc[index, 'waiting_time'] = (start_time - row['normalized_submit']).seconds

        completion_time = start_time + pd.Timedelta(seconds=row['elapsed'])



        df.loc[index, 'completion_time'] = completion_time

        node_occupancy[min_index] = completion_time

    #Pega apenas os jobs que terminam até o ultimo dia do mes 08
    filter_date = datetime.strptime('2023-08-01T00:00:00', "%Y-%m-%dT%H:%M:%S")
    df_throughput = df[df['completion_time'] <= filter_date]

    major_time = df_throughput['completion_time'].max()

    #total_jobs = len(df)
    total_jobs = len(df_throughput)
    print(f'Number Nodes: {n_nodes}')
    print(f"Total jobs: {total_jobs}")
    print(f"First Job: {df['normalized_submit'].min()}")
    print(f"Last Job: {major_time}")

    makespan = (major_time - df['normalized_submit'].min())
    td = pd.Timedelta(makespan)
    makespan_sec = td.total_seconds()
    makespan = td.total_seconds() / 60
    throughput =  total_jobs / (makespan_sec / 86400)

    return df, throughput, makespan







if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Benchmark AWS')
    parser.add_argument('window_size', type=str, help='Size of window in minutes')
    parser.add_argument('n_nodes', type=int, help='Number of nodes from our cluster')

    args = parser.parse_args()

    file_path = Path('2023_06_fwp_filtered.json')
    df = load_json(file_path)
    df = pd.DataFrame(df)

    df['submit'] = pd.to_datetime(df['submit'])
    df['start'] = pd.to_datetime(df['start'])
    df['end'] = pd.to_datetime(df['end'])

    df = get_jobs(df)

    wsize = args.window_size + 'min'
    n_nodes = args.n_nodes
    df['normalized_submit'] = df['submit'].dt.floor(wsize)
    df = df.sort_values(by=['normalized_submit', 'predicted_minimum_duration_in_seconds'])
    #print(df)
    simulated_df, throughput, makespan = simulate_execution_slurm(df,n_nodes)

    waiting_time_mean = simulated_df['waiting_time'].mean()
    print(f"Throughput: {throughput:.2f} jobs per day")
    print(f"Makespan: {makespan:.2f} minutes")
    print(f"Waiting Time Médio: {waiting_time_mean/60:.2f} minutes")
    #print("Simulated DataFrame:")
    print(simulated_df.head(20))