import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

def load_json(json_file):
    jobs = []
    if not json_file.exists():
        print(f"File {json_file} not found")
        raise FileNotFoundError
    with open(json_file, 'r') as file:
        items = json.load(file)

    if 'J48' in json_file.name:
        camp_name = 'predicted_minimum_duration_in_seconds'
    else:
        camp_name = 'elapsed_predicted'

    for item in items:
        data = {
            'id': item['id'],
            'submit': item['submit'],
            'elapsed': item['elapsed'],
            'start': item['start'],
            'end': item['end'],
            'predicted_minimum_duration_in_seconds': item[camp_name]
        }
        jobs.append(data)

    return jobs

def get_jobs(df, start_time, end_time):
    init_date = datetime.strptime(start_time, "%Y-%m-%dT%H:%M:%S")
    end_date = datetime.strptime(end_time, "%Y-%m-%dT%H:%M:%S")

    df = df[(df['submit'] >= init_date) & (df['submit'] <= end_date)]
    return df

def reorder_queue(df, scheduling_policy):
    if scheduling_policy == 'FCFS':
        return df.sort_values(by=['normalized_submit', 'submit'])
    elif scheduling_policy == 'SJF':
        return df.sort_values(by=['normalized_submit', 'elapsed'])
    else:
        raise ValueError(f"Unknown scheduling policy: {scheduling_policy}")

def simulate_execution_slurm(df, n_nodes):
    num_cores = n_nodes

    df['waiting_time'] = 0
    df['start_time'] = pd.NaT
    df['completion_time'] = pd.NaT

    current_time = df['normalized_submit'].min()
    node_occupancy = [current_time] * num_cores

    for index, row in df.iterrows():
        min_index = np.argmin(node_occupancy)
        node_start_time = node_occupancy[min_index]

        if row['normalized_submit'] >= node_start_time:
            start_time = row['normalized_submit']
        else:
            start_time = node_start_time

        df.loc[index, 'waiting_time'] = (start_time - row['normalized_submit']).seconds
        df.loc[index, 'start_time'] = start_time

        completion_time = start_time + timedelta(seconds=row['elapsed'])
        df.loc[index, 'completion_time'] = completion_time

        node_occupancy[min_index] = completion_time

    #df_completion_jobs_ago = df.copy()

    filter_date = datetime.strptime('2023-09-01T00:00:00', "%Y-%m-%dT%H:%M:%S")
    df_completion_jobs_ago = df[df['completion_time'] <= filter_date]

    major_time = df_completion_jobs_ago['completion_time'].max()
    total_jobs = len(df_completion_jobs_ago)

    print(f'Number Nodes: {n_nodes}')
    print(f"Total jobs: {total_jobs}")
    print(f"First Job - submitted time: {df['submit'].min()}")
    print(f"Last Job - completed time: {major_time}")

    makespan = (major_time - df_completion_jobs_ago['submit'].min())
    td = pd.Timedelta(makespan)
    makespan_sec = td.total_seconds()
    makespan_min = td.total_seconds() / 60
    waiting_time_mean = df_completion_jobs_ago['waiting_time'].mean()
    throughput = total_jobs / (makespan_sec / (60 * 60 * 24 ))

    df_output = df[['id', 'submit','elapsed','start_time', 'waiting_time', 'completion_time']]
    return df_output, waiting_time_mean, throughput, makespan_min

def get_model(model_name):
    if model_name == 'J48':
        return '2023_06_J48'
    if model_name == 'LR' or 'LinearRegression':
        return '2023_06_LinearRegression'
    if model_name == 'RF' or 'RandomForest':
        return '2023_06_RandomForest'
    return 'Error'

def graph(start_date,end_date, df1, df2 ):

    start_time = pd.to_datetime('2023-07-12 00:00:00')
    end_time = pd.Timestamp('2023-08-12 00:00:00')

    df1_filtered = df1[(df1['completion_time'] >= start_time) & (df1['completion_time'] <= end_time)]
    df2_filtered = df2[(df2['completion_time'] >= start_time) & (df2['completion_time'] <= end_time)]

    # Criar intervalos de 30 minutos
    time_intervals = pd.date_range(start=start_time, end=end_time, freq='D')

    # Contar o número de jobs concluídos em cada intervalo de 30 minutos para ambos os DataFrames
    counts_df1 = []
    counts_df2 = []
    interval_labels = []

    for i in range(len(time_intervals) - 1):
        interval_start = time_intervals[i]
        interval_end = time_intervals[i + 1]
        count_df1 = df1_filtered[(df1_filtered['completion_time'] >= interval_start) & (
                    df1_filtered['completion_time'] < interval_end)].shape[0]
        count_df2 = df2_filtered[(df2_filtered['completion_time'] >= interval_start) & (
                    df2_filtered['completion_time'] < interval_end)].shape[0]
        if count_df1 > 0 or count_df2 > 0:
            counts_df1.append(count_df1)
            counts_df2.append(count_df2)
            interval_labels.append(interval_start.strftime("%m-%d"))

    # Calcular as médias

    mean_df1 = sum(counts_df1) / len(counts_df1)
    mean_df2 = sum(counts_df2) / len(counts_df2)
    print(mean_df1)
    print(mean_df2)
    # print("Media sorted:")
    # print(mean_df1)
    # df1_filtered['Waiting Time'] = df1_filtered['Start'] - df1_filtered['Submit']
    # print(df1_filtered['Waiting Time'].mean())
    # print("Media unsorted")
    # print(mean_df2)
    # print(df2_filtered['queue_time'].mean())
    # Configurar o gráfico de barras
    bar_width = 0.2
    x = range(len(counts_df1))

    fig, ax = plt.subplots(figsize=(18, 8), dpi=150)

    # Barras para CSV 1
    bars1 = ax.bar([pos - bar_width / 2 for pos in x], counts_df1, width=bar_width, label='SJF', color='blue')

    # Barras para CSV 2
    bars2 = ax.bar([pos + bar_width / 2 for pos in x], counts_df2, width=bar_width, label='FCFS', color='green')

    # Adicionar linhas de média
    ax.axhline(mean_df1, color='blue', linestyle='--', linewidth=1.5, label=f'Average completed jobs SJF ')
    ax.axhline(mean_df2, color='red', linestyle='--', linewidth=1.5, label=f'Average completed jobs FCFS')

    # Configurar rótulos dos eixos e título
    ax.set_xlabel('Days')
    ax.set_ylabel('Completed Jobs')
    ax.set_title('Jobs Completed Per Day')

    # Definir rótulos do eixo X em intervalos maiores (a cada 2 horas)
    step = 2  # Isso significa que mostraremos a cada 2 horas (30 min * 4 = 120 min)
    filtered_interval_labels = [label if i % step == 0 else '' for i, label in enumerate(interval_labels)]
    ax.set_xticks(x)
    ax.set_xticklabels(filtered_interval_labels, rotation=45, ha='right')

    ax.legend()

    # Ajustar a visualização do gráfico
    # plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SLURM Job Scheduler Simulation')
    parser.add_argument('--start_time', type=str, default='2023-06-12T18:38:10', help='Start time for job filtering')
    parser.add_argument('--end_time', type=str, default='2023-06-12T21:32:18', help='End time for job filtering')
    parser.add_argument('--nodes', type=int, default=100, help='Number of nodes (cores)')
    parser.add_argument('--window', type=int, default=30, help='Time window size in minutes for job normalization')
    parser.add_argument('--model', type=str, default='J48', help='Choice a type of model to use')
    args = parser.parse_args()

    model_name = args.model
    if model_name == 'Error':
        print("Model Not Found. Exiting...")
        exit

    file = get_model(args.model)

    print(f'Using file: {file}')
    file_path = Path(f'{file}.json')
    df = load_json(file_path)
    df = pd.DataFrame(df)
    df['submit'] = pd.to_datetime(df['submit'])
    df['start'] = pd.to_datetime(df['start'])
    df['end'] = pd.to_datetime(df['end'])

    df = get_jobs(df, args.start_time, args.end_time)

    scheduling_policies = ['FCFS', 'SJF']
    wsize = str(args.window) + 'min'

    for policy in scheduling_policies:
        df_temp = df.copy()
        df_temp['normalized_submit'] = df_temp['submit'].dt.floor(wsize)
        print("Window size: ", wsize)
        df_temp = reorder_queue(df_temp, policy)

        simulated_df, waiting_time_mean, throughput, makespan = simulate_execution_slurm(df_temp, args.nodes)

        print(f"Scheduling Policy: {policy}")
        print(f"Throughput: {throughput:.2f} jobs per day")
        print(f"Makespan: {makespan:.2f} minutes")
        print(f"Average Waiting Time: {waiting_time_mean / 60:.2f} minutes")
        print(f"{10 * '-'}")
        if policy == 'FCFS':
            df_fcfs = simulated_df.copy()

        #simulated_df.to_csv(f"output/{file}_{policy}.csv", index=False)
        start_date = '2023-07-12 00:00:00'
        end_date = '2023-08-12 00:00:00'

    graph(start_date, end_date, simulated_df,df_fcfs)
