import pandas as pd
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from tqdm import tqdm
from sequential.seq2pat import Seq2Pat, Attribute
import pm4py
import matplotlib.pyplot as plt

# converters
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.obj import EventLog, Trace
from datetime import datetime, timedelta

# process mining
from pm4py.algo.discovery.alpha import algorithm as alpha_miner
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.algo.filtering.dfg.dfg_filtering import clean_dfg_based_on_noise_thresh

# social network analysis
from pm4py.algo.organizational_mining.sna import algorithm as sna_algorithm
from pm4py.visualization.sna import visualizer as pn_vis

# visualization
# (wvw: updated, courtesy https://stackoverflow.com/questions/75424412/no-module-named-pm4py-objects-petri-in-pm4py)
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.discovery.alpha import algorithm as alpha_miner
from pm4py.visualization.petri_net import visualizer as pn_visualizer
# (wvw: added)
from pm4py.visualization.dfg import visualizer as dfg_visualizer
from pm4py.visualization.heuristics_net import visualizer as hn_visualizer
from pm4py.visualization.process_tree import visualizer as pt_visualizer

# misc
from pm4py.objects.conversion.process_tree import converter as pt_converter
from pm4py.visualization.petri_net.util import performance_map

from collections import Counter
from collections_extended import frozenbag
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.obj import EventLog


# from pm4py.algo.filtering.log.variants.variants_filter import filter_log_variants_percentage
# from pm4py.objects.conversion.log.variants import to_data_frame

def read_data():
    df2 = pd.read_pickle(
        'decision_paths/subset_optimized_simple_size_0.2_fitnessweights_p0.34_f0.33_c0.33_weightmodel_weight_positive_simplified.pickle')
    return df2

def eventlog(df2):
    # Initialize an empty list to hold event log rows
    event_log0 = []
    event_log1 = []

    # Iterate over each decision path in the dataset
    timestamp = pd.Timestamp('2024-11-14')
    case_id = -1
    for i in tqdm(range(len(df2))):
        data = df2['rule_to_simplified_rules'].iloc[i]
        for path in data:
            case_id += 1
            for t, step in enumerate(path):
                # Create an event dictionary for the current step
                event = {
                    'case_id': case_id,
                    'activity': step.get('feature_abbrv'),
                    'timestamp': timestamp + pd.Timedelta(days=t)
                }
                # if event['activity'] in to_remove:
                #     # print(f" I removed {event['activity']} from the log")
                #     continue
                if pd.isna(event['activity']):
                    event['activity'] = f'predict{step['predict']}'
                # Append the event to the event log
                if path[-1]['predict'] == 1:
                    event_log1.append(event)
                else:
                    event_log0.append(event)

    event_log1 = pd.DataFrame(event_log1)
    event_log0 = pd.DataFrame(event_log0)
    event_log0.rename(columns={'timestamp': 'time:timestamp', 'case_id': 'case:concept:name', 'activity': 'concept:name'}, inplace=True)
    event_log1.rename(columns={'timestamp': 'time:timestamp', 'case_id': 'case:concept:name', 'activity': 'concept:name'}, inplace=True)
    return event_log0, event_log1

def processmining1(log0):
    # mine process from log
    dfg = dfg_discovery.apply(log0, variant=dfg_discovery.Variants.FREQUENCY)
    gviz = dfg_visualizer.apply(dfg, log=log0, variant=dfg_visualizer.Variants.FREQUENCY)
    dfg_visualizer.view(gviz)
    plt.show()

def processmining2(log0):
    heu_net = heuristics_miner.apply_heu(log0)
    gviz = hn_visualizer.apply(heu_net)
    hn_visualizer.view(gviz)
    plt.show()

    variants.columns = ['sequence', 'cov_amt']

    variants_sorted = variants.sort_values(by='cov_amt', ascending=False)
    variants_sorted = variants_sorted.reset_index(drop=True)
    variants_sorted['cov_perc'] = variants_sorted.apply(lambda row: 100 / num_seq * row['cov_amt'], axis=1)

    cumul_cov_perc = []
    cur_cov_perc = 0
    for _, row in variants_sorted.iterrows():  # yeah, yeah ...
        cur_cov_perc += row['cov_perc']
        cumul_cov_perc.append(cur_cov_perc)
    variants_sorted['cov_perc_cumul'] = cumul_cov_perc

    var_perc = 100 / num_var
    variants_sorted['var_perc_cumul'] = variants_sorted.apply(lambda row: (row.name + 1) * var_perc, axis=1)

    return variants_sorted


def print_variants_stats(vars_stats):
    for idx, row in vars_stats.iterrows():
        print(f"coverage: amt = {row['cov_amt']}, perc = {row['cov_perc']}, cumul = {row['cov_perc_cumul']}")
        print(f"variant: cnt = {row.name}, perc = {row['var_perc_cumul']}")
        print(row['sequence'])


# how many cases (percentage) do var_perc of variants (sorted desc by coverage) cover?
def get_case_coverage(var_perc, vars_stats):
    return get_x_coverage(var_perc, 'var_perc_cumul', 'cov_perc_cumul', vars_stats)


# how many variants (percentage) are needed (sorted desc by coverage) to cover case_perc of cases?
def get_variant_coverage(case_perc, vars_stats):
    return get_x_coverage(case_perc, 'cov_perc_cumul', 'var_perc_cumul', vars_stats)


# get all variants that are needed (sorted desc by coverage) to cover case_perc of cases
def get_covering_variants(case_perc, vars_stats):
    first_k = vars_stats.loc[vars_stats['cov_perc_cumul'] <= case_perc]
    return first_k


def get_x_coverage(perc, cmp_col, ret_col, vars_stats):
    first_k = vars_stats.loc[vars_stats[cmp_col] <= perc]
    return first_k.iloc[-1][ret_col]


def filter_traces_on_variants(log, variants):
    traces = log.groupby('case:concept:name')['concept:name'].apply(tuple).rename('sequence').reset_index()
    # add case's sequence to all events of that case
    merged_log = log.merge(traces)  # will merge on case:concept:name

    # filter all events with a sequence found in variants
    filtered_log = merged_log[merged_log['sequence'].isin(variants['sequence'])]
    filtered_log = filtered_log[['case:concept:name', 'concept:name', 'time:timestamp']]

    return filtered_log

def get_variants(log, unordered=False, verbose=False):
    if verbose:
        print("# total:", len(log['case:concept:name'].unique()))

    variants = log.groupby('case:concept:name')['concept:name'].agg(tuple).to_dict()
    variants = Counter(variants.values())

    if verbose:
        print("# unique variants:", len(list(variants.keys())))
    if unordered:
        unvariants = {frozenbag(variant): 0 for variant in variants.keys()}
        for variant_series, cov_amt in variants.items():
            unvariants[frozenbag(variant_series)] += cov_amt
        if verbose:
            print("# unique unordered variants:", len(unvariants))
        return unvariants
    else:
        return variants


def get_variants_stats(log, unordered=False):
    variants = get_variants(log, unordered)
    variants = pd.DataFrame(variants.items())
    num_seq = len(log['case:concept:name'].unique())
    num_var = variants.shape[0]

    variants.columns = ['sequence', 'cov_amt']

    variants_sorted = variants.sort_values(by='cov_amt', ascending=False)
    variants_sorted = variants_sorted.reset_index(drop=True)
    variants_sorted['cov_perc'] = variants_sorted.apply(lambda row: 100 / num_seq * row['cov_amt'], axis=1)

    cumul_cov_perc = []
    cur_cov_perc = 0
    for _, row in variants_sorted.iterrows():  # yeah, yeah ...
        cur_cov_perc += row['cov_perc']
        cumul_cov_perc.append(cur_cov_perc)
    variants_sorted['cov_perc_cumul'] = cumul_cov_perc

    var_perc = 100 / num_var
    variants_sorted['var_perc_cumul'] = variants_sorted.apply(lambda row: (row.name + 1) * var_perc, axis=1)

    return variants_sorted


def print_variants_stats(vars_stats):
    for idx, row in vars_stats.iterrows():
        print(f"coverage: amt = {row['cov_amt']}, perc = {row['cov_perc']}, cumul = {row['cov_perc_cumul']}")
        print(f"variant: cnt = {row.name}, perc = {row['var_perc_cumul']}")
        print(row['sequence'])


# how many cases (percentage) do var_perc of variants (sorted desc by coverage) cover?
def get_case_coverage(var_perc, vars_stats):
    return get_x_coverage(var_perc, 'var_perc_cumul', 'cov_perc_cumul', vars_stats)


# how many variants (percentage) are needed (sorted desc by coverage) to cover case_perc of cases?
def get_variant_coverage(case_perc, vars_stats):
    return get_x_coverage(case_perc, 'cov_perc_cumul', 'var_perc_cumul', vars_stats)


# get all variants that are needed (sorted desc by coverage) to cover case_perc of cases
def get_covering_variants(case_perc, vars_stats):
    first_k = vars_stats.loc[vars_stats['cov_perc_cumul'] <= case_perc]
    return first_k


def get_x_coverage(perc, cmp_col, ret_col, vars_stats):
    first_k = vars_stats.loc[vars_stats[cmp_col] <= perc]
    return first_k.iloc[-1][ret_col]


def filter_traces_on_variants(log, variants):
    traces = log.groupby('case:concept:name')['concept:name'].apply(tuple).rename('sequence').reset_index()
    # add case's sequence to all events of that case
    merged_log = log.merge(traces)  # will merge on case:concept:name

    # filter all events with a sequence found in variants
    filtered_log = merged_log[merged_log['sequence'].isin(variants['sequence'])]
    filtered_log = filtered_log[['case:concept:name', 'concept:name', 'time:timestamp']]

    return filtered_log


def is_subsequence(sub, full):
    it = iter(full)
    return all(item in it for item in sub)

def inimportant(df0,top=5,bottom=10, info = True):
    seq = [list(df0['sequence'].iloc[i]) for i in range(top)]
    seq2pat = Seq2Pat(seq)
    top_patterns = seq2pat.get_patterns(min_frequency=1)
    remove = []

    for t in range(bottom):
        trace = df0.tail(bottom)['sequence'].iloc[t]

        # Check for exact match
        found_full_trace = any(trace == p for p in top_patterns)

        # Check for adjacent subsequences (pairs of elements)
        adjacent_subsequences = []
        for i in range(len(trace)):
            for j in range(i + 1, len(trace)):
                subsequence = trace[i:j + 1]  # Adjacent subsequences
                found = [p for p in top_patterns if subsequence == p]
                if found:
                    adjacent_subsequences.append(subsequence)

        # Check for non-adjacent subsequences (elements in order but not adjacent)

        non_adjacent_subsequences = [sub for sub in
                                     (trace[i:j] for i in range(len(trace)) for j in range(i + 1, len(trace) + 1)) if
                                     any(is_subsequence(sub, p) for p in top_patterns)]
        non_adjacent_subsequences = [non for non in non_adjacent_subsequences if len(non) > (len(trace) / 2) + 1]

        if info:
            print(f"\n\n\nThis is trace -> {trace} | cov -> {df0.tail(bottom)['cov_amt'].iloc[t]}")
            # Output
            if found_full_trace:
                print("\nThe full trace exists in the top patterns.")
            else:
                print("\nThe full trace does not exist in the top patterns.")

            if adjacent_subsequences:
                print(f"\nFound adjacent subsequences: {adjacent_subsequences}")
            else:
                print("\nNo adjacent subsequences found.")

            if non_adjacent_subsequences:
                print(f"\nFound non-adjacent subsequences: {len(non_adjacent_subsequences)}")
                # print([non for non in non_adjacent_subsequences if len(non) > len(trace)/2])
            else:
                print("\nNo non-adjacent subsequences found.")

        if (not found_full_trace) and len(adjacent_subsequences) == 0 and len(non_adjacent_subsequences) == 0:
            remove.append(trace)

    return remove

def DF_Modifier(df0,remove):
    filtered_df = df0[~df0['sequence'].isin(remove)]
    filteredlog = [list(row[0]) for _, row in filtered_df.iterrows() for _ in range(row[1])]
    # Initialize an empty list to hold event log rows
    new_log = []

    # Iterate over each decision path in the dataset
    timestamp = pd.Timestamp('2024-11-14')
    case_id = -1
    for case_id, trace in enumerate(filteredlog):
        for t, activity in enumerate(trace):
            event = {
                'case_id': case_id,
                'activity': activity,
                'timestamp': timestamp + pd.Timedelta(days=t)
            }
            new_log.append(event)

    new_log = pd.DataFrame(new_log)

    return new_log

def eventlog_creator(df0,labels,label_list):
     # Create an empty event log
    event_log = EventLog()

    for C in label_list:
        sequences = df0.iloc[labels==labels[C]]['sequence']
        
        # Define base timestamp
        base_time = datetime(2024, 1, 1, 0, 0, 0)

        # Loop through each sequence
        for idx, sequence in enumerate(sequences):
            trace = Trace()
            case_id = f"Case_{idx+1}"
            
            for i, activity in enumerate(sequence):
                event = {
                    "case:concept:name": case_id,
                    "concept:name": activity,
                    "time:timestamp": base_time + timedelta(minutes=i)
                }
                trace.append(event)
            
            event_log.append(trace)

    return event_log

def trace2eventlog(traces):
    event_log = EventLog()
    base_time = datetime(2024, 1, 1, 0, 0, 0)
    for idx, sequence in enumerate(traces):
        trace = Trace()
        case_id = f"Case_{idx+1}"
        
        for i, activity in enumerate(sequence):
            event = {
                "case:concept:name": case_id,
                "concept:name": activity,
                "time:timestamp": base_time + timedelta(minutes=i)
            }
            trace.append(event)
        
        event_log.append(trace)

    return event_log


def export_xes(event_log, PATH):

    xes_exporter.apply(event_log, f"{PATH}.xes")