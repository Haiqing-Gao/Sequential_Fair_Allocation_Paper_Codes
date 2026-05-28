import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scienceplots
from matplotlib.ticker import PercentFormatter

# Update style and LaTeX settings
script_dir = os.path.dirname(os.path.abspath(__file__))
style_path = os.path.join(script_dir, 'PaperDoubleFig.mplstyle.txt')
plt.style.use(style_path)
plt.rc('text.latex', preamble=r'\usepackage{amsfonts}')
plt.rcParams['text.usetex'] = True

def load_data(same_CV=True):
    if same_CV:
        settings = ["same_mean", "same_var", "random", "same_CV"]
    else:
        settings = ["same_mean", "same_var", "random"]
    horizon = 3
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    all_data = []

    for setting in settings:
        dynamic_dir = os.path.join(script_dir, f'dynamic_{setting}_results')
        hete_dir = os.path.join(script_dir, f'hete_static_{setting}_results')
        ppa_dir = os.path.join(script_dir, f'ppa_{setting}_results')
        
        if not os.path.exists(dynamic_dir):
            print(f"Warning: {dynamic_dir} does not exist.")
        else:
            # Get all dynamic files
            dynamic_files = [f for f in os.listdir(dynamic_dir) if f.endswith('.csv')]
            for f in dynamic_files:
                try:
                    df = pd.read_csv(os.path.join(dynamic_dir, f))
                    if not df.empty:
                        df['curve'] = 'Joint Optimal'
                        df['setting'] = setting
                        all_data.append(df)
                except Exception as e:
                    print(f"Error reading {f}: {e}")

        if not os.path.exists(hete_dir):
            print(f"Warning: {hete_dir} does not exist.")
        else:
            # Get all dynamic files
            hete_worst_files = [f for f in os.listdir(hete_dir) if (f.startswith('worst_route') or f.startswith('worst_static')) and f.endswith('.csv')]
            for f in hete_worst_files:
                try:
                    df = pd.read_csv(os.path.join(hete_dir, f))
                    if not df.empty:
                        df['curve'] = 'Optimal Alloc + Worst Static'
                        df['setting'] = setting
                        all_data.append(df)
                except Exception as e:
                    print(f"Error reading {f}: {e}")
            
        if not os.path.exists(ppa_dir):
            print(f"Warning: {ppa_dir} does not exist.")
        else:
            # Get all ppa_worst_route or ppa_worst_static files
            ppa_worst_files = [f for f in os.listdir(ppa_dir) if (f.startswith('ppa_worst_route') or f.startswith('ppa_worst_static')) and f.endswith('.csv')]
            for f in ppa_worst_files:
                try:
                    df = pd.read_csv(os.path.join(ppa_dir, f))
                    if not df.empty:
                        df['curve'] = 'PPA + Worst Static'
                        df['setting'] = setting
                        all_data.append(df)
                except Exception as e:
                    print(f"Error reading {f}: {e}")
            
            # Get all ppa_decreasing_CV files
            ppa_dec_files = [f for f in os.listdir(ppa_dir) if f.startswith('ppa_decreasing_CV') and f.endswith('.csv')]
            for f in ppa_dec_files:
                try:
                    df = pd.read_csv(os.path.join(ppa_dir, f))
                    if not df.empty:
                        df['curve'] = 'PPA-deCV'
                        df['setting'] = setting
                        all_data.append(df)
                except Exception as e:
                    print(f"Error reading {f}: {e}")
            
    if not all_data:
        return pd.DataFrame()
        
    df_all = pd.concat(all_data, ignore_index=True)
    # Round supply for grouping
    df_all['supply_rounded'] = df_all['supply_divided_by_mean_demand'].round(1)
    # Efficiency calculation
    df_all['efficiency'] = (df_all['initial_supply'] - df_all['expected_waste']) / df_all['initial_supply']
    df_all['waste'] = df_all['expected_waste'] / df_all['initial_supply']
    
    return df_all

def plot_tradeoff(df, algorithm, metric, ylabel, ax1, ylim=0.35, x_axis='efficiency', curves=['Joint Optimal', 'PPA + Worst Static'], show_secondary=True, label_suffix="", secondary_metric=None, separate_algorithms_for=['Joint Optimal']):
    # Filter by curves first
    df_filtered = df[df['curve'].isin(curves)].copy()
    
    # Filter by algorithm
    if isinstance(algorithm, list):
        # Filter for rows that contain any of the algorithms in the list
        df_filtered = df_filtered[df_filtered['algorithm'].str.contains('|'.join(algorithm))]
        
        # Determine which rows to aggregate and which to keep separate
        # Curves in separate_algorithms_for will keep their algorithms
        # Others will be aggregated
        
        to_keep = df_filtered[df_filtered['curve'].isin(separate_algorithms_for)].copy()
        to_aggregate = df_filtered[~df_filtered['curve'].isin(separate_algorithms_for)].copy()
        
        if not to_aggregate.empty:
            to_aggregate['algorithm'] = 'Aggregated'
            
        df_alg = pd.concat([to_keep, to_aggregate])
    else:
        df_alg = df_filtered[df_filtered['algorithm'].str.startswith(algorithm)]
    
    if df_alg.empty:
        print(f"No data for algorithm {algorithm}")
        return

    # Average data across settings and types for each curve, algorithm, and supply level
    group_cols = ['curve', 'algorithm', 'supply_rounded']
    
    # We need to handle secondary metrics if present
    agg_dict = {x_axis: 'mean', metric: 'mean'}
    if (x_axis == 'waste' or x_axis == 'efficiency') and x_axis not in df_alg.columns:
        df_alg = df_alg.copy()
        df_alg['waste'] = df_alg['expected_waste'] / df_alg['initial_supply']
        df_alg['efficiency'] = (df_alg['initial_supply'] - df_alg['expected_waste']) / df_alg['initial_supply']

    if show_secondary:
        # We need to find which secondary metrics are available
        secondary_cols = [secondary_metric] if secondary_metric else ['haiqing_obj', 'one_time_fairness', 'waste']
        for col in secondary_cols:
            if col in df_alg.columns:
                agg_dict[col] = 'mean'
                
    df_plot = df_alg.groupby(group_cols).agg(agg_dict).reset_index()
    df_plot = df_plot.sort_values(['curve', 'algorithm', x_axis])

    # Algorithm and Curve color mapping
    alg_map = {'haiqing': 'Forward', 'manshadi': 'Ex-post', 'haiqing_ppa': 'Forward', 'manshadi_ppa': 'Ex-post', 'Aggregated': ''}
    
    colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange', 'tab:purple', 'tab:brown']
    
    # Unique combinations of curve and algorithm
    unique_combos = df_plot[['curve', 'algorithm']].drop_duplicates().values
    
    # Define markers for different curves and algorithms
    curve_markers = {
        'Joint Optimal': 'o',
        'PPA + Worst Static': 's',
        'PPA-deCV': '^',
        'Optimal Alloc + Worst Static': 'D',
        'Aggregated': 'o'
    }
    
    for i, (curve, alg) in enumerate(unique_combos):
        data = df_plot[(df_plot['curve'] == curve) & (df_plot['algorithm'] == alg)]
        alg_name = alg_map.get(alg, alg)
        if alg == 'Aggregated':
            label = f"{curve}"
        else:
            label = f"{curve} ({alg_name})" if len(unique_combos) > 1 else f"{curve}"
        
        color = colors[i % len(colors)]
        marker = curve_markers.get(curve, 'o')
        # If we have both algorithms for the same curve, we need to vary markers
        if curve == 'Joint Optimal':
             if alg == 'haiqing' or alg == 'haiqing_ppa':
                 marker = 'o'
             elif alg == 'manshadi' or alg == 'manshadi_ppa':
                 marker = 'v'
        elif curve == 'PPA + Worst Static':
             if alg == 'haiqing' or alg == 'haiqing_ppa':
                 marker = 's'
             elif alg == 'manshadi' or alg == 'manshadi_ppa':
                 marker = 'p' # pentagon
             else:
                 marker = 's'
        elif curve == 'Optimal Alloc + Worst Static':
             if alg == 'haiqing':
                 marker = 'D'
             elif alg == 'manshadi':
                 marker = 'P' # plus
             else:
                 marker = 'D'

        ax1.plot(data[x_axis], data[metric], marker=marker, color=color, label=label, markersize=12)

    ax1.set_ylabel(ylabel)
    ax1.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax1.set_ylim(0, ylim)

    if show_secondary:
        # Create secondary y-axis
        ax2 = ax1.twinx()
        for i, (curve, alg) in enumerate(unique_combos):
            data = df_plot[(df_plot['curve'] == curve) & (df_plot['algorithm'] == alg)]
            color = colors[i % len(colors)]
            
            # Determine which secondary metric to use
            if secondary_metric:
                sec_metric = secondary_metric
            elif alg == 'haiqing' or alg == 'haiqing_ppa':
                sec_metric = 'haiqing_obj'
            elif alg == 'manshadi' or alg == 'manshadi_ppa':
                sec_metric = 'one_time_fairness'
            elif curve == 'Optimal Alloc + Worst Static':
                if 'haiqing_obj' in data.columns and not data['haiqing_obj'].isna().all():
                    sec_metric = 'haiqing_obj'
                elif 'one_time_fairness' in data.columns and not data['one_time_fairness'].isna().all():
                    sec_metric = 'one_time_fairness'
                else:
                    continue
            else:
                if 'haiqing_obj' in data.columns and not data['haiqing_obj'].isna().all():
                    sec_metric = 'haiqing_obj'
                elif 'one_time_fairness' in data.columns and not data['one_time_fairness'].isna().all():
                    sec_metric = 'one_time_fairness'
                elif 'waste' in data.columns and not data['waste'].isna().all():
                    sec_metric = 'waste'
                else:
                    continue
            
            # Use distinct markers for secondary axis if they belong to the same algorithm/curve
            if curve == 'Joint Optimal':
                sec_marker = '*' if (alg == 'haiqing' or alg == 'haiqing_ppa') else 'h'
            else:
                sec_marker = 'X'
                
            alg_name = alg_map.get(alg, alg)
            sec_label_part = f"{curve} ({alg_name})" if len(unique_combos) > 1 else f"{curve}"
            metric_label = r"$\Delta_{\mathit{eff}}$" if sec_metric == 'efficiency' or sec_metric == 'waste' else "Obj"
            label = f"{sec_label_part} {metric_label}"
            
            ax2.plot(data[x_axis], data[sec_metric], linestyle='--', marker=sec_marker, color=color, alpha=0.75, markersize=12, label=label)
        
        # Set secondary y-axis label - if multiple, we might need a generic one or just the symbols
        sec_label = r"$\Delta_{\mathit{eff}}$" if secondary_metric == 'waste' else "Objective Values"
        ax2.set_ylabel(sec_label)
        ax2.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    
    x_label = r"$\Delta_{\mathit{eff}}$" if (x_axis == 'efficiency' or x_axis == 'waste') else "Capacity Level"
    ax1.set_xlabel(x_label)
    
    if x_axis == 'efficiency':
        ax1.set_xlim(left=0.5)
        ax1.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    elif x_axis == 'waste':
        ax1.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    else:
        ax1.set_xlim(0.1, 1.8)

def plot_algorithm_comparison(df, metric, ylabel, ax):
    # Filter for 'Joint Optimal' curve and algorithms 'haiqing' and 'manshadi'
    df_filtered = df[(df['curve'] == 'Joint Optimal') & (df['algorithm'].isin(['haiqing', 'manshadi']))].copy()
    
    if df_filtered.empty:
        print(f"No data for algorithm comparison for metric {metric}")
        return

    # Map algorithm names to legend labels
    label_map = {
        'haiqing': 'forward',
        'manshadi': 'expost'
    }
    df_filtered['algorithm_label'] = df_filtered['algorithm'].map(label_map)

    # Average data across settings and types for each algorithm and supply level
    df_plot = df_filtered.groupby(['algorithm_label', 'supply_rounded']).agg({
        'efficiency': 'mean',
        metric: 'mean'
    }).reset_index().sort_values(['algorithm_label', 'efficiency'])

    colors = {'forward': 'tab:blue', 'expost': 'tab:red'}
    markers = {'forward': 'o', 'expost': 'v'}
    for alg in ['forward', 'expost']:
        data = df_plot[df_plot['algorithm_label'] == alg]
        ax.plot(data['efficiency'], data[metric], linestyle='-', marker=markers[alg], color=colors[alg], label=alg, markersize=12)
    
    ax.set_xlabel(r"$\Delta_{\mathit{eff}}$")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 0.35)
    # ax.set_xlim(left=0.5)
    
    # Formatting
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))

def generate_gap_table(df):
    algorithms = ["haiqing", "manshadi"]
    metrics = ["ex_post_preference", "ex_ante_preference"]
    # Mapping for base objective metrics
    # \mexpost (haiqing) uses 'haiqing_obj' (Z*0)
    # \hqexpost (manshadi) uses 'one_time_fairness' (W*0)
    
    # Filter for the two curves we're comparing
    df_filtered = df[df['curve'].isin(['Joint Optimal', 'PPA + Worst Static'])].copy()
    
    # Normalize algorithm names
    df_filtered['algorithm_base'] = df_filtered['algorithm'].apply(lambda x: 'haiqing' if x.startswith('haiqing') else 'manshadi')
    
    # Aggregate data across settings and types for each (algorithm, curve, supply_rounded)
    df_agg = df_filtered.groupby(['algorithm_base', 'curve', 'supply_rounded']).agg({
        'ex_post_preference': 'mean',
        'ex_ante_preference': 'mean',
        'haiqing_obj': 'mean',
        'one_time_fairness': 'mean'
    }).reset_index()

    # Pivot to have curves as columns
    pivot_metrics = ['ex_post_preference', 'ex_ante_preference', 'haiqing_obj', 'one_time_fairness']
    df_pivot = df_agg.pivot(index=['algorithm_base', 'supply_rounded'], columns='curve', values=pivot_metrics)
    df_pivot.columns = [f"{metric}_{curve}" for metric, curve in df_pivot.columns]
    df_pivot = df_pivot.reset_index()
    
    # Calculate gaps: (Joint Optimal - PPA + Worst Static) * 100
    for metric in pivot_metrics:
        df_pivot[f"{metric}_gap"] = (df_pivot[f"{metric}_Joint Optimal"] - df_pivot[f"{metric}_PPA + Worst Static"]) * 100

    # We need a function to get stats for a specific algorithm and its relevant metrics
    def get_stats(alg):
        # Determine base obj metric for this algorithm
        base_obj_gap_col = 'haiqing_obj_gap' if alg == 'haiqing' else 'one_time_fairness_gap'
        
        alg_df = df_pivot[df_pivot['algorithm_base'] == alg]
        
        cols = {
            'base_obj': base_obj_gap_col,
            'EP': 'ex_ante_preference_gap',
            'EA': 'ex_post_preference_gap'
        }
        
        res = {}
        for key, col in cols.items():
            gap = alg_df[col]
            res[key] = {
                'Average gap': gap.mean(),
                'Max positive gap': gap.max(),
                'Max negative gap': gap.min(),
                'Median': gap.median()
            }
        return res

    all_results = {
        'haiqing': get_stats('haiqing'),
        'manshadi': get_stats('manshadi')
    }

    # Generate LaTeX
    latex = "\\begin{table}[h]\n"
    latex += "\\centering\n"
    latex += "\\begin{tabular}{l cc cc}\n"
    latex += "\\hline\\hline\n"
    latex += "& \\multicolumn{2}{c}{\\textbf{\\mexpost}} & \\multicolumn{2}{c}{\\textbf{\\hqexpost}} \\\\\n"
    latex += "\\cline{2-3} \\cline{4-5}\n"
    latex += "\\textbf{Performance measure (\\%)} & \\textbf{$\\DfairEP$} & \\textbf{$\\DfairEA$} & \\textbf{$\\DfairEP$} & \\textbf{$\\DfairEA$} \\\\\n"
    latex += "\\hline\n"
    
    rows = [
        ('Average gap', 'Average gap'),
        ('Max pos/neg gap', 'Max pos/neg gap'),
        ('Median', 'Median')
    ]
    
    for row_label, stat_key in rows:
        line = f"{row_label:<18}"
        if stat_key == 'Max pos/neg gap':
            # mexpost (haiqing)
            for col in ['EP', 'EA']:
                pos = all_results['haiqing'][col]['Max positive gap']
                neg = all_results['haiqing'][col]['Max negative gap']
                if abs(pos) > abs(neg):
                    line += f" & {pos:.1f}"
                else:
                    line += f" & {neg:.1f}"
            # hqexpost (manshadi)
            for col in ['EP', 'EA']:
                pos = all_results['manshadi'][col]['Max positive gap']
                neg = all_results['manshadi'][col]['Max negative gap']
                if abs(pos) > abs(neg):
                    line += f" & {pos:.1f}"
                else:
                    line += f" & {neg:.1f}"
        else:
            # mexpost (haiqing)
            for col in ['EP', 'EA']:
                val = all_results['haiqing'][col][stat_key]
                line += f" & {val:.1f}"
            # hqexpost (manshadi)
            for col in ['EP', 'EA']:
                val = all_results['manshadi'][col][stat_key]
                line += f" & {val:.1f}"
        
        line += " \\\\\n"
        latex += line

    latex += "\\hline\\hline\n"
    latex += "\\end{tabular}\n"
    latex += "\\caption{Differences between the Joint Optimal and PPA + Worst Static policies. Entries are (Joint Optimal) minus (PPA + Worst Static).}\n"
    latex += "\\label{tab:jo_vs_ppa}\n"
    latex += "\\end{table}"
    
    print("\nGenerated LaTeX Table:")
    print(latex)
    
    # Save to file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, "joint_opt_gaps.tex"), "w") as f:
        f.write(latex)
    
    return latex

def load_homo_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    homo_dir = os.path.join(script_dir, 'homo_results')
    
    if not os.path.exists(homo_dir):
        print(f"Warning: {homo_dir} does not exist.")
        return pd.DataFrame()
    
    all_data = []
    homo_files = [f for f in os.listdir(homo_dir) if f.endswith('.csv')]
    for f in homo_files:
        try:
            df = pd.read_csv(os.path.join(homo_dir, f))
            if not df.empty:
                df['setting'] = 'homo'
                all_data.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    if not all_data:
        return pd.DataFrame()
        
    df_all = pd.concat(all_data, ignore_index=True)
    df_all['supply_rounded'] = df_all['supply_divided_by_mean_demand'].round(1)
    df_all['curve'] = 'Joint Optimal'
    return df_all

def generate_homo_plot(df_homo):
    if df_homo.empty:
        print("No homogeneous data loaded.")
        return

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Calculate waste and efficiency
    if 'waste' not in df_homo.columns:
        if 'expected_waste' in df_homo.columns and 'initial_supply' in df_homo.columns:
            df_homo['waste'] = df_homo['expected_waste'] / df_homo['initial_supply']
        elif 'efficiency' in df_homo.columns:
            df_homo['waste'] = 1 - df_homo['efficiency']
            
    if 'efficiency' not in df_homo.columns:
        if 'waste' in df_homo.columns:
            df_homo['efficiency'] = 1 - df_homo['waste']
        elif 'expected_waste' in df_homo.columns and 'initial_supply' in df_homo.columns:
            df_homo['efficiency'] = 1 - (df_homo['expected_waste'] / df_homo['initial_supply'])

    # 1. Homo Tradeoff vs Capacity Level
    fig1, axes1 = plt.subplots(1, 2, figsize=(15, 6))
    
    metrics = [
        ('ex_post_preference', r'$\Delta^{\mathit{XP}}_{\mathit{fair}}$'),
        ('ex_ante_preference', r'$\Delta^{\mathit{XA}}_{\mathit{fair}}$')
    ]
    
    alg_map = {'haiqing': 'Forward', 'manshadi': 'Ex-post'}
    colors = {'haiqing': 'tab:blue', 'manshadi': 'tab:red'}
    markers = {'haiqing': 'o', 'manshadi': 'v'}
    
    for i, (metric, ylabel) in enumerate(metrics):
        # Aggregate data across all files for each algorithm and supply level
        df_plot = df_homo.groupby(['algorithm', 'supply_rounded']).agg({
            metric: 'mean'
        }).reset_index()
        df_plot = df_plot.sort_values(['algorithm', 'supply_rounded'])
        
        for alg in ['haiqing', 'manshadi']:
            data = df_plot[df_plot['algorithm'] == alg]
            if data.empty:
                continue
            
            label = alg_map.get(alg, alg)
            color = colors.get(alg, 'black')
            marker = markers.get(alg, 'o')
            
            axes1[i].plot(data['supply_rounded'], data[metric], linestyle='-', marker=marker, color=color, label=label, markersize=10)
            
        axes1[i].set_ylabel(ylabel)
        axes1[i].set_xlabel("Capacity Level")
        axes1[i].yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        axes1[i].set_ylim(0, 0.40)
        axes1[i].set_xlim(0.1, 1.8)

    handles, labels = axes1[0].get_legend_handles_labels()
    fig1.legend(handles, labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, -0.1))
    
    plt.subplots_adjust(wspace=0.3, hspace=0.3, left=0.1, right=0.9, top=0.9, bottom=0.1)
    save_path1 = os.path.join(output_dir, 'homo_tradeoff_vs_capacity.pdf')
    fig1.savefig(save_path1, bbox_inches='tight', pad_inches=0.1, dpi=900)
    print(f"Saved homogeneous capacity plot to {save_path1}")
    plt.close(fig1)

    # 2. Homo Tradeoff vs Efficiency (Waste)
    fig2, axes2 = plt.subplots(1, 2, figsize=(15, 6))
    
    for i, (metric, ylabel) in enumerate(metrics):
        # Aggregate data across all files for each algorithm and supply level
        df_plot = df_homo.groupby(['algorithm', 'supply_rounded']).agg({
            'efficiency': 'mean',
            metric: 'mean'
        }).reset_index()
        df_plot = df_plot.sort_values(['algorithm', 'efficiency'])
        
        for alg in ['haiqing', 'manshadi']:
            data = df_plot[df_plot['algorithm'] == alg]
            if data.empty:
                continue
            
            label = alg_map.get(alg, alg)
            color = colors.get(alg, 'black')
            marker = markers.get(alg, 'o')
            
            axes2[i].plot(data['efficiency'], data[metric], linestyle='-', marker=marker, color=color, label=label, markersize=10)
            
        axes2[i].set_ylabel(ylabel)
        axes2[i].set_xlabel(r"$\Delta_{\mathit{eff}}$")
        axes2[i].xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        axes2[i].yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        axes2[i].set_ylim(0, 0.40)

    handles, labels = axes2[0].get_legend_handles_labels()
    fig2.legend(handles, labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, -0.1))
    
    plt.subplots_adjust(wspace=0.3, hspace=0.3, left=0.1, right=0.9, top=0.9, bottom=0.1)
    save_path2 = os.path.join(output_dir, 'homo_tradeoff_vs_efficiency.pdf')
    fig2.savefig(save_path2, bbox_inches='tight', pad_inches=0.1, dpi=900)
    print(f"Saved homogeneous efficiency plot to {save_path2}")
    plt.close(fig2)

def generate_tradeoff_vs_efficiency_plot(df_combined):
    if df_combined.empty:
        print("No combined data loaded.")
        return

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Calculate waste and efficiency
    if 'waste' not in df_combined.columns:
        if 'expected_waste' in df_combined.columns and 'initial_supply' in df_combined.columns:
            df_combined['waste'] = df_combined['expected_waste'] / df_combined['initial_supply']
        elif 'efficiency' in df_combined.columns:
            df_combined['waste'] = 1 - df_combined['efficiency']
            
    if 'efficiency' not in df_combined.columns:
        if 'waste' in df_combined.columns:
            df_combined['efficiency'] = 1 - df_combined['waste']
        elif 'expected_waste' in df_combined.columns and 'initial_supply' in df_combined.columns:
            df_combined['efficiency'] = 1 - (df_combined['expected_waste'] / df_combined['initial_supply'])

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    metrics = [
        ('ex_post_preference', r'$\Delta^{\mathit{XP}}_{\mathit{fair}}$'),
        ('ex_ante_preference', r'$\Delta^{\mathit{XA}}_{\mathit{fair}}$')
    ]
    
    alg_map = {'haiqing': 'Forward', 'manshadi': 'Ex-post'}
    colors = {'haiqing': 'tab:blue', 'manshadi': 'tab:red'}
    markers = {'haiqing': 'o', 'manshadi': 'v'}
    
    for i, (metric, ylabel) in enumerate(metrics):
        # Aggregate data across all files for each algorithm and supply level
        df_plot = df_combined.groupby(['algorithm', 'supply_rounded']).agg({
            'efficiency': 'mean',
            metric: 'mean'
        }).reset_index()
        df_plot = df_plot.sort_values(['algorithm', 'efficiency'])
        
        for alg in ['haiqing', 'manshadi']:
            data = df_plot[df_plot['algorithm'] == alg]
            if data.empty:
                continue
            
            label = alg_map.get(alg, alg)
            color = colors.get(alg, 'black')
            marker = markers.get(alg, 'o')
            
            axes[i].plot(data['efficiency'], data[metric], linestyle='-', marker=marker, color=color, label=label, markersize=12)
            
        axes[i].set_ylabel(ylabel)
        axes[i].set_xlabel(r"$\Delta_{\mathit{eff}}$")
        axes[i].xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        axes[i].yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        axes[i].set_ylim(0, 0.3)
        # ax.set_xlim(left=0) # Optionally set xlim if needed

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, -0.1))
    
    plt.subplots_adjust(wspace=0.3, hspace=0.3, left=0.1, right=0.9, top=0.9, bottom=0.1)
    save_path = os.path.join(output_dir, 'tradeoff_vs_efficiency.pdf')
    fig.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=900)
    print(f"Saved tradeoff vs efficiency plot to {save_path}")
    plt.close(fig)

# def load_worst_static_data(same_CV=True):
#     if same_CV:
#         settings = ["same_mean", "same_var", "random", "same_CV"]
#     else:
#         settings = ["same_mean", "same_var", "random"]
#     script_dir = os.path.dirname(os.path.abspath(__file__))
#
#     all_data = []
#
#     for setting in settings:
#         hete_dir = os.path.join(script_dir, f'hete_static_{setting}_results')
#
#         if not os.path.exists(hete_dir):
#             print(f"Warning: {hete_dir} does not exist.")
#         else:
#             worst_static_files = [f for f in os.listdir(hete_dir) if f.startswith('worst_static_') and f.endswith('.csv')]
#             for f in worst_static_files:
#                 try:
#                     df = pd.read_csv(os.path.join(hete_dir, f))
#                     if not df.empty:
#                         df['curve'] = 'Optimal Alloc + Worst Static'
#                         df['setting'] = setting
#                         all_data.append(df)
#                 except Exception as e:
#                     print(f"Error reading {f}: {e}")
#
#     if not all_data:
#         return pd.DataFrame()
#
#     df_all = pd.concat(all_data, ignore_index=True)
#     df_all['supply_rounded'] = df_all['supply_divided_by_mean_demand'].round(1)
#
#     return df_all

def main():
    df = load_data()
    # df_worst = load_worst_static_data()
    # df_with_worst = pd.concat([df, df_worst], ignore_index=True) if not df_worst.empty else df

    if df.empty:
        print("No data loaded.")
        return
    
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Helper to save grouped plots
    def save_group(fig, filename):
        plt.subplots_adjust(wspace=0.3, hspace=0.3, left=0.1, right=0.9, top=0.9, bottom=0.1)
        save_path = os.path.join(output_dir, filename)
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=900)
        print(f"Saved grouped plot to {save_path}")
        plt.close(fig)

    # 1. Tradeoff vs Efficiency (4 plots)
    df_homo = load_homo_data()
    df_combined = pd.concat([df, df_homo], ignore_index=True) if not df_homo.empty else df
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    axes = axes.flatten()
    configs = [
        ('haiqing', 'ex_post_preference', r'$\Delta^{\mathit{XP}}_{\mathit{fair}}$'),
        ('haiqing', 'ex_ante_preference', r'$\Delta^{\mathit{XA}}_{\mathit{fair}}$'),
        ('manshadi', 'ex_post_preference', r'$\Delta^{\mathit{XP}}_{\mathit{fair}}$'),
        ('manshadi', 'ex_ante_preference', r'$\Delta^{\mathit{XA}}_{\mathit{fair}}$'),
    ]
    for i, (alg, metric, ylabel) in enumerate(configs):
        plot_tradeoff(df_combined, alg, metric, ylabel, axes[i], x_axis='efficiency', show_secondary=False)
    
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, -0.05))
    save_group(fig, 'grouped_tradeoff_vs_efficiency.pdf')

    # 2. PPA + Decreasing CV (2 plots, combined algorithms)
    df_without_same_CV = load_data(same_CV=False)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    configs_dec = [
        ('ex_post_preference', r'$\Delta^{\mathit{XP}}_{\mathit{fair}}$'),
        ('ex_ante_preference', r'$\Delta^{\mathit{XA}}_{\mathit{fair}}$'),
    ]
    for i, (metric, ylabel) in enumerate(configs_dec):
        plot_tradeoff(df_without_same_CV, ['haiqing', 'manshadi'], metric, ylabel, axes[i], ylim=0.3, curves=['Joint Optimal', 'PPA-deCV'], x_axis='efficiency', show_secondary=False)
    
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, -0.1))
    save_group(fig, 'grouped_tradeoff_vs_efficiency_dec.pdf')

    # 3. Algorithm Comparison (2 plots)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    comparison_configs = [
        ('ex_post_preference', r'$\Delta^{\mathit{XP}}_{\mathit{fair}}$'),
        ('ex_ante_preference', r'$\Delta^{\mathit{XA}}_{\mathit{fair}}$'),
    ]
    for i, (metric, ylabel) in enumerate(comparison_configs):
        plot_algorithm_comparison(df, metric, ylabel, axes[i])
    
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, -0.1))
    save_group(fig, 'grouped_algorithm_comparison.pdf')

    # 4. Capacity Level (2 plots, combined algorithms)
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    configs_cap = [
        ('ex_post_preference', r'$\Delta^{\mathit{XP}}_{\mathit{fair}}$'),
        ('ex_ante_preference', r'$\Delta^{\mathit{XA}}_{\mathit{fair}}$'),
    ]
    for i, (metric, ylabel) in enumerate(configs_cap):
        plot_tradeoff(df, ['haiqing', 'manshadi'], metric, ylabel, axes[i], ylim=0.3, x_axis='supply_divided_by_mean_demand', curves=['Joint Optimal', 'PPA + Worst Static'], show_secondary=True)
    
    # Filter unique handles and labels
    handles, labels = axes[0].get_legend_handles_labels()
    # Add secondary axis handles from ax2 of the first subplot
    # Simplistic check for secondary axes
    sec_axes = [a for a in fig.axes if a.bbox.bounds == axes[0].bbox.bounds and a is not axes[0]]
    if sec_axes:
        h2, l2 = sec_axes[0].get_legend_handles_labels()
        handles += h2
        labels += l2
    
    unique_h_l = {}
    for h, l in zip(handles, labels):
        if l not in unique_h_l:
            unique_h_l[l] = h
    
    # We want a clean legend with all curve/algorithm and objective entries
    final_handles = list(unique_h_l.values())
    final_labels = list(unique_h_l.keys())
    
    # Sort or arrange them if needed, but showing all is safer
    fig.legend(final_handles, final_labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.15))
    save_group(fig, 'grouped_tradeoff_vs_capacity.pdf')

    # 4.1. New Capacity Level Plot with Worst Static Route (2 plots, combined algorithms)
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    for i, (metric, ylabel) in enumerate(configs_cap):
        plot_tradeoff(df, ['haiqing', 'manshadi'], metric, ylabel, axes[i],
                    ylim=0.37,
                    x_axis='supply_divided_by_mean_demand', 
                    curves=['Joint Optimal', 'PPA + Worst Static', 'Optimal Alloc + Worst Static'], 
                    show_secondary=False,
                    separate_algorithms_for=['Joint Optimal', 'Optimal Alloc + Worst Static'])
    
    # Filter unique handles and labels
    handles, labels = axes[0].get_legend_handles_labels()
    
    unique_h_l = {}
    for h, l in zip(handles, labels):
        if l not in unique_h_l:
            unique_h_l[l] = h
    
    final_handles = list(unique_h_l.values())
    final_labels = list(unique_h_l.keys())
    
    fig.legend(final_handles, final_labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.15))
    save_group(fig, 'tradeoff_vs_capacity_value_routing.pdf')

    # 4.2. New Objective vs Capacity Plot (2 subplots)
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    obj_configs = [
        ('one_time_fairness', r'$W_0^{\mathbf{\mu}}$'),
        ('haiqing_obj', r'$Z_0^{\mathbf{\mu}}$'),
    ]
    for i, (metric, ylabel) in enumerate(obj_configs):
        plot_tradeoff(df, ['haiqing', 'manshadi'], metric, ylabel, axes[i],
                    ylim=1.05,
                    x_axis='supply_divided_by_mean_demand',
                    curves=['Joint Optimal', 'PPA + Worst Static', 'Optimal Alloc + Worst Static'], 
                    show_secondary=False,
                    separate_algorithms_for=[]) # Aggregate all for objective plot
    
    # Filter unique handles and labels
    handles, labels = axes[0].get_legend_handles_labels()
    
    unique_h_l = {}
    for h, l in zip(handles, labels):
        if l not in unique_h_l:
            unique_h_l[l] = h
    
    final_handles = list(unique_h_l.values())
    final_labels = list(unique_h_l.keys())
    
    fig.legend(final_handles, final_labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.15))
    save_group(fig, 'obj_vs_capacity.pdf')

    # 5. Fair vs Efficiency (Capacity Level)
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Load and combine dynamic routing and homo data for this specific plot
    df_dynamic = load_data()
    df_homo = load_homo_data()
    df_combined = pd.concat([df_dynamic, df_homo], ignore_index=True)
    
    if not df_combined.empty:
        # Ensure efficiency and waste are calculated
        if 'efficiency' not in df_combined.columns:
            df_combined['efficiency'] = (df_combined['initial_supply'] - df_combined['expected_waste']) / df_combined['initial_supply']
        if 'waste' not in df_combined.columns:
            df_combined['waste'] = df_combined['expected_waste'] / df_combined['initial_supply']
        
        for i, (metric, ylabel) in enumerate(configs_cap):
            plot_tradeoff(df_combined, ['haiqing', 'manshadi'], metric, ylabel, axes[i], x_axis='supply_divided_by_mean_demand',
                        curves=['Joint Optimal'], show_secondary=True, secondary_metric='efficiency')
    else:
        print("No combined hete/homo data for Fair vs Efficiency plot.")
    
    # Filter unique handles and labels
    handles, labels = axes[0].get_legend_handles_labels()
    sec_axes = [a for a in fig.axes if a.bbox.bounds == axes[0].bbox.bounds and a is not axes[0]]
    if sec_axes:
        h2, l2 = sec_axes[0].get_legend_handles_labels()
        handles += h2
        labels += l2
    
    unique_h_l = {}
    for h, l in zip(handles, labels):
        if l not in unique_h_l:
            unique_h_l[l] = h
    
    final_handles = list(unique_h_l.values())
    final_labels = list(unique_h_l.keys())
    
    fig.legend(final_handles, final_labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.15))
    save_group(fig, 'grouped_fair_efficiency_vs_capacity.pdf')

    # # Generate the gap table
    # generate_gap_table(df)
    #
    # # Generate homogeneous plot
    # df_homo = load_homo_data()
    # # generate_homo_plot(df_homo)
    #
    # Generate tradeoff vs efficiency plot (combined hete and homo)
    df_combined = pd.concat([df, df_homo], ignore_index=True)
    generate_tradeoff_vs_efficiency_plot(df_combined)

def generate_pmf_plot():
    # Define the 8 dictionaries
    dists = [
        # 1. Uniform
        {1: 1/5, 2: 1/5, 3: 1/5, 4: 1/5, 5: 1/5},
        # 2. Symmetric concave
        {1: 1/10, 2: 1/5, 3: 2/5, 4: 1/5, 5: 1/10},
        # 3. Symmetric convex
        {1: 2/5, 2: 3/40, 3: 1/20, 4: 3/40, 5: 2/5},
        # 4. Decreasing
        {1: 2/5, 2: 3/10, 3: 1/5, 4: 7/100, 5: 3/100},
        # 5. Increasing
        {1: 3/100, 2: 7/100, 3: 1/5, 4: 3/10, 5: 2/5},
        # 6. Second peak
        {1: 3/10, 2: 2/5, 3: 1/5, 4: 7/100, 5: 3/100},
        # 7. Second valley
        {1: 1/5, 2: 7/100, 3: 3/100, 4: 3/10, 5: 2/5},
        # 8. W-shape
        {1: 2/5, 2: 3/100, 3: 3/10, 4: 7/100, 5: 1/5}
    ]
    
    names = [
        "Uniform", "Symmetric concave", "Symmetric convex", "Decreasing",
        "Increasing", "Second peak", "Second valley", "W-shape"
    ]
    
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for dist, name in zip(dists, names):
        # Use the same figure size as defined in PaperDoubleFig.mplstyle.txt (7.2, 4.45)
        # or slightly adjust if needed, but the style file should handle it.
        fig, ax = plt.subplots()
        
        # Increase font sizes slightly for PMF plots
        pmf_params = {
            "axes.labelsize": 32,
            "axes.titlesize": 36,
            "xtick.labelsize": 32,
            "ytick.labelsize": 32,
        }
        plt.rcParams.update(pmf_params)
        
        x = sorted(dist.keys())
        y = [dist[k] for k in x]
        
        ax.bar(x, y, color='tab:blue', alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.set_title(name)
        ax.set_xticks(x)
        ax.set_ylim(0, 0.5)
        ax.set_xlabel("Value")
        ax.set_ylabel("Probability")
        
        filename = f"pmf_{name.replace(' ', '_').lower()}.pdf"
        save_path = os.path.join(output_dir, filename)
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=900)
        print(f"Saved PMF plot to {save_path}")
        plt.close(fig)
    
    # Reset rcParams to original style for other plots
    plt.style.use(style_path)

if __name__ == "__main__":
    generate_pmf_plot()
    main()
