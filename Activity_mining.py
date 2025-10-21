
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class GlobalTraceSegmentation:
    """
    Implementation of the Global Trace Segmentation algorithm for Activity Mining.
    
    Based on the paper:
    "Activity Mining by Global Trace Segmentation"
    by Christian W. Günther, Anne Rozinat, and Wil M.P. van der Aalst
    """
    
    def __init__(self, window_size: int = 6, attenuation_factor: float = 0.8):
        """
        Initialize the Global Trace Segmentation algorithm.
        
        Parameters:
        -----------
        window_size : int
            The look-back window size for scanning event correlations (default: 6)
        attenuation_factor : float
            The attenuation factor 'a' for distance-based correlation decay (0 < a <= 1)
        """
        self.window_size = window_size
        self.attenuation_factor = attenuation_factor
        self.event_classes = None
        self.correlation_matrix = None
        self.cluster_hierarchy = None
        self.cluster_levels = None
        
    def fit(self, traces: List[List[str]]) -> 'GlobalTraceSegmentation':
        """
        Fit the segmentation model on the provided event log.
        
        Parameters:
        -----------
        traces : List[List[str]]
            List of traces, where each trace is a list of event class names
            
        Returns:
        --------
        self : GlobalTraceSegmentation
            Fitted model
        """
        print(f"Fitting model on {len(traces)} traces...")
        
        # Phase 1: Scan Global Event Class Correlation
        self._scan_correlation(traces)
        
        # Phase 2: Build Event Class Cluster Hierarchy
        self._build_hierarchy()
        
        print(f"✓ Model fitted successfully. Found {len(self.event_classes)} event classes.")
        return self
    
    def _scan_correlation(self, traces: List[List[str]]) -> None:
        """
        Phase 1: Scan the event log to build the global event class correlation matrix.
        """
        print("Phase 1: Scanning event class correlations...")
        
        # Identify all unique event classes
        all_events = set()
        for trace in traces:
            all_events.update(trace)
        self.event_classes = sorted(list(all_events))
        n_classes = len(self.event_classes)
        
        # Create mapping from event class to index
        event_to_idx = {event: idx for idx, event in enumerate(self.event_classes)}
        
        # Initialize correlation matrix
        self.correlation_matrix = np.zeros((n_classes, n_classes))
        
        # Scan through all traces
        for trace_idx, trace in enumerate(traces):
            # Process each event in the trace
            for pos in range(len(trace)):
                reference_event = trace[pos]
                ref_idx = event_to_idx[reference_event]
                
                # Look back within the window
                start_pos = max(0, pos - self.window_size)
                for look_back_pos in range(start_pos, pos):
                    previous_event = trace[look_back_pos]
                    prev_idx = event_to_idx[previous_event]
                    
                    # Calculate distance
                    distance = pos - look_back_pos
                    
                    # Update correlation with attenuation
                    correlation_increment = 1.0 * (self.attenuation_factor ** distance)
                    self.correlation_matrix[ref_idx, prev_idx] += correlation_increment
                    self.correlation_matrix[prev_idx, ref_idx] += correlation_increment  # Symmetric
        
        print(f"  ✓ Correlation matrix built ({n_classes} x {n_classes})")
    
    def _build_hierarchy(self) -> None:
        """
        Phase 2: Build the event class cluster hierarchy using Agglomerative Hierarchical Clustering.
        """
        print("Phase 2: Building cluster hierarchy...")
        
        n_classes = len(self.event_classes)
        
        # Initialize: each event class is its own cluster
        active_clusters = {i: {i} for i in range(n_classes)}  # cluster_id -> set of event indices
        cluster_correlation = self.correlation_matrix.copy()
        
        # Track hierarchy
        self.cluster_hierarchy = []
        self.cluster_levels = []
        
        # Store initial level (Level 0 - all separate)
        level_clusters = [{i} for i in range(n_classes)]
        self.cluster_levels.append(level_clusters)
        
        next_cluster_id = n_classes
        
        # Iteratively merge clusters
        for iteration in range(n_classes - 1):
            # Find the pair with highest correlation
            max_corr = -1
            best_pair = None
            
            cluster_ids = list(active_clusters.keys())
            for i in range(len(cluster_ids)):
                for j in range(i + 1, len(cluster_ids)):
                    id1, id2 = cluster_ids[i], cluster_ids[j]
                    corr = cluster_correlation[id1, id2]
                    if corr > max_corr:
                        max_corr = corr
                        best_pair = (id1, id2)
            
            if best_pair is None:
                break
            
            # Merge the best pair
            id1, id2 = best_pair
            new_cluster = active_clusters[id1] | active_clusters[id2]
            
            # Store merge information
            merge_info = {
                'level': iteration + 1,
                'merged': (id1, id2),
                'new_cluster_id': next_cluster_id,
                'cluster_members': new_cluster,
                'correlation': max_corr
            }
            self.cluster_hierarchy.append(merge_info)
            
            # Update active clusters
            del active_clusters[id1]
            del active_clusters[id2]
            active_clusters[next_cluster_id] = new_cluster
            
            # Update correlation matrix using complete linkage
            # Correlation between new cluster and any other = min of all pairwise correlations
            new_correlations = {}
            for other_id in active_clusters.keys():
                if other_id == next_cluster_id:
                    continue
                
                min_corr = float('inf')
                for member1 in new_cluster:
                    for member2 in active_clusters[other_id]:
                        corr = self.correlation_matrix[member1, member2]
                        min_corr = min(min_corr, corr)
                
                new_correlations[other_id] = min_corr
            
            # Expand correlation matrix
            n_active = len(active_clusters)
            new_corr_matrix = np.zeros((max(active_clusters.keys()) + 1, 
                                       max(active_clusters.keys()) + 1))
            
            for id_i in active_clusters.keys():
                for id_j in active_clusters.keys():
                    if id_i == next_cluster_id and id_j in new_correlations:
                        new_corr_matrix[id_i, id_j] = new_correlations[id_j]
                    elif id_j == next_cluster_id and id_i in new_correlations:
                        new_corr_matrix[id_i, id_j] = new_correlations[id_i]
                    elif id_i != next_cluster_id and id_j != next_cluster_id:
                        new_corr_matrix[id_i, id_j] = cluster_correlation[id_i, id_j]
            
            cluster_correlation = new_corr_matrix
            
            # Store this level's cluster configuration
            level_clusters = [cluster.copy() for cluster in active_clusters.values()]
            self.cluster_levels.append(level_clusters)
            
            next_cluster_id += 1
        
        print(f"  ✓ Hierarchy built with {len(self.cluster_hierarchy)} levels")
    
    def transform(self, traces: List[List[str]], abstraction_level: int) -> List[List[str]]:
        """
        Phase 3: Transform traces to the specified abstraction level.
        
        Parameters:
        -----------
        traces : List[List[str]]
            Original traces to transform
        abstraction_level : int
            The hierarchy level to use (0 = no abstraction, higher = more abstraction)
            
        Returns:
        --------
        transformed_traces : List[List[str]]
            Traces with events replaced by cluster labels
        """
        if self.cluster_levels is None:
            raise ValueError("Model must be fitted before transformation")
        
        if abstraction_level < 0 or abstraction_level >= len(self.cluster_levels):
            raise ValueError(f"Abstraction level must be between 0 and {len(self.cluster_levels)-1}")
        
        print(f"Phase 3: Transforming traces to abstraction level {abstraction_level}...")
        
        # Get cluster configuration at this level
        clusters_at_level = self.cluster_levels[abstraction_level]
        
        # Create mapping from event class to cluster label
        event_to_idx = {event: idx for idx, event in enumerate(self.event_classes)}
        event_to_cluster = {}
        
        for cluster_idx, cluster in enumerate(clusters_at_level):
            cluster_label = f"Cluster_{cluster_idx}"
            for event_idx in cluster:
                event_class = self.event_classes[event_idx]
                event_to_cluster[event_class] = cluster_label
        
        # Transform traces
        transformed_traces = []
        for trace in traces:
            transformed_trace = []
            prev_cluster = None
            
            for event in trace:
                cluster = event_to_cluster.get(event, event)
                
                # Only add if different from previous (collapse consecutive)
                if cluster != prev_cluster:
                    transformed_trace.append(cluster)
                    prev_cluster = cluster
            
            transformed_traces.append(transformed_trace)
        
        print(f"  ✓ Transformed {len(traces)} traces")
        return transformed_traces
    
    def get_cluster_info(self, abstraction_level: int) -> pd.DataFrame:
        """
        Get information about clusters at a specific abstraction level.
        
        Parameters:
        -----------
        abstraction_level : int
            The hierarchy level to inspect
            
        Returns:
        --------
        cluster_info : pd.DataFrame
            DataFrame with cluster information
        """
        if abstraction_level < 0 or abstraction_level >= len(self.cluster_levels):
            raise ValueError(f"Abstraction level must be between 0 and {len(self.cluster_levels)-1}")
        
        clusters_at_level = self.cluster_levels[abstraction_level]
        
        cluster_data = []
        for cluster_idx, cluster in enumerate(clusters_at_level):
            event_names = [self.event_classes[idx] for idx in cluster]
            cluster_data.append({
                'Cluster_ID': f"Cluster_{cluster_idx}",
                'Size': len(cluster),
                'Event_Classes': ', '.join(event_names)
            })
        
        return pd.DataFrame(cluster_data)
    
    def get_correlation_matrix(self) -> pd.DataFrame:
        """
        Get the event class correlation matrix as a DataFrame.
        
        Returns:
        --------
        correlation_df : pd.DataFrame
            Correlation matrix with event class labels
        """
        if self.correlation_matrix is None:
            raise ValueError("Model must be fitted first")
        
        return pd.DataFrame(
            self.correlation_matrix,
            index=self.event_classes,
            columns=self.event_classes
        )
    
    def plot_hierarchy_summary(self) -> pd.DataFrame:
        """
        Get a summary of the hierarchy structure.
        
        Returns:
        --------
        hierarchy_summary : pd.DataFrame
            Summary of each level in the hierarchy
        """
        summary_data = []
        for level, clusters in enumerate(self.cluster_levels):
            summary_data.append({
                'Level': level,
                'Number_of_Clusters': len(clusters),
                'Avg_Cluster_Size': np.mean([len(c) for c in clusters]),
                'Max_Cluster_Size': max([len(c) for c in clusters])
            })
        
        return pd.DataFrame(summary_data)


print("=" * 70)
print("GLOBAL TRACE SEGMENTATION - Implementation Complete")
print("=" * 70)
print("\nClass: GlobalTraceSegmentation")
print("\nKey Methods:")
print("  • fit(traces) - Fit the model on event log traces")
print("  • transform(traces, level) - Transform traces to abstraction level")
print("  • get_cluster_info(level) - Get cluster composition at a level")
print("  • get_correlation_matrix() - Get event class correlation matrix")
print("  • plot_hierarchy_summary() - Get hierarchy structure summary")
print("\n" + "=" * 70)
