import numpy as np

def extract_sleep_stage_segment(sleep_data, sleep_tag, stage_label, segment_number=0, 
                               segment_length=-1, skip_first_epoch=False, fs=128):
    """
    Extract time series corresponding to a specific sleep stage occurrence.
    
    Parameters
    ----------
    sleep_data : np.ndarray, shape (n_channels, n_samples)
        Multi-channel sleep time series data
    sleep_tag : pd.DataFrame
        DataFrame with column 'L' containing sleep stage labels
    stage_label : str
        Sleep stage to extract ('W', 'N1', 'N2', 'N3', 'R', 'L')
    segment_number : int, default=0
        Which occurrence of the stage to extract (0 = first, 1 = second, etc.)
    segment_length : int, default=-1
        Length of the segment to extract in epochs (30 seconds each). 
        If -1, extract the entire segment.
        If positive, extract only the specified number of epochs.
    skip_first_epoch : bool, default=False
        If True, skip the first epoch and start from the second.
        Useful to avoid transition artifacts at stage boundaries.
        If segment_length is specified, extracts epochs 2 to (segment_length+1).
        If not enough epochs are available, falls back to including the first epoch.
    fs : int, default=128
        Sampling frequency in Hz
        
    Returns
    -------
    segment_data : np.ndarray or None
        Time series of the requested segment, shape (n_channels, n_samples_segment)
        Returns None if segment not found
    start_idx : int or None
        Starting sample index of the segment
    end_idx : int or None
        Ending sample index of the segment (exclusive)
    info : dict
        Dictionary with segment information (duration, etc.)
        
    Examples
    --------
    >>> # Extract first N1 segment (complete)
    >>> n1_data, start, end, info = extract_sleep_stage_segment(sleep_data, sleep_tag, 'N1', segment_number=0)
    
    >>> # Extract first 5 epochs (2.5 minutes) of third N2 segment
    >>> n2_data, start, end, info = extract_sleep_stage_segment(sleep_data, sleep_tag, 'N2', segment_number=2, segment_length=5)
    
    >>> # Extract 3 epochs from first N1 segment, skipping first epoch (extracts epochs 2, 3, 4)
    >>> n1_data, start, end, info = extract_sleep_stage_segment(sleep_data, sleep_tag, 'N1', segment_number=0, 
    ...                                                          segment_length=3, skip_first_epoch=True)
    """
    
    # Time parameters
    samples_per_epoch = fs * 30  # 3840 samples per 30-second epoch
    N = sleep_data.shape[0]
    
    # Expand labels to match time series resolution
    stages = sleep_tag['L'].values
    expanded_labels = np.repeat(stages, samples_per_epoch)
    n_samples = min(len(expanded_labels), sleep_data.shape[1])
    expanded_labels = expanded_labels[:n_samples]
    
    # Find all indices with the requested stage
    stage_indices = np.where(expanded_labels == stage_label)[0]
    
    if len(stage_indices) == 0:
        print(f"No samples found for stage '{stage_label}'!")
        return None, None, None, None
    
    # Find consecutive segments
    # A new segment starts when the difference between consecutive indices > 1
    breaks = np.where(np.diff(stage_indices) > 1)[0] + 1
    
    # Split indices into segments
    if len(breaks) > 0:
        segments = np.split(stage_indices, breaks)
    else:
        segments = [stage_indices]
    
    # Check if requested segment exists
    n_segments = len(segments)
    if segment_number >= n_segments:
        print(f"Segment {segment_number} not found! Stage '{stage_label}' has only {n_segments} segment(s).")
        print(f"Valid segment_number values: 0 to {n_segments - 1}")
        return None, None, None, None
    
    # Extract the requested segment
    segment_indices = segments[segment_number]
    segment_start_idx = segment_indices[0]
    segment_end_idx = segment_indices[-1] + 1  # Full segment end
    
    # Calculate available epochs in this segment
    full_n_samples = segment_end_idx - segment_start_idx
    full_n_epochs = full_n_samples // samples_per_epoch
    
    # Determine extraction parameters based on skip_first_epoch and segment_length
    first_epoch_actually_skipped = False
    
    if skip_first_epoch:
        # Check if we can actually skip the first epoch
        if segment_length > 0:
            # Need segment_length + 1 epochs total to skip first and extract segment_length
            epochs_needed = segment_length + 1
            can_skip = full_n_epochs >= epochs_needed
        else:
            # Need at least 2 epochs to skip first
            can_skip = full_n_epochs >= 2
        
        if can_skip:
            # Can skip first epoch as requested
            start_idx = segment_start_idx + samples_per_epoch
            first_epoch_actually_skipped = True
            
            if segment_length > 0:
                end_idx = start_idx + (segment_length * samples_per_epoch)
                n_epochs_extracted = segment_length
                epochs_range = f"epochs 2-{segment_length + 1}"
            else:
                # Extract all remaining epochs after skipping first
                end_idx = segment_end_idx
                n_epochs_extracted = full_n_epochs - 1
                epochs_range = f"epochs 2-{full_n_epochs}"
        else:
            # FALLBACK: Not enough epochs to skip, include first epoch
            print(f"  Warning: Requested skip_first_epoch=True, but segment has only {full_n_epochs} epoch(s).")
            print(f"           Falling back to including first epoch.")
            
            start_idx = segment_start_idx
            
            if segment_length > 0:
                if segment_length > full_n_epochs:
                    # Extract all available
                    end_idx = segment_end_idx
                    n_epochs_extracted = full_n_epochs
                else:
                    end_idx = start_idx + (segment_length * samples_per_epoch)
                    n_epochs_extracted = segment_length
                epochs_range = f"epochs 1-{n_epochs_extracted}"
            else:
                # Extract entire segment
                end_idx = segment_end_idx
                n_epochs_extracted = full_n_epochs
                epochs_range = f"epochs 1-{full_n_epochs}"
    
    else:
        # Start from first epoch (no skip)
        start_idx = segment_start_idx
        
        if segment_length > 0:
            if segment_length > full_n_epochs:
                print(f"  Warning: Requested {segment_length} epochs, but segment only has {full_n_epochs} epochs.")
                print(f"           Extracting all {full_n_epochs} available epochs.")
                end_idx = segment_end_idx
                n_epochs_extracted = full_n_epochs
            else:
                end_idx = start_idx + (segment_length * samples_per_epoch)
                n_epochs_extracted = segment_length
            epochs_range = f"epochs 1-{n_epochs_extracted}"
        else:
            # Extract entire segment
            end_idx = segment_end_idx
            n_epochs_extracted = full_n_epochs
            epochs_range = f"epochs 1-{full_n_epochs}"
    
    # Extract the data
    segment_data = sleep_data[:, start_idx:end_idx]
    
    # Calculate segment information
    n_samples_segment = end_idx - start_idx
    duration_seconds = n_samples_segment / fs
    duration_minutes = duration_seconds / 60
    
    info = {
        'stage': stage_label,
        'segment_number': segment_number,
        'total_segments': n_segments,
        'start_idx': start_idx,
        'end_idx': end_idx,
        'n_samples': n_samples_segment,
        'n_epochs': n_epochs_extracted,
        'full_segment_epochs': full_n_epochs,
        'duration_seconds': duration_seconds,
        'duration_minutes': duration_minutes,
        'n_channels': N,
        'sampling_frequency': fs,
        'truncated': segment_length > 0 and segment_length < full_n_epochs,
        'first_epoch_skipped': first_epoch_actually_skipped,
        'skip_requested': skip_first_epoch,
        'epochs_range': epochs_range
    }
    
    # Print information
    print(f"Stage '{stage_label}' - Segment {segment_number + 1}/{n_segments}:")
    print(f"  Full segment epochs: {full_n_epochs}")
    print(f"  Extracted: {epochs_range} ({n_epochs_extracted} epochs)")
    if skip_first_epoch:
        if first_epoch_actually_skipped:
            print(f"  First epoch skipped: Yes ✓")
        else:
            print(f"  First epoch skipped: No (fallback - not enough epochs)")
    print(f"  Start index: {start_idx}")
    print(f"  End index: {end_idx}")
    print(f"  Number of samples: {n_samples_segment}")
    print(f"  Duration: {duration_seconds:.2f} seconds = {duration_minutes:.2f} minutes")
    print(f"  Shape of extracted time series: {segment_data.shape}")
    
    return segment_data, start_idx, end_idx, info



# Helper function to list all available segments
def list_sleep_stage_segments(sleep_data, sleep_tag, stage_label, fs=128):
    """
    List all consecutive segments of a given sleep stage.
    
    Parameters
    ----------
    sleep_data : np.ndarray
        Sleep time series data
    sleep_tag : pd.DataFrame
        Sleep stage labels
    stage_label : str
        Sleep stage to analyze
    fs : int, default=128
        Sampling frequency
        
    Returns
    -------
    segments_info : list of dict
        List of dictionaries containing info about each segment
    """
    
    samples_per_epoch = fs * 30
    stages = sleep_tag['L'].values
    expanded_labels = np.repeat(stages, samples_per_epoch)
    n_samples = min(len(expanded_labels), sleep_data.shape[1])
    expanded_labels = expanded_labels[:n_samples]
    
    stage_indices = np.where(expanded_labels == stage_label)[0]
    
    if len(stage_indices) == 0:
        print(f"No samples found for stage '{stage_label}'!")
        return []
    
    breaks = np.where(np.diff(stage_indices) > 1)[0] + 1
    segments = np.split(stage_indices, breaks) if len(breaks) > 0 else [stage_indices]
    
    segments_info = []
    print(f"\nStage '{stage_label}' has {len(segments)} consecutive segment(s):\n")
    
    for i, seg_indices in enumerate(segments):
        start_idx = seg_indices[0]
        end_idx = seg_indices[-1] + 1
        n_samples_seg = end_idx - start_idx
        duration_sec = n_samples_seg / fs
        duration_min = duration_sec / 60
        
        info = {
            'segment_number': i,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'n_samples': n_samples_seg,
            'duration_seconds': duration_sec,
            'duration_minutes': duration_min,
            'start_time_minutes': start_idx / fs / 60
        }
        
        segments_info.append(info)
        
        print(f"  Segment {i}:")
        print(f"    Time: {info['start_time_minutes']:.2f} - {(end_idx/fs/60):.2f} minutes")
        print(f"    Indices: {start_idx} - {end_idx}")
        print(f"    Duration: {duration_sec:.2f} s ({duration_min:.2f} min)")
        print(f"    Samples: {n_samples_seg}\n")
    
    return segments_info