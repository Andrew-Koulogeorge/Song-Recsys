import pandas as pd
import numpy as np
import glob
import json
import matplotlib.pyplot as plt

'''
Parse k the playlist json files and return the data in useful format 
'''
def parse_playlist_dataset(start_range=0, end_range=10):
    # Containers for our three DataFrames
    playlist_song_records = []   # For DF1: each row maps a pid to a track_uri
    track_records = {}           # For DF2: key = track_uri, value = track metadata
    playlist_records = []        # For DF3: playlist-level metadata

    # Use glob to get the JSON files and sort them.
    # Adjust the path if your files are in a different directory.
    file_pattern = "spotify_train_set/data/mpd.slice.*.json"
    files = sorted(glob.glob(file_pattern))[start_range:end_range]  # Process only the first 10 files
    print(f"Number of files being extracted: {len(files)}")
    print(files)
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # Each file has an "info" field and a "playlists" field.
            playlists = data.get("playlists", [])
            
            for playlist in playlists:
                pid = playlist.get("pid")
                
                # Record playlist-level metadata (DF3)
                playlist_record = {
                    "pid": pid,
                    "playlist_name": playlist.get("name"),
                    "description": playlist.get("description"),  # may be None if not provided
                    "modified_at": playlist.get("modified_at"),
                    "num_artists": playlist.get("num_artists"),
                    "num_albums": playlist.get("num_albums"),
                    "num_tracks": playlist.get("num_tracks"),
                    "num_followers": playlist.get("num_followers"),
                    "num_edits": playlist.get("num_edits"),
                    "duration_ms": playlist.get("duration_ms"),
                    "collaborative": playlist.get("collaborative")
                }
                playlist_records.append(playlist_record)
                
                # Process each track in the playlist (DF1 and DF2)
                tracks = playlist.get("tracks", [])
                for track in tracks:
                    track_uri = track.get("track_uri")
                    
                    # DF1: Map pid to track_uri (include position if needed)
                    playlist_song_records.append({
                        "pid": pid,
                        "track_uri": track_uri,
                        "pos": track.get("pos")
                    })
                    
                    # DF2: Record track metadata only once per unique track_uri
                    if track_uri not in track_records:
                        track_records[track_uri] = {
                            "track_uri": track_uri,
                            "track_name": track.get("track_name"),
                            "artist_name": track.get("artist_name"),
                            "artist_uri": track.get("artist_uri"),
                            "album_name": track.get("album_name"),
                            "album_uri": track.get("album_uri"),
                            "duration_ms": track.get("duration_ms")
                        }
    # Create the DataFrames
                    
    # DF1: pid -> track_uri mapping
    df_playlist_song = pd.DataFrame(playlist_song_records)

    # DF2: track_uri -> track metadata
    df_tracks = pd.DataFrame(list(track_records.values()))

    # DF3: pid -> playlist metadata
    df_playlists = pd.DataFrame(playlist_records)

    return (df_playlist_song, df_tracks, df_playlists)




def create_validation_data(start_range, end_range):
    df_playlist_song_val, _, _ = parse_playlist_dataset(start_range, end_range)
    playlist_to_tracks_val = df_playlist_song_val.groupby("pid")["track_uri"].apply(list).to_dict()
    for pid in playlist_to_tracks_val.keys():
        track_list = playlist_to_tracks_val[pid]
        seen_songs = track_list[:len(track_list)//2]
        holdout_songs = track_list[len(track_list)//2:]
        playlist_to_tracks_val[pid] = {"seen":seen_songs, "heldout":holdout_songs}
    return playlist_to_tracks_val



"""
Compute % of relevant tracks predicted 

NOTE: This metric can also be computed on an artist level, where a correct prediction is any song from the same artist. 
"""
def R_precision(predicted_songs: set[str], heldout_songs:set[str]) -> float:
    assert(len(predicted_songs) == len(heldout_songs))
    num_matches = len(predicted_songs.intersection(heldout_songs))
    return (num_matches/len(heldout_songs),num_matches)





''' Plotting Heler Functions '''

def plot_percentage_distribution(scores):
    """
    Create a histogram showing the distribution of percentage scores.
    
    Args:
        scores: A list or array of percentage values (0-100)
    """
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set bin edges from 0 to 100 with steps of 5
    bins = np.arange(0, 105, 5)  # 0, 5, 10, ..., 100
    
    # Create the histogram
    n, bins, patches = ax.hist(scores, bins=bins, alpha=0.7, 
                              color='skyblue', edgecolor='black')
    
    # Add a grid for easier reading
    ax.grid(axis='y', alpha=0.75, linestyle='--')
    
    # Add titles and labels
    ax.set_title('Distribution of R-Precision for Each Playlist', fontsize=15)
    ax.set_xlabel('Percentage Score', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    
    
    # Add some statistics as text
    stats_text = (f"Mean: {np.mean(scores):.1f}%\n"
                  f"Median: {np.median(scores):.1f}%\n"
                  f"Std Dev: {np.std(scores):.1f}%\n"
                  f"Min: {min(scores):.1f}%\n"
                  f"Max: {max(scores):.1f}%")
    
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def plot_clicks_needed_distribution(clicks):
    """
    Create a histogram showing the distribution of percentage scores.
    
    Args:
        scores: A list or array of percentage values (0-100)
    """
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set bin edges from 0 to 100 with steps of 5
    bins = np.arange(1, 52, 1)  # 0, 5, 10, ..., 100
    
    # Create the histogram
    n, bins, patches = ax.hist(clicks, bins=bins, alpha=0.7, 
                              color='skyblue', edgecolor='black')
    
    # Add a grid for easier reading
    ax.grid(axis='y', alpha=0.75, linestyle='--')
    
    # Add titles and labels
    ax.set_title('Number of clicks needed to generate relevant song', fontsize=15)
    ax.set_xlabel('Percentage Score', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    
    
    # Add some statistics as text
    stats_text = (f"Mean: {np.mean(clicks):.1f}\n"
                  f"Median: {np.median(clicks):.1f}\n"
                  f"Std Dev: {np.std(clicks):.1f}\n"
                  f"Min: {min(clicks):.1f}\n"
                  f"Max: {max(clicks):.1f}")
    
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()    