import requests
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
import time
import os
import sys
import datetime

# HydraGNN imports for Graph Generation
from hydragnn.preprocess.graph_samples_checks_and_updates import (get_radius_graph, get_radius_graph_pbc)
from hydragnn.preprocess.graph_samples_checks_and_updates import (PBCDistance, PBCLocalCartesian, Distance, LocalCartesian)

# ---------------- CONFIGURATION ---------------- #
CSV_URL = "https://raw.githubusercontent.com/chenebuah/ML_abx3_dataset/main/oqmd_data.csv"
OQMD_URL = "http://oqmd.org/oqmdapi/formationenergy"
OUTPUT_FILE = "oqmd_dataset.pt"
BATCH_SIZE = 100 # Fetching OQMD batch size (don't go too large or server errors appear)
# ----------------------------------------------- #

# Class for building graphs from 
class GraphBuilder:
    def __init__(self):
        # Initialize Lookups
        self.ATOM_NUMS = {
            'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
            'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
            'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
            'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36,
            'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48,
            'In': 49, 'Sn': 50, 'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54,
            'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71,
            'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
            'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86,
            'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90, 'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100, 'Md': 101, 'No': 102, 'Lr': 103
        }

        # Initialize Graph Tools
        graphConfig = {"radius":5, "max_neighbours":20, "loop":False}
        self.RadiusGraph = get_radius_graph(**graphConfig)
        self.RadiusGraphPBC = get_radius_graph_pbc(**graphConfig)
        self.transform = Distance(norm=False, cat=False)
        self.transform_pbc = PBCDistance(norm=False, cat=False)

    def process_entry(self, entry_data, formation_energy):
        """
        Takes raw dictionary data and turns it into a PyG Data object
        """
        try:
            # Extract Lattice (3x3 matrix)
            lattice = np.array(entry_data['unit_cell'])
            sites = entry_data['sites']
            
            numAtoms = len(sites)
            coords = np.empty((numAtoms,3))
            atomic_numbers = np.empty((numAtoms,1))
            
            for i, site in enumerate(sites):
                # Parse: "Element @ x y z"
                parts = site.split()
                element_sym = parts[0]
                frac_coords = np.array([float(parts[2]), float(parts[3]), float(parts[4])])
                
                # Convert Fractional to Cartesian: Cart = Frac . Lattice
                cart_coords = np.dot(frac_coords, lattice)
                
                coords[i,:] = cart_coords
                atomic_numbers[i,0] = self.ATOM_NUMS.get(element_sym, 0)
            
            # Tensorize
            atomic_numbers = torch.from_numpy(atomic_numbers).float()
            coords = torch.from_numpy(coords).float()
            energy = torch.tensor([formation_energy]).float()
            
            # Feature vector
            x = torch.cat([atomic_numbers, coords], dim=1) 
            
            data = Data(
                x=x,
                y=energy,
                pos=coords,
                atomic_number=atomic_numbers,
                energy=energy
            )

            data.pbc = [True, True, True] 

            # Generate Edges
            try:
                data = self.RadiusGraphPBC(data)
                data = self.transform_pbc(data)
            except:
                data = self.RadiusGraph(data)
                data = self.transform(data)
                
            return data

        except Exception as e:
            print(f"Graph Build Error: {e}")
            return None

# Fetch data from website database. Waits if being rate limited.
def fetch_data(url, base_wait_time=5, max_attempts=5):   
    # Starting wait time for failing query
    wait_time = base_wait_time 
    for attempt in range(max_attempts):
        # Get query
        response = requests.get(url)
        # If it was good, return response
        if response.status_code == 200:
            return response.json()
        # If we are being rate-limited
        if response.status_code == 429:
            print(f"Rate limited! Waiting {wait_time} seconds...")
            time.sleep(wait_time)
            wait_time *= 2  # Double the wait time for the next attempt
        else:
            print(f"Error {response.status_code} for {url}")
            break
    print(f"Max Attempts Reached for {url}")
    return None

# For better formatting
def format_time(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))

def main():
    # Load targets
    print("--- Loading Targets ---")
    df = pd.read_csv(CSV_URL)
    
    # Create a set for lookup (using the Ef [formation energy] from the github csv)
    target_map = pd.Series(df.Ef.values, index=df.entry_id).to_dict()
    total_targets = len(target_map)
    print(f"Looking for {total_targets} specific IDs.")

    # Make graph builder class
    builder = GraphBuilder()

    # Create OQMD lookup params (set specific)
    # Jamie::TODO::Create script to figure out "ideal" lookups if this is found to be generally useful
    params = {
        'filter': 'generic=ABC3',
        'fields': 'entry_id,sites,unit_cell',
        'limit': BATCH_SIZE,
    }

    # Get first url from params
    url = requests.Request('GET', OQMD_URL, params=params).prepare().url

    # Loop until no more data OR all items found
    num_matched, num_scanned, page_num = 0,0,0
    data_list = []
    ts = time.time()
    while url:
        # Get the data
        data_json = fetch_data(url, base_wait_time=1, max_attempts=6)
        items = data_json['data']
        
        # If no items returned, we are done
        if not items: break
        
        # Find matches
        batch_matches = [item for item in items if item['entry_id'] in target_map]
        
        # Search next page
        links = data_json.get('links', {})
        next_link = links.get('next')

        # Move to next page
        url = next_link
        
        # Update counters
        num_scanned += len(items)
        page_num += 1

        # If we have batch_matchesFor all matches
        if batch_matches:
            num_matched += len(batch_matches)
            for item in batch_matches:
                # Get Target Energy from our CSV map
                target_ef = target_map[item['entry_id']]
                
                # Build Graph
                graph = builder.process_entry(item, target_ef)
                
                # Add to list
                if graph is not None: data_list.append(graph)         

        # Progress Update (Overwrite line for clean output)
        time_elapsed = time.time() - ts
        frac_done = num_matched/total_targets
        etf = time_elapsed*(1/frac_done-1)
        sys.stdout.write(f"\rPage {page_num} | Scanned: {num_scanned} | Found: {num_matched}/{total_targets} | Time Elapsed(s): {format_time(time_elapsed)} | Time remaining(s): {format_time(etf)}")
        sys.stdout.flush()

        # Break if we have found everything
        if (total_targets == num_matched): break

    # Save the list to disk
    print()
    print(f"Saving {len(data_list)} samples to {OUTPUT_FILE}...")
    torch.save(data_list, OUTPUT_FILE)
    print("Done.")

if __name__ == "__main__":
    main()