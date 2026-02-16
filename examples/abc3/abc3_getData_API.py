import pandas as pd
import torch
from torch_geometric.data import Data
from mp_api.client import MPRester  # pip install "mp-api == 0.41.2"
import os
import numpy as np

# HydraGNN imports for Graph Generation
from hydragnn.preprocess.graph_samples_checks_and_updates import (get_radius_graph, get_radius_graph_pbc)
from hydragnn.preprocess.graph_samples_checks_and_updates import (PBCDistance, PBCLocalCartesian, Distance, LocalCartesian)

# API Key
from api_keys import MP_API_KEY
# ---------------- CONFIGURATION ---------------- #
# Get your key here: https://next-gen.materialsproject.org/api
API_KEY = MP_API_KEY
OUTPUT_FILE = "abc3_dataset.pt"
# ----------------------------------------------- #

# Get csv from github
def get_csv_from_url():
    url = "https://raw.githubusercontent.com/chenebuah/ML_abx3_dataset/main/abc3_data.csv"
    print(f"Reading csv from: {url}...")
    df = pd.read_csv(url)
    return df

# Create HydraGNN dataset
def create_dataset_from_csv(df):
    # Get csv
    df = get_csv_from_url()
    
    # Clean column names (strip whitespace)
    df.columns = df.columns.str.strip()
    
    # Map mp_id to target property (formation energy)
    # Note: Trust dataset (csv) values
    id_to_energy = dict(zip(df['mp_id'], df['formation_energy (eV/atom)']))
    target_ids = list(df['mp_id'].unique())
    print(f"Found {len(target_ids)} unique materials. Fetching structures from MP API...")
   
    # Fetch Structures from Materials Project
    with MPRester(API_KEY) as mpr:
        # We fetch only the structure to save bandwidth
        docs = mpr.materials.summary.search(
            material_ids=target_ids, 
            fields=["material_id", "structure"]
        )
    print(f"Successfully fetched {len(docs)} structures.")

    # Make the graph objects 
    graphConfig = {"radius":5, "max_neighbours":20, "loop":False}
    RadiusGraph = get_radius_graph(**graphConfig)
    RadiusGraphPBC = get_radius_graph_pbc(**graphConfig)

    # Make the transform objects
    transform_coordinates = Distance(norm=False, cat=False)
    transform_coordinates_pbc = PBCDistance(norm=False, cat=False)

    # Convert to PyTorch Geometric Data objects
    data_list = []
    for doc in docs:
        mp_id = doc.material_id
        structure = doc.structure # This is a Pymatgen Structure object
        
        # Get the target label from your CSV
        if mp_id not in id_to_energy:
            continue      
        
        y_value = id_to_energy[mp_id]
        # Skip nans
        if (np.isnan(y_value)):
            print("Nan found, skipping")
            continue

        # Extract Atomic Numbers (Z)
        # atomic_numbers will be shape [num_atoms]
        atomic_numbers = torch.tensor(
            [site.specie.Z for site in structure], dtype=torch.long
        )

        # Extract Cartesian Coordinates (Positions)
        # coords will be shape [num_atoms, 3]
        coords = torch.tensor(
            structure.cart_coords, dtype=torch.float
        )

        # Make input vector
        atom_numbers = torch.tensor(atomic_numbers.view(-1, 1).float()) # Reshape to [num_atoms, 1]
        x = torch.cat([atom_numbers, coords], dim=1)
        y = torch.tensor([y_value])
             
        # Make object
        data = Data(
            x=x,
            y=y,
            pos=coords,
            atomic_number=atom_numbers,
            energy=y
        )

        # Add periodic condition
        data.pbc = [True, True, True] 

        # Turn positions -> Graph  
        try:
            data = RadiusGraphPBC(data)
            data = transform_coordinates_pbc(data)
        except:
            data = RadiusGraph(data)
            data = transform_coordinates(data)

        data_list.append(data)

    # Save the list to disk
    print(f"Saving {len(data_list)} samples to {OUTPUT_FILE}...")
    torch.save(data_list, OUTPUT_FILE)
    print("Done.")

if __name__ == "__main__":
    # Fail if file doesn't exist
    create_dataset_from_csv()