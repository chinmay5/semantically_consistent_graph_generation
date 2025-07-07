import glob
import io
import os
import shutil
from tqdm import tqdm

from bootstrap.vessap_setup.graph_cleanup_utils.convert_files import convert_vtp_graph_to_pyg_and_save_as_pt
from bootstrap.vessap_setup.graph_cleanup_utils.pre_process_all_vessels import remove_neg_rad_edges_for_all_graphs
from environment_setup import PROJECT_ROOT_DIR, time_logging


@time_logging
def preprocess_graph_to_remove_neg_edges(unprocessed_vessap_path, non_neg_rad_graph_loc):
    if os.path.exists(non_neg_rad_graph_loc):
        print("non-neg rad folder exists. Cleaning it and re-generating")
        shutil.rmtree(non_neg_rad_graph_loc)
        os.makedirs(non_neg_rad_graph_loc)
    for split in ['train_data', 'test_data']:
        # Make all the directories before processing
        os.makedirs(os.path.join(non_neg_rad_graph_loc, split), exist_ok=True)
    for split in ['train_data', 'test_data']:
        print(f"Processing {split=}")
        split_path = os.path.join(unprocessed_vessap_path, split, 'vtp')
        split_save_path = os.path.join(non_neg_rad_graph_loc, split)
        remove_neg_rad_edges_for_all_graphs(split_path=split_path, split_save_path=split_save_path,
                                            non_neg_rad_graph_loc=non_neg_rad_graph_loc, split=split)


@time_logging
def create_dataset(non_neg_rad_graph_loc, make_undirected):
    construct_graph_path = os.path.join(PROJECT_ROOT_DIR, 'data', 'vessel', f'pt_files')
    if os.path.exists(construct_graph_path):
        print("Existing data folder found. Cleaning it now.")
        shutil.rmtree(construct_graph_path)
    os.makedirs(construct_graph_path)
    # Create the string buffer object
    buffer = io.StringIO()
    idx = 0

    print("Generating undirected graphs. This is done by using pyG utility function")
    print("--------------")
    print(f"{make_undirected=}")
    print("--------------")
    try:
        # We would be processing only the training split. The test split is untouched.
        for filename in tqdm(glob.glob(f"{non_neg_rad_graph_loc}/train_data/*.vtp")):
            convert_vtp_graph_to_pyg_and_save_as_pt(filename=filename, idx=idx,
                                                    buffer=buffer, save_path=construct_graph_path,
                                                    make_undirected=make_undirected,
                                                    )
            idx += 1
    finally:
        final_content = buffer.getvalue()
        buffer.close()
        # Write the final content to a text file
        with open(os.path.join(non_neg_rad_graph_loc, f"updated_graph_changelog.txt"), "w") as file:
            file.write(final_content)


@time_logging
def generate_data(unprocessed_vessap_path, non_neg_rad_graph_loc, make_undirected):
    preprocess_graph_to_remove_neg_edges(unprocessed_vessap_path=unprocessed_vessap_path,
                                         non_neg_rad_graph_loc=non_neg_rad_graph_loc)
    create_dataset(non_neg_rad_graph_loc=non_neg_rad_graph_loc,
                   make_undirected=make_undirected)


if __name__ == '__main__':
    unprocessed_vessap_graph_path = '/mnt/elephant/chinmay/vessap_new/'
    non_neg_rad_graph_loc = '/mnt/elephant/chinmay/construct_vessap/non_neg_rad_graph'
    if os.path.exists(non_neg_rad_graph_loc):
        print("Cleaning existing destination path")
        folder_contents = os.listdir(non_neg_rad_graph_loc)
        # Retaining the changelog files since they might come in handy
        candidates = [x for x in folder_contents if 'changelog' not in x]
        for x in candidates:
            shutil.rmtree(os.path.join(non_neg_rad_graph_loc, x))
    generate_data(unprocessed_vessap_path=unprocessed_vessap_graph_path,
                  non_neg_rad_graph_loc=non_neg_rad_graph_loc,
                  make_undirected=True)
