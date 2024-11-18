import pandas as pd

from utils import list_files_in_folder

folder_path = "/Users/kfirraiby/Desktop/git/chemistry_QA/outputs/post_summery_sec_round_gpt_4o_v2"
file_paths = list_files_in_folder(folder_path, file_type='csv')
print(file_paths)


def merge_csvs(file_paths):
    # Initialize an empty list to store DataFrames
    dfs = []

    # Iterate through the file paths
    for i, file_path in enumerate(file_paths, start=1):
        print(i, file_path)
        # Read the CSV file into a DataFrame, set the index, and rename the column
        df = pd.read_csv(folder_path + '/' + file_path).set_index('Keys').rename(columns={'Values': file_path.rstrip('_output_post_summery_sec_round_gpt_4o_v2.csv')})
        dfs.append(df)

    # Merge DataFrames on their indices
    merged_df = pd.concat(dfs, axis=1, join='inner')

    return merged_df


# Example usage with a list of file paths


result_df = merge_csvs(file_paths)
print(result_df.shape)
result_df.to_excel("/Users/kfirraiby/Desktop/git/chemistry_QA/final_database/second_round_gpt_4o_v2.xlsx", index=True)



