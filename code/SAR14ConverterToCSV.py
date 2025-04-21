"""
Converts .txt file with the SAR14 dataset to .csv file and replaces rating with the sentiment label
either 'positive' or 'negative'.
"""



import pandas as pd
import json
import re



with open('config.json', 'r') as f:
    config = json.load(f)

path_to_sar14_dataset_txt = config['DatasetsPaths']['path_to_sar14_dataset_txt']
path_to_sar14_dataset_csv= config['DatasetsPaths']['path_to_sar14_dataset_csv']
SAR14_data = []



# Regular expression to distinguish the review from the band
# It has the following form: "" + space + , + band
pattern = r'^"(.+?)"\s*,\s*(\d+)\s*$'  

with open(path_to_sar14_dataset_txt, "r", encoding="utf-8") as file:
    lines = file.readlines()  # Read file as a list of lines.

for line in lines:
    line = line.strip() #Removing any excessive spaces
    match = re.match(pattern, line)  
    
    if match:
        review = match.group(1)  # Review without quotes
        rating = int(match.group(2))  # Band
        SAR14_data.append((review, rating))
    else:
        print(f"There is no match in line: {line}")  

SAR14_csv = pd.DataFrame(SAR14_data, columns=["review", "rating"])

# Adding labels for reviews
def sentiment_label(rating: int) -> str:
    """
    Assigns labels to reviews according to its rating.

    Args:
        rating (int): Review's rating.

    Returns:
        str: A label assigned to the certain review.
    """
    return "positive" if rating >= 7 else "negative"

# Saving data to .csv file
SAR14_csv["sentiment"] = SAR14_csv["rating"].apply(sentiment_label)
del SAR14_csv["rating"]
SAR14_csv.to_csv(path_to_sar14_dataset_csv, index=False, encoding="utf-8")
