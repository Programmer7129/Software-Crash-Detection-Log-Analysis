# import pandas as pd

# # Load the dataset
# file_path = 'Mac_2k.log_structured.csv'
# data = pd.read_csv(file_path)

# # Prepare data for training
# def preprocess(data):
#     entries = []
#     for _, row in data.iterrows():
#         entry = f"""
#         <log_entry>
#         lineId: {row['LineId']}
#         month: {row['Month']}
#         date: {row['Date']}
#         time: {row['Time']}
#         user: {row['User']}
#         component: {row['Component']}
#         pid: {row['PID']}
#         address: {row['Address']}
#         content: {row['Content']}
#         eventId: {row['EventId']}
#         eventTemplate: {row['EventTemplate']}
#         </log_entry>
#         """
#         entries.append(entry)
#     return entries

# log_entries = preprocess(data)

# # Save the preprocessed data to a text file
# with open('log_entries.txt', 'w') as f:
#     for entry in log_entries:
#         f.write(entry + '\n')


import pandas as pd

# Load the dataset
file_path = 'Mac_2k.log_structured.csv'
data = pd.read_csv(file_path)

# Reduce dataset size for faster training
data = data.sample(frac=0.1, random_state=42)

# Prepare data for training
def preprocess(data):
    entries = []
    for _, row in data.iterrows():
        entry = f"""
        <log_entry>
        lineId: {row['LineId']}
        month: {row['Month']}
        date: {row['Date']}
        time: {row['Time']}
        user: {row['User']}
        component: {row['Component']}
        pid: {row['PID']}
        address: {row['Address']}
        content: {row['Content']}
        eventId: {row['EventId']}
        eventTemplate: {row['EventTemplate']}
        </log_entry>
        """
        entries.append(entry)
    return entries

log_entries = preprocess(data)

# Save the preprocessed data to a text file
with open('log_entries.txt', 'w') as f:
    for entry in log_entries:
        f.write(entry + '\n')
