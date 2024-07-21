import sys
import pandas as pd
import pickle

try:
    # Load the pre-trained model, label encoders, feature names, and eventid to eventtemplate mapping
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('label_encoders.pkl', 'rb') as le_file:
        label_encoders = pickle.load(le_file)
    with open('feature_names.pkl', 'rb') as fn_file:
        feature_names = pickle.load(fn_file)
    with open('eventid_to_eventtemplate.pkl', 'rb') as et_file:
        eventid_to_eventtemplate = pickle.load(et_file)

    # Read the log file
    log_file_path = sys.argv[1]
    feedback = sys.argv[2]
    additional_comments = sys.argv[3]

    with open(log_file_path, 'r') as file:
        log_lines = file.readlines()

    # Preprocess the input data
    log_entries = []
    for line in log_lines:
        columns = line.strip().split(',')
        # Skip header row
        if columns[0].lower() == 'lineid':
            continue
        if len(columns) == 11:
            lineId, month, date, time, user, component, pid, address, content, eventId, eventTemplate = columns
            log_entries.append({
                'lineid': int(lineId),
                'month': month,
                'date': int(date),
                'time': time,
                'user': user,
                'component': component,
                'pid': int(pid),
                'address': address,
                'content': content,
                'eventid': eventId,
                'eventtemplate': eventTemplate,
                'feedback': feedback,
                'additionalcomments': additional_comments
            })

    input_df = pd.DataFrame(log_entries)

    # Convert column names to lowercase
    input_df.columns = input_df.columns.str.lower()

    # Keep only relevant columns for prediction
    input_df = input_df[['month', 'user', 'component', 'content', 'eventtemplate']]

    # Encode categorical variables, handle unseen labels by ignoring them
    for column in input_df.columns:
        if column in label_encoders:
            le = label_encoders[column]
            known_classes = le.classes_
            input_df[column] = input_df[column].apply(lambda x: le.transform([x])[0] if x in known_classes else -1)

    # Ensure the input features match the model's feature names
    input_df = input_df.reindex(columns=feature_names, fill_value=0)

    # Predict the eventid for each log entry
    predictions = model.predict(input_df)

    # Map the predicted eventid to the corresponding eventtemplate
    descriptive_predictions = [eventid_to_eventtemplate.get(pred, "Unknown Event") for pred in predictions]

    # Print the descriptive predictions
    for prediction in descriptive_predictions:
        print(prediction)
except Exception as e:
    print(f"Error: {e}")


