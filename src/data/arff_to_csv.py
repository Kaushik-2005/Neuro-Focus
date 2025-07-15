import pandas as pd

def convert_arff_to_csv():
    # Read the ARFF file
    data_lines = []
    columns = []
    
    with open('data/EEG Eye State.arff', 'r') as f:
        lines = f.readlines()
    
    # Process the file
    start_data = False
    for line in lines:
        line = line.strip()
        if not line or line.startswith('%'):  # Skip empty lines and comments
            continue
            
        if line.startswith('@ATTRIBUTE'):
            # Extract column name from attribute declaration
            col_name = line.split()[1]
            columns.append(col_name)
        elif line == '@DATA':
            start_data = True
        elif start_data:
            # Clean and append data lines
            data_lines.append(line)
    
    # Create DataFrame from the data
    df = pd.DataFrame([line.split(',') for line in data_lines], columns=columns)
    
    # Convert numeric columns to float
    for col in df.columns:
        if col != 'eyeDetection':  # Skip the target column
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Save as CSV
    df.to_csv('data/EEG_Eye_State.csv', index=False)
    print(f"Conversion completed: EEG_Eye_State.csv created with {len(df)} rows")

if __name__ == "__main__":
    convert_arff_to_csv() 