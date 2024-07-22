import os
import pandas as pd



# in a given directory, read csv one by one and print rows one by one
def read_csv_files(directory):
    df = pd.DataFrame()
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            print(filename)
            df = pd.concat([df, pd.read_csv(directory + filename)])
    system_prompt = "Extract the entities and relationships from the following text:"
    df['text'] = "<s>[INST] " + system_prompt + " </s>[SYS]" + " User prompt [/INST] " + df['input'] + " </s> Model answer <s>" + df['output'] + "</s>"
    df.to_csv("output_data/combined.csv", index=False)
    
read_csv_files("output_data/")