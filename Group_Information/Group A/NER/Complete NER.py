# Set working directory to /Users/jonahgoh/Documents/DSS5105/Code/Complete NER 3 Apr
import os

# Set the working directory
os.chdir('/Users/jonahgoh/Documents/DSS5105/Code/Complete NER 3 Apr')

import pandas as pd
import numpy as np
from collections import Counter
import re
import unicodedata
import json
import datetime
import os
import tableExtraction

def get_numeric_table(df):
    """ 
    If has enough numeric data, return the df (remove * and unicode), otherwise return empty df
    """
    numeric_perc = df.iloc[:, 1:].map(cal_numeric_values).mean().mean()
    none_perc = df.isna().mean().mean()
    # print(f'numerc_perc:{numeric_perc}')
    if numeric_perc >= 0.3 and none_perc<=0.5:  # small threshold considering tables may not in good structure
        df = df.map(lambda x: re.sub(r"(?<!\n)[ \t\r]+(?!\n)", " ", 
                            unicodedata.normalize("NFKC", x.replace("*", "")))
            .replace("\u202f", "")  # remove the narrow no-break space
            .strip() if isinstance(x, str) else x)
        return df
    return pd.DataFrame() 

def get_dominant_mode(lst, threshold):
    if not lst:
        return None 
    counter = Counter(lst)
    most_common = counter.most_common()  # List of (element, frequency) pairs sorted by frequency
    
    return most_common[0][0] if most_common[0][1] > threshold else None

def cal_numeric_values(value):
    if value is None:
        return False
    value = str(value)
    exclude_chars = {',', '.', '-', '%', ' ', '\n'}
    # Check if the value contains digits
    if any(char.isdigit() for char in value):
        non_digit_count = sum(1 for char in value if not char.isdigit() and char not in exclude_chars)            
        if non_digit_count <= 3:
            return True
    return False

def detect_structured_columns_bottom(df):
    """
    Return empty df if no column detected
    otherwise, return df with cleaned columns (combine rows \separate rows\) and remove redundant bottom
    check repetition in column names
    """
    # check whether has detected columns
    cols = []
    df_1 = df.copy()
    if list(df.columns) == list(range(len(df.columns))):
        # return the first row which has appropriate amount of numeric data
        numeric_rows = df.iloc[:, 1:].map(cal_numeric_values).sum(axis=1)/df.shape[1]
        # print(numeric_rows)
        idx_col = numeric_rows[numeric_rows >= 0.3].index[0] if (numeric_rows >= 0.3).any() else None
        idx_bot =  numeric_rows[numeric_rows >= 0.3].index[-1] if (numeric_rows >= 0.3).any() else None
        print(f'idx_col: {idx_col}')
        # for columns
        if not idx_col:
            return pd.DataFrame()
        elif idx_col == 0:
            print('No column names, need to check other tables in the same page..')
            return df
        elif idx_col == 1:
            # check \n
            cols = df.iloc[0].astype(str).tolist()
            print(f'First row as columns.')
            idx_bot -= idx_col
        elif 1 < idx_col <= 4:
            cols_0 = [' '.join(df.iloc[:idx_col, c].dropna().astype(str)).strip() for c in range(df.shape[1])]
            # print(f'cols_0:{cols_0}')
            if len([c for c in cols_0 if len(c)>0])>=df.shape[1]-1:
                print(f'Combine 0-{idx_col} rows as columns.')
                cols = cols_0
                idx_bot -= idx_col
        else:
            # check whether rows above is deletable
            for i in range(3):
                df_up = df.iloc[:idx_col-i]
                if df_up.isna().sum().sum()/(df_up.shape[0]*df_up.shape[1])>=0.7:
                    print(f'Combine {idx_col-i}-{idx_col} rows as columns.')
                    cols = [' '.join(df.iloc[idx_col-i:idx_col, c].dropna().astype(str)).strip() for c in range(df.shape[1])]
                    idx_bot -= i-1
                    break
            if not cols:
                print('No column names, need to check other tables in the same page..')
                return df
            
        # for bottom
        end_idx = -1
        print(f'idx_bot:{idx_bot}')
        if 1 <= df.shape[0] - (idx_bot+1) <= 3:
            df_len = df.iloc[idx_bot+1:].map(lambda r: len(str(r)) if r else None)
            num = df_len.notna().sum().sum()
            if df.iloc[idx_bot+1:].isna().sum(axis=1).sum()>= df.shape[1]*(df.shape[0]-2) or df_len.sum(skipna=True).sum()/num>=40:
                end_idx = idx_bot
            
        if cols:
            if end_idx != -1:
                df_1 = df.iloc[idx_col:end_idx+1].reset_index(drop=True)
            else:
                df_1 = df.iloc[idx_col:].reset_index(drop=True)
        elif end_idx != -1:
            df_1 = df.iloc[:end_idx+1]
        else:
            df_1 = df.copy()
    
    else:
        # check repetition
        suffix_pattern = re.compile(r"\.\d+$")  # concern: confuse with date
        repeat_columns = [col for col in df.columns if suffix_pattern.search(str(col))]
        if repeat_columns:
            print('Repeat col names (info lost, cannot differentiate)!')
            return pd.DataFrame()
            
    # check and update cols
    if cols:
        cols = [re.sub(r'\s+', ' ', col) for col in cols]
        df_1.columns = cols
        
    return df_1

def detect_clean_index(df):
    """ 
    Background: 
    1. May have wrongly combined two rows together (one of it being overall statement): by captial letter
    2. Potentially wrongly combining two cols as index
    3. remove index which is too long and other values are empty
    Based on the clean index, may need to further combine columns (if two first rows only has index)
    """
    def insert_newline_before_uppercase(x):
        pattern_idx =  r'[a-z][A-Z][a-z]'
        match = re.search(pattern_idx, x)
        if match:
            # Insert '\n\n' before the uppercase letter
            return x[:match.start()+1] + '\n\n' + x[match.start()+1:]
        return x
    
    def separate_2_rows(df):
        # clean index: num in the end (superscript/subscript): as two rows maybe combined, they can in between letters
        pattern_num = r'(?<=[a-z])\d{1}(?=[A-Z])|(?<=[a-z])\d{1}(?=\s|$)'
        df.iloc[:, 0] = df.iloc[:, 0].str.replace(pattern_num, '', regex=True)
        # detect wrongly combining two rows: upper and lower letter, blank space
        pattern_idx =  r'[a-z][A-Z][a-z]'
        df_idx = df.iloc[:, 0].astype(str).apply(insert_newline_before_uppercase).tolist()
        # print(f'df_idx:\n{df_idx}')
        # detect wrongly combining two cols
        num_list = [df_idx[i].count('\n\n') for i in range(df.shape[0])]
        if df.shape[0] >= 4 and len([n for n in num_list if n==1]) >= df.shape[0]*0.7:
            df_split = df_idx.str.split('\n\n', expand=True, n=1)
            # check columns
            col_0 = df.columns[0]
            match = re.search(pattern_idx, col_0)
            if match:
                idx = match.start(1)
                col = [col_0[:idx], col_0[idx:]]
                cols = col + df.columns.tolist()[1:]
            else:
                cols = df.columns.tolist()[0] + [None] + df.columns[1:]
            df = pd.concat([df_split, df.iloc[:, 1:]], axis=1)
            df.columns = cols

        elif df.shape[0] >= 4 and len([n for n in num_list if n==1]) > 0:
            # separatae two rows: assume the latter one is the next row
            for i in range(df.shape[0]):
                split_text = df_idx[i].split('\n\n', 1)  # splits only on the first occurrence
                if len(split_text) > 1 and i+1 < df.shape[0]:
                    newline = split_text[1]
                    df_idx.insert(i+1, newline)
                    df.iloc[i, 0] = split_text[0]
                    df = pd.concat([df.iloc[:i+1], pd.DataFrame([[newline] + [None]*(df.shape[1]-1)], columns = df.columns), df.iloc[i+1:]], axis=0).reset_index(drop=True)
                    if i>0 and df.iloc[i-1, 1:].isna().sum() > df.shape[1]-2:
                        df.iloc[i+1, 1:] = df.iloc[i, 1:]
                        df.iloc[i, 1:] = [None]*(df.shape[1]-1)
                    continue
        df = df.reset_index(drop=True)
        
        return df
            
    def remove_remark_rows(df):
        # check long index with missing values in other columns 
        num_row = df.shape[0]
        # remove_idx = []
        for i in range(num_row):
            # print(df.loc[1, df.columns[0]])
            if len(str(df.loc[i, df.columns[0]]))> 40 and df.loc[i, df.columns[1:]].isna().sum()==df.shape[1]-1:
                df.drop(index=[i], inplace=True)
            elif (
                pd.notna(df.loc[i, df.columns[0]])  # Ensure it's not NaN
                and len(str(df.loc[i, df.columns[0]])) > 0  # Ensure it's not an empty string
                and str(df.loc[i, df.columns[0]])[0].islower()  # Convert to string before indexing
                and df.loc[i, df.columns[1:]].isna().sum() == df.shape[1] - 1
            ):
                df.drop(index=[i], inplace=True)
        df = df.reset_index(drop=True)
        return df
    
    def check_row_repeat(df):
        value_rows = [tuple(df.iloc[i, 1:]) for i in range(df.shape[0]) if df.iloc[i, 1:].isna().sum()<=1]
        value_idx = [i for i in range(df.shape[0]) if df.iloc[i, 1:].isna().sum()<=1]
        value_set = set(value_rows)
        denominator = len(value_rows) if len(value_rows)>0 else None
        if denominator is None:
            return pd.DataFrame()
        if len(value_set)/denominator<0.6:
            print('Repetative indexes!')
            for row in value_rows:
                idxes = [idx for idx, item in enumerate(value_rows) if item == row]
                if len(idxes)>1:
                    repeat_idx = value_idx[idxes[1]]
                    df = df.iloc[:repeat_idx]
                    # check if redundant rows are reserved
                    df = remove_remark_rows(df)
                    col_num = df.shape[0]
                    for i in range(1, 3):
                        df.loc[col_num-i, df.columns[1:]].isna().sum() == df.shape[1]
                        df.drop(index=[col_num-i], inplace=True)
                    break
                else:
                    i += 1
        return df   
    
    stable_flag = 0
    df_0 = df.copy()
    df_0 = df_0.replace(r'^\s*$', None, regex=True)
    while stable_flag == 0 and not df_0.empty:
        df_1 = separate_2_rows(df_0)
        df_2 = remove_remark_rows(df_1)
        df_3 = check_row_repeat(df_2)
        if df_3.shape[0]==df_0.shape[0] and df_3.shape[1] == df_0.shape[1]:
            stable_flag = 1
        elif df_3.empty:
            return pd.DataFrame()
        else:
            df_0 = df_3.copy()
    
    df = df_3.copy()
    df = remove_remark_rows(df)
    
    return df

def combine_rows(df):
    """ 
    key principle: check capital letters
    1. two rows all only has captial index: combine the one with lowercase start into the uppercase row
    2. if only has one col not emtpy: combine with the last row by default
    example: sheet10_102
    
    special: need while loop - sometimes needs more than 1-off combining
    """
    # column names iloc[0,0]
    df_0 = df.iloc[0].isna().astype(int)
    if df.columns[0] is None and df_0[0]==1 and df_0[1:].sum()==0:
        df.columns = [df.iloc[0,0]] + df.columns[1:].tolist()
        df.drop(index=[0], inplace=True)
        df = df.reset_index(drop=True)
        print('Update cols by including df.iloc[0,0]')
    
    # other rows
    df.reset_index(drop=True)
    df = df.astype(object).where(pd.notna(df), None)
    # print(f'na_row:{[i for i in range(df.shape[0]) if df.iloc[i].isna().sum()>=df.shape[1]-2]}')
    combine_up_rows = [
        i for i in range(1, df.shape[0])
        if df.iloc[i].isna().sum() >= df.shape[1] - 2
        and isinstance(df.iloc[i-1, 0], str) and df.iloc[i-1, 0][0].isupper()
        and (pd.isna(df.iloc[i, 0]) or (isinstance(df.iloc[i, 0], str) and df.iloc[i, 0][0].islower()))
    ]
    
    combine_down_rows = [
        i for i in range(0, df.shape[0]-1)
        if i not in combine_up_rows
        and (pd.isna(df.iloc[i+1, 0]) 
            or (isinstance(df.iloc[i+1, 0], str) and len(df.iloc[i+1, 0]) <= 5) 
            or (isinstance(df.iloc[i+1, 0], str) and df.iloc[i+1, 0][0].islower()))
        and df.iloc[i].isna().sum() >= df.shape[1] - 2
    ]
    # print(f'combine up rows:{combine_up_rows}')
    # print(f'combine down rows:{combine_down_rows}')
    df_1 = df.copy()
    # use index name instead of index (removing row may change the idnex but not the name)
    for i in range(df_1.shape[0]-1, -1, -1):  # Iterate in reverse to avoid index shift
        if i in combine_up_rows:
            if i > 0:  # Ensure there is a previous row to combine with
                df_1.loc[i-1] = df.loc[i-1].astype(str).where(df.loc[i-1].notna(), '') + ' ' + df.loc[i].astype(str).where(df.loc[i].notna(), '')
                df_1.drop(index=[i], inplace=True)
        elif i in combine_down_rows:
            if i+1 < df_1.shape[0]:  # Ensure there is a next row to combine with
                df_1.iloc[i+1] = df.loc[i].astype(str).where(df.loc[i].notna(), '') + ' ' + df.loc[i+1].astype(str).where(df.loc[i+1].notna(), '')
                df_1.drop(index=[i], inplace=True)
    
    df_1 = df_1.reset_index(drop=True)
    return df_1

def separate_tables_vertically(df, sheet_name, sheets_dict):
    """ 
    After setting cols and indexes
    
    Conditions:
	1. The current row should have a non-empty index and None (or NaN) values in all columns.
	2. The next row should either:
	    Have the first column non-empty and other columns may or may not be non-empty (i.e., not all None).
	    Or, should have the first column non-empty and other columns can also have values.
	3. The next-next row should not have all None values, meaning it should contain at least one non-empty value.
 
    Return: 
    updated sheets_dict
    """
    condition_indices = []
    if df.shape[0]<=4:
        return sheets_dict
    
    i = 2
    while i < len(df) - 2:  
        row = df.iloc[i]
        # condition 1: current row - index non-empty, rest None
        index_non_empty = pd.notna(df.index[i]) 
        other_cols_all_na = row.isna().sum() == len(row) 
        if index_non_empty and other_cols_all_na:
            # condition 2: next row
            next_first_col_non_empty = pd.notna(df.index[i+1])
            # condition 3: heck next-next row (should not have all None values)
            next_next_row = df.iloc[i + 2]
            next_next_row_non_empty = next_next_row.notna().sum() > df.shape[1]*0.3
            if (next_first_col_non_empty):
                if next_next_row_non_empty:
                    condition_indices.append(i)
                    i += 3
                else:
                    i+=1
            else:
                i+=1
        else:
            i += 1
    
    if not condition_indices:
        return sheets_dict
    # generate separate df
    del sheets_dict[sheet_name]
    print(f'condition_indices:{condition_indices}')
    condition_indices = [0] + condition_indices
    for idx in range(1, len(condition_indices)):
        start_idx = condition_indices[idx-1]  
        end_idx = condition_indices[idx]      
        separate_df = df.iloc[start_idx:end_idx]
        # separate_dfs.append(separate_df)
        new_key = str(idx-1) + "_" + sheet_name
        sheets_dict[new_key] = separate_df
        # print(f'sheet {sheet_name} add df:{separate_df.head(2)}')
    # handling the last segment (if the last row satisfies the condition)
    if condition_indices and condition_indices[-1] != len(df) - 1:
        last_segment = df.iloc[condition_indices[-1]:]
        # separate_dfs.append(last_segment)
        new_key = str(len(condition_indices))+ '-' + sheet_name
        sheets_dict[new_key] = last_segment

    return sheets_dict

def refer_set_cols_indexes(df, sheet_name, sheets_dict):
    """ 
    if the cols names or indexes are missing, then check it out in other tables in the same page (searching)
    if the dimension matches, then update and return the new df
    
    if no cols/index then just return empty dataframe (drop)
    """
    
    match = re.search(r'_(\d+)$', sheet_name)
    page_num = int(match.group(1))
    sheets = [key for key in sheets_dict.keys() if re.search(r'_(\d+)$', key) and re.search(r'_(\d+)$', key).group(1) == str(page_num) and key!=sheet_name]
    col_num = df.shape[1]
    
    # column matching
    if list(df.columns) == list(range(col_num)) or df.columns.isna().sum()>1 or any(col == 'nan' for col in df.columns[1:]):
        print('Missing columns names')
        keys_list = []
        if len(sheets)>0:
            # naive: if the length do not match then just delete
            keys_list = [key for key in sheets if sheets_dict[key].columns.tolist()!= list(range(sheets_dict[key].shape[1])) and sheets_dict[key].shape[1]==col_num]
            if keys_list:
                if len(keys_list)==1:
                    df.columns = sheets_dict[keys_list[0]].columns
                else:
                    keys_list_1 = keys_list + [sheet_name] 
                    # index in the sheets_dict
                    idx = list(sheets_dict.keys()).index(sheet_name)
                    idx_list = sorted([list(sheets_dict.keys()).index(k) for k in keys_list_1])
                    # index in the sorted key_idx_list
                    df_idx = idx_list.index(idx)
                    key = list(sheets_dict.keys())[idx_list[1]] if df_idx == 0 else list(sheets_dict.keys())[idx_list[df_idx-1]]
                    df.columns = sheets_dict[key].columns
                print('Match column names!')
        if len(keys_list)==0:
            print('No matching column names')
            return pd.DataFrame()
                
    # index matching and setting
    first_col_values = df.iloc[:, 0]
    numeric_ratio = first_col_values.apply(cal_numeric_values).mean()
    num_row = df.shape[0]
    if numeric_ratio >= 0.6:  
        print('Missing index')
        potential_keys = []
        if len(sheets)>0:
            for key in sheets:
                first_col_values_other = sheets_dict[key].iloc[1:, 0]
                non_null_ratio_other = first_col_values_other.notna().mean()
                numeric_ratio_other = first_col_values_other.apply(cal_numeric_values).mean()

                if non_null_ratio_other == 1 and numeric_ratio_other < 0.1 and sheets_dict[key].shape[0]==num_row:
                    potential_keys.append(key)

            if potential_keys:
                if len(potential_keys) == 1:
                    df.index = sheets_dict[potential_keys[0]].iloc[:, 0]
                else:
                    potential_keys.append(sheet_name)
                    idx = list(sheets_dict.keys()).index(sheet_name)
                    idx_list = sorted([list(sheets_dict.keys()).index(k) for k in potential_keys])
                    df_idx = idx_list.index(idx)
                    key = list(sheets_dict.keys())[idx_list[1]] if df_idx == 0 else list(sheets_dict.keys())[idx_list[df_idx - 1]]
                    df.index = sheets_dict[key].iloc[:, 0]
        if len(potential_keys)==0:
            print('Missing index names!')
            return pd.DataFrame()
    else:
        # print(df.iloc[:, 0])
        if df.columns[0] is not None and len(str(df.columns[0])) >= 3 and df.columns[0] != 'Unnamed: 0':
            df.iloc[:, 0] = str(df.columns[0]) + ' ' + df.iloc[:, 0]
        df.index = df.iloc[:, 0]
        df = df.iloc[:, 1:]
        # print(f'index:{df.index}')
    df = df[~pd.isna(df.index)]
    return df 
    

def get_most_recent_year_data(df):
    """
    Returns a DataFrame with the data for the most recent year while keeping the index.
    Ensures that only valid year columns (numeric, within reasonable year range) are considered.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame with years as column names.
    
    Returns:
    pd.DataFrame: A DataFrame with the most recent year's data.
    """
    current_year = datetime.datetime.now().year
    
    # identify valid year columns
    valid_year_columns = [
        pd.to_numeric(col, errors='coerce', downcast='integer') for col in df.columns
        if pd.to_numeric(col, errors='coerce', downcast='integer') is not np.nan
        and 1900 <= pd.to_numeric(col, errors='coerce', downcast='integer') <= current_year
    ]
    if not valid_year_columns:
        print("No valid year columns found in the DataFrame.")
        return df
    
    df.columns = pd.to_numeric(df.columns, errors='coerce')
    most_recent_year = max(valid_year_columns)
    idx = list(df.columns).index(most_recent_year)
    most_recent_data = df.iloc[:, idx] # reserve index

    return most_recent_data.to_frame(name=most_recent_year)


def combine_units_with_values(df):
    """
    Detects the unit column dynamically and appends the unit to each numeric value.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame where one column contains unit values.

    Returns:
    pd.DataFrame: Transformed DataFrame with unit values appended to numeric values.
    """
    # identify possible unit columns
    unit_col = ''
    # print(df.columns)
    for col in df.columns:
        # print(df.columns, col)
        cols = df.columns.tolist()
        if len(set(cols)) == len(cols) and isinstance(col, str) and "unit" in col.lower() and df[col].dtype==object and len(set(df[col]))<df.shape[0]*0.7:
            unit_col = col
            print(f'find unit col:{col}')
            break

    # merge unit with value columns
    if len(unit_col)>0:
        for col in df.columns:
            if col not in [df.columns[0], unit_col] and df[col].apply(cal_numeric_values).mean()>=0.3:  # exclude the first column (index/metric) and unit column
                df[col] = df[col].astype(str) + " " + df[unit_col]
        # Ddop the unit column
        df.drop(columns=[unit_col], inplace=True)

    return df

def convert_to_json(df):
    # df not empty
    # Combine the segment phrase
    def check_segment_phrase(df):
        i = 0
        phrase = []
        while i <= 2:
            if df.iloc[i].isna().sum() == df.shape[1] and not pd.isna(df.index[i]):
                phrase.append(str(df.index[i]))
                idx = i
                i += 1
            else:
                break
        if phrase:
            prefix = ' '.join(phrase)
            df = df.iloc[idx:]
            df.index = prefix + ' ' + df.index
        return df
    
    df = check_segment_phrase(df)
    df = combine_units_with_values(df)
    df = get_most_recent_year_data(df)
    
    # print(f'index:{df.index}')
    # print(f'cols:{df.columns}')
    structured_dict = {
        f"{index_name} - {col}": df.loc[index_name, col]
        for index_name in df.index
        for col in df.columns
        if not df.columns.duplicated().any() and not df.index.duplicated().any()
        and pd.notna(df.loc[index_name, col]) and bool(re.search(r'\d', str(df.loc[index_name, col]))) # remove those without numeric data ('\d+)
    }
    json_output = json.dumps(structured_dict, indent=4)
    
    return json_output

def tabledict_to_json(sheets_dict, file_name, to_excel=True, to_json=True, output_dir= ''):
    #  cleaning and formating
    sheets_dict_1 = sheets_dict.copy()
    for sheet_name, df in sheets_dict.items():
        print(f'Process sheet: {sheet_name}..')
        df = get_numeric_table(df)
        if df.empty:
            del sheets_dict_1[sheet_name]
            continue
        else:
            print('Get numeric table!')
        df = detect_structured_columns_bottom(df)
        if df.empty:
            del sheets_dict_1[sheet_name]
            continue
        else:
            print('Get structured columns!')
        df = detect_clean_index(df)
        if df.empty:
            del sheets_dict_1[sheet_name]
            continue
        else:
            print('Clean index!')
        df = combine_rows(df)
        print('Combined rows!')
        df = refer_set_cols_indexes(df, sheet_name, sheets_dict_1)
        if df.empty:
            del sheets_dict_1[sheet_name]
            continue
        else:
            print('Refer cols and indexes!')
        sheets_dict_1[sheet_name] = df
        print('Set index!')
        # print(df)
        sheets_dict_1 = separate_tables_vertically(df, sheet_name, sheets_dict_1)
        print('Update dict!')
        # print(f'update dict:\n{sheets_dict_1}')
    
    # print(sheets_dict_1)
        
    if len(output_dir)>0:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    else:
        output_dir = os.getcwd()
        
    # saving into json    
    if to_json:
        json_output_lst = [convert_to_json(df) for df in sheets_dict_1.values() if len(convert_to_json(df).strip())>5]
        file_path = os.path.join(output_dir, file_name + '.json')
        with open(file_path, 'w') as json_file:
            json.dump(json_output_lst, json_file, indent=4)
        print(f"JSON data has been saved to {file_path}")   
         
    if to_excel:
        output_file = os.path.join(output_dir, file_name + '_processed' + '.xlsx')
        with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
            for sheet_name, df in sheets_dict_1.items():
                # Save each dataframe to a separate sheet
                sheet_name_cut = sheet_name[-31:] if len(sheet_name)>31 else sheet_name
                if not df.empty:
                    df = combine_units_with_values(df)
                    df.to_excel(writer, sheet_name=sheet_name_cut, index=True)
                else:
                    continue
        print(f'Excel has been saved to {output_file}')
    return sheets_dict_1


def table_to_json_pipeline(pdf_path, output_dir, save_tables=True, to_excel=True, to_json=False):
    file_name_with_extension = os.path.basename(pdf_path)
    file_name = os.path.splitext(file_name_with_extension)[0]
    sheets_dict, df_dict_list = tableExtraction.detect_extract_tables(pdf_path, save_tables=save_tables)
    # print(df_dict_list)
    sheets_dict_1 = tabledict_to_json(sheets_dict, file_name, to_excel=to_excel, to_json=to_json, output_dir=output_dir)
    return sheets_dict_1
        

# order: numeric_table -> detect_structured_columns_bottom (empty) -> detect_clean_index (empty)-> combine_rows -> refer_set_cols_indexes  -> separate_tables_vertically (for json expressing) -> json
if __name__ == "__main__":
    # test
    file_path = "/Users/jonahgoh/Downloads/2023_longi-sustainability-report.pdf" 
    output_dir = './table'
    # sheets_dict = pd.read_excel(file_path, sheet_name=None)  
    # file_name_with_extension = os.path.basename(file_path)
    # file_name = os.path.splitext(file_name_with_extension)[0]
    table_to_json_pipeline(file_path, output_dir=output_dir, to_json=True)

import pandas as pd
import ollama
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the pre-trained sentence transformer model
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Read in the DataFrame
excel_file_path = "/Users/jonahgoh/Documents/DSS5105/Code/Complete NER 3 Apr/table/2023_longi-sustainability-report_processed.xlsx"

# Load the Excel file with openpyxl engine
xls = pd.ExcelFile(excel_file_path, engine='openpyxl')
sheet_names = xls.sheet_names

# Initialize an empty string to accumulate all the text from each sheet
df_text = ""

# Iterate over each sheet and append its content to the df_text string
for sheet_name in sheet_names:
    # Read the current sheet with openpyxl engine
    df = pd.read_excel(xls, sheet_name=sheet_name, engine='openpyxl')
    
    # Convert the current sheet to text (as a string) and append to df_text
    df_text += df.to_string(index=False) + "\n\n"  # Add extra newline for separation between sheets

# Define the keywords for Environmental ESG metrics
keywords_1 = [
    "GHG emissions", "Scope 1", "Scope 2", "Scope 3", 
    "water", "energy", "consumption", "withdrawal", 
    "renewable", "energy intensity", "water saving"
]

keywords_2 = [
    "Number of Employees", "Employee Gender",
    "Number of Hires",
    "Workplace Injuries",
    "Total Fatal Incident"
]

# Encode both sets of keywords using the model
keyword_embeddings_1 = model.encode(keywords_1)
keyword_embeddings_2 = model.encode(keywords_2)

# Flatten the DataFrame into a list of rows for comparison (based on the appended text)
rows = df_text.split("\n")

# Create an empty list to store the relevant rows
relevant_rows_1 = []
relevant_rows_2 = []

# For each row in the text, calculate cosine similarity with each keyword from both keyword sets
for idx, row in enumerate(rows):
    # Skip empty rows
    if not row.strip():
        continue
    
    # Encode the row text
    row_embedding = model.encode([row])  # Encode the row text
    
    # Calculate cosine similarity between the row and each set of keywords
    similarities_1 = cosine_similarity(row_embedding, keyword_embeddings_1)
    similarities_2 = cosine_similarity(row_embedding, keyword_embeddings_2)
    
    # If any keyword from keywords_1 has high similarity, append to relevant_rows_1
    if any(similarity > 0.4 for similarity in similarities_1[0]):
        relevant_rows_1.append(row)  # Append the row if it matches the criteria

    # If any keyword from keywords_2 has high similarity, append to relevant_rows_2
    if any(similarity > 0.4 for similarity in similarities_2[0]):
        relevant_rows_2.append(row)  # Append the row if it matches the criteria

# Convert the relevant rows into a single string (markdown format) for each category
relevant_text_1 = "\n".join(relevant_rows_1)
relevant_text_2 = "\n".join(relevant_rows_2)

# Create separate prompts for Ollama for each case
prompt_1 = f"""
You are an expert in environmental sustainability metrics and ESG (Environmental, Social, Governance) data. 
Below is a table of sustainability indicators for a company. Extract all relevant **Environmental ESG** metrics, 
such as GHG emissions, energy consumption, water withdrawal, water consumption, and other environmental indicators from the table below.

Please return only the relevant rows with environmental data, such as:
- Scope 1 GHG emissions
- Scope 2 GHG emissions
- Scope 3 GHG emissions
- Water usage and water saving data
- Energy consumption and energy consumption intensity
- Any other environmental performance metrics
- Renewable energy data

Give the latest value for each metric. Output one line per metric, no nesting.

{relevant_text_1}
"""

prompt_2 = f"""
Extract and return the below data:
- Number of Employees
- Number of New Hires
- Total Workplace Injuries
- Total Fatal Incident

Give the latest value for each metric. Output one line per metric, no nesting.

{relevant_text_2}
"""

# Use Ollama's API to get insights for the environmental data (keywords_1)
response_1 = ollama.chat(
    model="llama2",
    messages=[{"role": "user", "content": prompt_1}],
    options={"temperature": 0}
)

# Extract the response content from the assistant's message
assistant_response_1 = response_1['message']['content']

# Print the response for environmental data
print("Environmental ESG Data:\n", assistant_response_1)

# Use Ollama's API to get insights for the social data (keywords_2)
response_2 = ollama.chat(
    model="llama2",
    messages=[{"role": "user", "content": prompt_2}],
    options={"temperature": 0}
)

# Extract the response content from the assistant's message
assistant_response_2 = response_2['message']['content']

# Print the response for social data
print("Social ESG Data:\n", assistant_response_2)

import json
import re
import ollama
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import defaultdict
from flatten_json import flatten

# Function to check if a string contains a number
def contains_number(value):
    return bool(re.search(r'\d', str(value)))

def clean_duplicate_keys(response_text):
    """
    This function processes the response to handle duplicate keys by grouping them
    into a list.
    """
    # Find all keys and their corresponding values
    pattern = r'"([^"]+)":\s*"([^"]+)"'
    matches = re.findall(pattern, response_text)

    # Create a dictionary to store unique keys, with values as lists
    data = defaultdict(list)

    for key, value in matches:
        data[key].append(value)

    return dict(data)

def extract_json(response_text):
    """
    This function processes the raw JSON-like text, fixing common issues such as 
    duplicate keys and malformed structures.
    """
    try:
        # Clean the input text by handling duplicate keys
        cleaned_data = clean_duplicate_keys(response_text)

        # Attempt to convert the cleaned data into a JSON object (Python dictionary)
        extracted_data = json.loads(json.dumps(cleaned_data))  # Convert back to dict
        
        # Optionally flatten the nested dictionary (if any)
        flattened_data = flatten(extracted_data)

        return flattened_data
    except json.JSONDecodeError:
        print("LLaMA returned an invalid JSON. Raw output:", response_text)
        return {}

def flatten_json(nested_data):
    """
    Recursively flattens nested JSON structures by expanding sub-keys into separate keys.
    Example:
    Input: {"Energy Source Breakdown": {"Coal": "30%", "Natural Gas": "25%"}}
    Output: {"Energy from Coal": "30%", "Energy from Natural Gas": "25%"}
    """
    flattened_data = {}

    def _flatten(sub_dict, parent_key=""):
        for key, value in sub_dict.items():
            new_key = f"{parent_key} {key}".strip() if parent_key else key  # Ensure spacing
            if isinstance(value, dict):  # If value is a dictionary, recursively flatten it
                _flatten(value, new_key)
            else:
                flattened_data[new_key] = value

    _flatten(nested_data)
    return flattened_data

# model = "Muennighoff/SGPT-125M-weightedmean-nli-bitfit")
# Muennighoff/SGPT-125M-weightedmean-nli-bitfit
# Muennighoff/SGPT-1.3B-weightedmean-nli-bitfit
# Muennighoff/SGPT-5.8B-weightedmean-nli-bitfit

# model = "bert-base-nli-mean-tokens"

# BAAI/gte-base

def extract_and_match(input_text, custom_categories, threshold=0.5, model_name="sentence-transformers/all-mpnet-base-v2"):
    """
    Extracts ESG, and sustainability metrics from input text using LLaMA 2 and matches them to custom categories using a BERT-based model.
    
    Parameters:
        input_text (str): The input text containing financial and sustainability-related information.
        custom_categories (list): List of predefined categories for mapping extracted values.
        threshold (float): Cosine similarity threshold for category matching.
        model_name (str): Sentence Transformer model for embeddings (default is 'all-mpnet-base-v2').
    
    Returns:
        pd.DataFrame: A DataFrame with columns ['Category', 'Matched Value'].
    """

    # Step 1: Query LLaMA 2 for extracting financial & ESG metrics
    prompt = f"""Extract key ESG (Environmental, Social, Governance), and environmental sustainability metrics from the provided text and output only a **valid JSON object** with the following rules:
    Example input:
    "Here are the relevant environmental ESG metrics from the table:
        * Scope 1 GHG emissions: 50,000,000 tCO2e (latest value)  
        * Scope 2 GHG emissions: 25,000,000 tCO2e (latest value)  
        * Scope 3 GHG emissions: 30,000,000 tCO2e (latest value)  
        * Water usage and water saving data:  
            + Total water consumption: 6,000 kt (latest value)  
            + Reclaimed water reuse rate: 60% (latest value)  
            + Water conservation: 6,500 kt (latest value)  
        * Energy consumption and energy consumption intensity:  
            + Total energy consumption: 200 million tce (latest value)  
            + Energy consumption intensity: 15,000 tce/ (latest value)  
        * Renewable energy data:  
            + Renewable energy usage proportion: 40 billion kWh (latest value)  
            + Renewable energy usage: 45 billion kWh (latest value)  
            + Renewable energy intensity: 100 billion kWh/ (latest value)  
        
    Output (no nesting):
    {{
        "Scope 1 GHG emissions": "50,000,000 tCO2e",
        "Scope 2 GHG emissions": "25,000,000 tCO2e",
        "Scope 3 GHG emissions": "30,000,000 tCO2e",
        "Total water consumption": "6,000 kt",
        "Reclaimed water reuse rate": "60%",
        "Water conservation": "6,500 kt",
        "Total energy consumption": "200 million tce",
        "Energy consumption intensity": "15,000 tce/",
        "Renewable energy usage proportion": "40 billion kWh",
        "Renewable energy usage": "45 billion kWh",
        "Renewable energy intensity": "100 billion kWh/"
    }}

    Now, extract key financial, ESG metrics and environmental sustainability from the below sentence and output only a JSON object, No nesting. One key-value pair per line. Separate each GHG emission scope:
    "{input_text}"
    Output:
    """

    response = ollama.chat(
        model="llama2",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0}
    )

    response_text = response["message"]["content"].strip()

    # Extract only the JSON using regex
    extracted_data = extract_json(response_text)
    
    # Expand nested JSON if any
    extracted_data = flatten(extracted_data)

    print(extracted_data)

    # Step 2: Separate percentage-based and non-percentage values 
    percentage_values = {k: v for k, v in extracted_data.items() if v is not None and "%" in str(v) and contains_number(v)}
    non_percentage_values = {k: v for k, v in extracted_data.items() if v is not None and "%" not in str(v) and contains_number(v)}

    # Step 3: Classify percentage values only to categories with "%" and non-percentage values to the rest
    percentage_categories = [cat for cat in custom_categories if "%" in cat]
    non_percentage_categories = [cat for cat in custom_categories if "%" not in cat]

    # Initialize dictionary to store mapped values
    final_mapped_values = {category: "NA" for category in custom_categories}

    # Encode the custom categories and extracted data labels
    model = SentenceTransformer(model_name)

    # List of confidence scores for each match
    confidence_scores = []

    # **Match percentage-based values to percentage categories**
    if percentage_values:
        category_embeddings = model.encode(percentage_categories, convert_to_tensor=True)
        extracted_embeddings = model.encode(list(percentage_values.keys()), convert_to_tensor=True)
        if extracted_embeddings.shape[0] > 0 and category_embeddings.shape[0] > 0:
            similarity_matrix = cosine_similarity(extracted_embeddings.cpu().numpy(), category_embeddings.cpu().numpy())
        else:
            similarity_matrix = np.array([])  # Prevents error when empty

        if similarity_matrix.size > 0:
            for i, extracted_label in enumerate(percentage_values.keys()):
                best_match_idx = np.argmax(similarity_matrix[i])
                best_match_category = percentage_categories[best_match_idx]
                best_match_score = similarity_matrix[i][best_match_idx]

                if best_match_score >= threshold:
                    final_mapped_values[best_match_category] = percentage_values[extracted_label]
                    # Log confidence score for each match
                    confidence_scores.append(best_match_score)

    # **Match non-percentage values to non-percentage categories**
    if non_percentage_values:
        category_embeddings = model.encode(non_percentage_categories, convert_to_tensor=True)
        extracted_embeddings = model.encode(list(non_percentage_values.keys()), convert_to_tensor=True)
        if extracted_embeddings.shape[0] > 0 and category_embeddings.shape[0] > 0:
            similarity_matrix = cosine_similarity(extracted_embeddings.cpu().numpy(), category_embeddings.cpu().numpy())
        else:
            similarity_matrix = np.array([])  # Prevents error when empty

        if similarity_matrix.size > 0:
            for i, extracted_label in enumerate(non_percentage_values.keys()):
                best_match_idx = np.argmax(similarity_matrix[i])
                best_match_category = non_percentage_categories[best_match_idx]
                best_match_score = similarity_matrix[i][best_match_idx]

                if best_match_score >= threshold:
                    final_mapped_values[best_match_category] = non_percentage_values[extracted_label]
                    confidence_scores.append(best_match_score)
    
    # **Step 4: Rule-Based Matching for Remaining "NA" Values**
    def word_overlap(label1, label2):
        words1 = set(label1.lower().split())
        words2 = set(label2.lower().split())
        return len(words1 & words2)  # Count common words

    for extracted_label, extracted_value in extracted_data.items():
        for custom_label in custom_categories:
            if final_mapped_values[custom_label] == "NA":  # Only apply rule-based if still unmatched
                common_word_count = word_overlap(extracted_label, custom_label)
                if common_word_count >= 2:  # If two or more words match
                    final_mapped_values[custom_label] = extracted_value

    # Convert to pandas DataFrame
    df = pd.DataFrame(final_mapped_values.items(), columns=["Category", "Matched Value"])
    # Print confidence scores
    print("Confidence Scores:", confidence_scores)
    return df

custom_categories = [
    "GHG Emission - Scope 1",
    "GHG Emission - Scope 2",
    "GHG Emission - Scope 3",
    "Emission Intensity",
    "Total Energy Consumption",
    "Total water usage or consumption",
    "Water intensity",
    "Total waste generated",
    "% of waste recycled",
    "% of renewable energy or product produced"
]

input_text = assistant_response_1

# Run the function
df_results = extract_and_match(input_text, custom_categories, threshold=0.8)

# Run for 2nd part
input_text = assistant_response_2
custom_categories = [
    "Total Workplace Injuries",
    "Total Fatal Incident",
    "Number of Employees",
    "Employee Diversity",
    "Number of New Hires"
]
df_results_2 = extract_and_match(input_text, custom_categories, threshold=0.5)

df = pd.concat([df_results, df_results_2], ignore_index=True)
df