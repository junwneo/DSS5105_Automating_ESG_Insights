import camelot
import pandas as pd
import re
import numpy as np
import pdfplumber
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path
from shapely.geometry import box
from shapely.strtree import STRtree
import unicodedata

import fitz
from PIL import Image
import io
import cv2
import numpy as np

def remove_text_from_rect(page, rect):
    """
    Removes text from a given rectangular area on the page by deleting the text content.

    Parameters:
    - page (fitz.Page): The page object.
    - rect (fitz.Rect): The rectangular area where text should be removed.
    """
    page.add_redact_annot(rect)  # This marks the area for redaction
    
    # Apply the redactions (remove text) from the page
    page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)
    
    
def save_cropped_pdf(doc, page_num, split_column, new_doc):
    """
    Splits a PDF page into left and right parts and fully removes the unwanted content.

    Parameters:
    - doc (fitz.Document): The original PDF document.
    - page_num (int): The page number to split.
    - split_column (float): The x-coordinate at which the page should be split.
    - new_doc (fitz.Document): The new PDF document where the cropped pages will be added.
    """
    page = doc[page_num]
    page_width = page.rect.width
    page_height = page.rect.height

    left_rect = fitz.Rect(0, 0, split_column, page_height)
    right_rect = fitz.Rect(split_column, 0, page_width, page_height)
    new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
    remove_text_from_rect(new_doc[-1], right_rect)
    new_doc[-1].set_cropbox(left_rect)
    
    new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
    remove_text_from_rect(new_doc[-1], left_rect)
    new_doc[-1].set_cropbox(right_rect)
    
    return new_doc

def fitz_rect_to_camelot_bbox(fitz_rect, page_height):
    """
    Convert a FitZ Rect to Camelot's bbox format (left, top, right, bottom).
    
    Args:
        fitz_rect: The fitz.Rect object.
        page_height: The height of the page, which is necessary to adjust the y-axis (since FitZ and Camelot have different y-axis origins).
        
    Returns:
        tuple: A tuple representing the bbox in Camelot format (left, top, right, bottom).
    """
    # FitZ Rect has: x0, y0, x1, y1
    # Camelot expects bbox in format: (left, top, right, bottom)
    if fitz_rect == 'All':
        return ['All']
    left = fitz_rect.x0
    right = fitz_rect.x1
    top = page_height - fitz_rect.y0  
    bottom = page_height - fitz_rect.y1  
    bbox = [left, top, right, bottom]
    return list(map(str, bbox))


def split_pdf_at_gap(pdf_path, pages_to_split, threshold_length=300, high_intensity=250):
    """
    Splits a PDF at a detected gap in the middle third of each specified page and saves the result as a new text-based PDF.
    
    Parameters:
    - pdf_path (str): Path to the input PDF file.
    - pages_to_split (list): List of page indices that contain tables and should be split.
    - threshold_length (int): Length of the high-intensity region required to be considered a gap.
    - high_intensity (int): Pixel intensity threshold for detecting a gap (higher value indicates a gap).
    """
    doc = fitz.open(pdf_path)
    new_doc = fitz.open()  # Create a new PDF document

    for page_num in pages_to_split:
        page = doc.load_page(page_num)
        page_height = page.rect.height
        page_width = page.rect.width

        # Convert the page to an image for analysis
        pix = page.get_pixmap(dpi=300)
        image = Image.open(io.BytesIO(pix.tobytes("png")))
        image_np = np.array(image)
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGBA2GRAY)

        height, width = gray.shape
        image_page_ratio = height / page_height
        start_col = width // 3
        end_col = 2 * width // 3
        middle_image = gray[:, start_col:end_col]
        histogram = np.mean(middle_image, axis=0)
        high_intensity_region = []
        gap_start, gap_end = None, None  

        # Loop through the histogram to detect long sequences of high intensity values (gap)
        for i in range(len(histogram)):
            if histogram[i] > high_intensity:  
                high_intensity_region.append(i)
            else:
                if len(high_intensity_region) > threshold_length:
                    gap_start = high_intensity_region[0]
                    gap_end = high_intensity_region[-1]
                    break
                high_intensity_region = []

        if gap_start is None or gap_end is None:
            print(f"No significant gap detected in page {page_num}. Inserting original page.")
            # If no gap detected, just insert the original page (as a blank or empty page to maintain structure)
            page = doc.load_page(page_num)  # Load the original page
            new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
            # camelot_bboxes.append('All')
        else:
            split_column_img = start_col + gap_start + (gap_end - gap_start) // 2  
            split_column = split_column_img / image_page_ratio
            print(f"Splitting page {page_num} at column: {split_column}")
            # Save cropped text-based pages into the new PDF document
            new_doc = save_cropped_pdf(doc, page_num, split_column, new_doc)

    # Save the new text-based PDF
    new_pdf_path = pdf_path.replace('.pdf', '_split_text.pdf')
    new_doc.save(new_pdf_path)
    print(f"Saved the split text PDF: {new_pdf_path}")
    page_count = new_doc.page_count
    return new_pdf_path, page_count
 

def detect_extract_tables(pdf_path, save_tables=True):
    print('Begin reading pdf...\n')
    
    table_pages = detect_tables(pdf_path)
    print(f'Finish detecting tables:{table_pages}') 
    # table_pages = [0]
    
    # check pdf separation and update table_pages
    new_pdf_path, new_page_count = split_pdf_at_gap(pdf_path, table_pages)
    
    table_num = 0
    df_dict_list_all = []
    for page_num in range(new_page_count):
        print(f'\nProcessing page {page_num}..')
        df_dict_list_page = get_page_tables_adjusted(new_pdf_path, page_num, save_tables=False, start_idx=table_num)
        if not df_dict_list_page:
            continue
        table_num += len(df_dict_list_page)
        df_dict_list_all += df_dict_list_page
    
    if save_tables:
        xlsx_name = Path(pdf_path).stem + '_all' + '.xlsx'  # all
        xlsx_dir = './output/table'
        xlsx_path = Path(xlsx_dir) / xlsx_name
        xlsx_path.parent.mkdir(parents=True, exist_ok=True) 
        save_tables_to_xlsx(df_dict_list_all, xlsx_path, start_idx=0)
    
    sheets_dict = get_sheet_dict(df_dict_list_all, start_idx=0)
    
    return sheets_dict, df_dict_list_all  


def get_page_tables_adjusted(pdf_path, page_num, save_tables=True, start_idx=0, flavor_list = ['hybrid', 'lattice']):
    # extract: tablelist has no title or columns detected, pure list of df
    table_dict, df_dict_flavors = get_best_table_camelot(pdf_path, flavor_list, ",".join(map(str, [page_num+1])))  
    # print(table_dict['df'])
    print(df_dict_flavors)
    
    if not table_dict:
        return []
    
    # refine/cleaning: go through each one
    # element in df_dict_list = {'title': title, 'df': df, 'page': page_string, bbox':bbox} for each table
    df_dict_list = [clean_table_df(table_df) for table_df in table_dict['df']] 
    for i in range(len(table_dict['df'])):
        df_dict_list[i]['bbox'] = table_dict['bbox'][i]
        df_dict_list[i]['page'] = page_num
    df_dict_list = [df_dict for df_dict in df_dict_list if not df_dict['df'].empty]
    
    # update bbox
    # refined_plumber_bbox = find_table_position(table_df, pdf_file, table_bbox, page_num=0)
    df_dict_list_1 = list(map(lambda df_dict: {**df_dict, 'bbox': find_table_position(df_dict['df'], pdf_path, df_dict['bbox'], page_num)}, df_dict_list))
    # final sanity check after update the position
    index_remove = remove_redundant_tables(
        {'df': [df_dict['df'] for df_dict in df_dict_list_1], 'bbox':[df_dict['bbox'] for df_dict in df_dict_list_1]}, 
        'pdfplumber'
    )
    df_dict_list_2 = [d for i,d in enumerate(df_dict_list_1) if i not in index_remove]
    # adjust column order  (page_num=0)
    df_dict_list_3 = list(map(lambda df_dict_2: {**df_dict_2, 'df': adjust_column_name_order(df_dict_2['bbox'], df_dict_2['df'], pdf_path, page_num)}, df_dict_list_2))
        
    if [df_dict for df_dict in df_dict_list_3 if len(df_dict['df'])>0]:
        print(f'page {page_num} got {len(df_dict_list_3)} final df!')
    
    #  save df_dict_list into xlsx, each df_dict into an independent sheet, with the sheetname being its title (if title is none then just use Sheet + number)
    if save_tables:
        xlsx_name = Path(pdf_path).stem + '.xlsx'
        xlsx_dir = './output/table'
        xlsx_path = Path(xlsx_dir) / xlsx_name
        xlsx_path.parent.mkdir(parents=True, exist_ok=True) 
        save_tables_to_xlsx(df_dict_list_3, xlsx_path, start_idx)
    
    return df_dict_list_2


def detect_tables(pdf_path):
    table_pages = []
    with pdfplumber.open(pdf_path) as pdf:
        page_num = len(pdf.pages)
        for i, page in enumerate(pdf.pages):
            tables = page.extract_tables()
            if tables:
                dfs = [pd.DataFrame(table) for table in tables if any(any(cell for cell in row) for row in table)]
                if dfs:
                    table_pages.append(i)
            elif 2 <= abs(i - page_num)<= 12:
                table_pages.append(i)
    return table_pages



def remove_redundant_tables(df_dict_0, bbox_type, page_height=None):
    """
    Background:
    - There are cases that the table may be extracted multiple times but differently, which may lead the algorithm choosing the wrong option
    
    Method:
    - check bbox
    - content (TBD)
    """        
    # print(f'camelot_bbox:{df_dict_0['bbox']}')
    num = len(df_dict_0['df'])
    index_remove = []
    df_list = df_dict_0['df']
    bbox_list = df_dict_0['bbox']
    for i in range(num):
        if df_list[i].empty or i > len(bbox_list) - 1 or bbox_list[i] is None:
            index_remove.append(i)
    df_dict_0 = {'df': [df for i, df in enumerate(df_list) if i not in index_remove], 'bbox':[b for i, b in enumerate(bbox_list) if i not in index_remove]}
            
    if bbox_type == 'camelot' and page_height:
        bbox_list = [box(b[0], page_height - b[-1], b[2], page_height - b[1]) for b in df_dict_0['bbox']]
    elif bbox_type == 'pdfplumber':
        bbox_list = [box(b[0], b[3], b[2], b[1]) for b in df_dict_0['bbox']]
    else:
        print('Wrong input in remove_redundant_tables!')
        return []
    # print(f'bbox_list{bbox_list}')
    if len(bbox_list) <= 1:
        return df_dict_0
    # R-tree reduces unnecessary comparisons by organizing boxes hierarchically
    print('Check containing ..')
    tree = STRtree(bbox_list)    
    remove_index_list = []
    epsilon = 20
    for i, bbox in enumerate(bbox_list):
        possible_matches = tree.query(bbox)  # Fast spatial query
        for j in possible_matches:
            if i != j:
                candidate = bbox_list[j]
                # print(f'candidate {j}; target:{i}')
            else:
                continue
            if candidate != bbox and candidate.buffer(epsilon).contains(bbox):  # Ensure not the same object(maybe result of separate tables)
                # to be safe: check the information density (didn't check content)
                print(f'{j} Contain {i} inside!')
                candi_df = df_dict_0['df'][j]
                target_df = df_dict_0['df'][i]
                candi_info = candi_df.shape[0]*candi_df.shape[1] - candi_df.isna().sum().sum()
                target_info = target_df.shape[0]*target_df.shape[1] - target_df.isna().sum().sum()
                if candi_info > target_info and candi_df.shape[1] >= target_df.shape[1]:
                    print(f'Remove table {i}')
                    remove_index_list.append(i)
                    break
    
    return remove_index_list


def read_tables_camelot(src, flavor, pages):
    tables = camelot.read_pdf(src, flavor=flavor, pages=pages, suppress_stdout=True, strip_text=' ', split_text=True)
    table_list = [] 
    position_list = []  
    
    for table in tables:
        table_df = table.df
        # print(table_df.head())
        table_list.append(table_df)
        position_list.append(table._bbox)  # (x1, y1, x2, y2)
    
    return table_list, position_list

def get_best_table_camelot(pdf_path, flavor_list, pages='all'):    
    def unpack_rows(df):
        """ 
        if there is no pattern of number of \n, then just leave it unchanged, which later will be deprecated by None check
        special case: if there is for the first few rows, then maybe it's evenly spread for the row
        """
        def count_segments(df_col):
            df_col = df_col.dropna().astype(str)
            counts = [len(e.split('\n')) for e in df_col]
            num = sum(counts)
            return num
        
        def most_frequent_number(count_list):
            count = Counter(count_list)
            most_common = count.most_common(1)
            if most_common and most_common[0][1] > 1:
                return most_common[0][0]
            return None
        
        # premise: Missing values (to mitigate collateral mistakes, stick to stirct conditions)
        col_count_df = df.apply(count_segments, axis=0)
        if df.shape[1] < 2:
            return pd.DataFrame()
        
        mode = most_frequent_number(col_count_df.tolist())
        print(f'mode: {mode}, entries_num:{df.shape[0]}')
        if not mode or mode<=2*df.shape[0] or (df.shape[0]<=2 and mode<6):
            return df
        print('Unpack rows..')
        df_1 = pd.DataFrame(np.zeros((mode, df.shape[1])), columns=df.columns).astype(str)
        string_list = ['\n'.join(pd.Series(c).dropna().astype(str)) for c in df.values.T]
        elements_list = [s.split('\n') for s in string_list]
        for i, l in enumerate(elements_list):
            num = len(l)
            if num==mode:
                df_1.iloc[:, i] = l
            elif num > mode:
                n = num//mode
                m = num%mode
                list_1 = []
                for k in range(mode - 1):
                    list_1.append('\n'.join(map(str, l[n * k : n * (k + 1)])))
                list_1.append('\n'.join(map(str, l[-m:])))
                df_1.iloc[:, i] = list_1
            else:
                list_1 = l + ['']*(mode-num)
                df_1.iloc[:, i] = list_1
        # print(f'Get df_1: {df_1}')
        
        df_2 = df_1.map(lambda x: re.sub(r"(?<!\n)[ \t\r]+(?!\n)", " ", 
                            unicodedata.normalize("NFKC", x.replace("*", "")))
            .replace("\u202f", "")  # remove the narrow no-break space
            .strip() if isinstance(x, str) else x)
        return df_2
        
    def complement_cols(df):
        # for numeric value based tables
        if df.empty or df.shape[0]<=1 or df.shape[1]<=1:
            return df
        
        df_1 = df.copy()
        # check column names
        numeric_num = (df.iloc[:, 1:].map(pd.to_numeric, errors='coerce').notnull()).sum().sum()
        numeric_perc = numeric_num/(df.shape[0]*df.shape[1]-df.shape[0])
        # print(f'numeric_perc for col complementary: {numeric_perc}')
        # print(df)
        if numeric_perc >= 0.6:
            if pd.to_numeric(df_1.iloc[0], errors='coerce').notnull().sum()>1:
                df_1 = pd.concat([pd.DataFrame([[None]*df_1.shape[1]]), df_1]).reset_index(drop=True)
                # return df_1
                # check index
                if pd.to_numeric(df_1.iloc[:, 0], errors='coerce').notnull().sum()>1:
                    new_col = pd.Series([None] * df_1.shape[0])
                    df_1 = pd.concat([new_col, df_1], axis=1).reset_index(drop=True)
        df_1.columns = pd.RangeIndex(start=0, stop=df_1.shape[1], step=1)
        return df_1
        
    
    def separate_tables(df):
        """
        horizonally total same columns; 
        TBD: vertically (prob: column names remain unclean)
        """
        def check_first_last_col(df, col):
            df_col = df.iloc[1:, col].replace({None: pd.NA, "": pd.NA}).dropna()
            # print(f'check col:\n{df.iloc[:, col]}')
            if len(df_col) < 3:
                return [df]
            # print(f'df_last: \n{df_last}')
            df_count_n = df_col.apply(lambda x: x.count('\n'))
            num = len([e for e in df_count_n.values if e>0])
            perc = num/len(df_col)
            num_n = sum([e for e in df_count_n.values if e>0])
            perc_n = num_n/num if num>0 else 0
            # print(f'df_count_n:\n{df_count_n}', '\n', perc, perc_n)
            if len(set(df_count_n.values)) <= 3 and perc>=0.5 and perc_n>=3: 
                # print('Separate last column!')
                df_1 = df.drop(df.columns[col], axis=1)
                df_2 = df.iloc[:, col]
                # print(df_2)
                df_split = df_2.str.split('\n', expand=True)
                df_split.dropna(how='all', axis=0, inplace=True)
                if df_split.iloc[0, -1] is None:
                    df_split.iloc[0, 1:] = df_split.iloc[0, :-1]
                    df_split.iloc[0, 0] = None
                # print(df_split)
                df_split.columns = [i for i in range(df_split.shape[1])]
                df_list = [df_1, df_split]
                return df_list
            return [df]
        
        # although there is no repeated column names, repeated None structure
        # Iterate over each row to find the column where the NaN starts and all after are NaN
        def find_column_index(row):
            if len(row) <=4:
                return None
            for i in range(1, len(row)-1):
                if pd.isna(row[i]):  # If the value is NaN
                    # Check if all values after this column are also NaN
                    if all(pd.isna(row[i:])):
                        if i <=3:
                            return None
                        else:
                            return i
            return None
                    
        def find_repeatedness(df, axis):
            if axis == 0: # vertically (index)
                first_text = df.iloc[:, 0].dropna().tolist()
            else: # horizontal
                first_text = df.iloc[0].dropna().tolist()
            first_text = [x for x in first_text if x != '']
            # print(f'first_text:{first_text}')
            num = df.shape[0] if axis == 0 else df.shape[1]
            if len(first_text)>=4 and num>2:
                # print('check whether have multiple tables horizontally')
                if len(first_text)%2 == 0:
                    first_half = first_text[:int(len(first_text)/2)]
                    second_half = first_text[int(len(first_text)/2):]
                    if len(first_half) >2 and first_half[1:] == second_half[1:]:
                        element = str(first_half[-1])
                        match = df.iloc[0].apply(lambda x: str(x).strip() == element) if axis==1 else df.iloc[:, 0].apply(lambda x: str(x).strip() == element)
                        true_indices = match[match].index
                        if true_indices.any():
                            # print(f'match:{true_indices}')
                            separate_index = true_indices[0]
                            df_list = [df.iloc[:, :separate_index+1], df.iloc[:, separate_index+1:]] if axis==1 else [df.iloc[:separate_index+1, :], df.iloc[separate_index+1:, :]]
                            return df_list
                else:
                    first_half = first_text[:int((len(first_text)-1)/2)]
                    second_half = first_text[int((len(first_text)-1)/2):]
                    if len(set(first_half) & set(second_half)) == (len(first_text)-1)/2 or len(first_half) >2 and first_half[1:] == second_half[1:]:
                        element = str(first_half[-1])
                        match = df.iloc[0].apply(lambda x: str(x).strip() == element) if axis==1 else df.iloc[:, 0].apply(lambda x: str(x).strip() == element)
                        true_indices = match[match].index
                        if true_indices.any():
                            # print(f'match:{true_indices}')
                            separate_index = true_indices[0]
                            df_list = [df.iloc[:, :separate_index+1], df.iloc[:, separate_index+1:]] if axis==1 else [df.iloc[:separate_index+1, :], df.iloc[separate_index+1:, :]]
                            # print(f'\nseparate tables:\n{df_list[1]},\n{df.iloc[1]}')
                            return df_list
                        
                    first_half = first_text[1:int((len(first_text)-1)/2)+1]
                    second_half = first_text[int((len(first_text)-1)/2)+1:]
                    if len(set(first_half) & set(second_half)) == (len(first_text)-1)/2 or len(first_half) >2 and first_half[1:] == second_half[1:]:
                        element = str(first_half[-1])
                        match = df.iloc[0].apply(lambda x: str(x).strip() == element) if axis==1 else df.iloc[:, 0].apply(lambda x: str(x).strip() == element)
                        true_indices = match[match].index
                        if true_indices.any():
                            # print(f'match:{true_indices}')
                            separate_index = true_indices[0]
                            df_list = [df.iloc[:, :separate_index+1], df.iloc[:, separate_index+1:]] if axis==1 else [df.iloc[:separate_index+1, :], df.iloc[separate_index+1:, :]]
                            # print(f'\nseparate tables:\n{df_list[1]},\n{df.iloc[1]}')
                            return df_list
                    
            return [df]
                
        
        def separate_table_horizontal(df):
            df_list = find_repeatedness(df, axis=1)
            if len(df_list)==2:
                return df_list
            num = df.shape[0]
            df_1 = df.replace({None: pd.NA, '': pd.NA})
            df_1.columns = list(range(df.shape[1]))
            susp_col_idx = df_1.apply(find_column_index, axis=1)
            susp_col_idx = [x for x in susp_col_idx if pd.notna(x) and str(int(x)).isdigit()]
            # print(f'susp_col_idx:{susp_col_idx}')
            if len(set(susp_col_idx))<=5 and len(susp_col_idx)/num>0.3 and len(susp_col_idx)>=10:
                col = int(sorted(susp_col_idx)[0])
                print(f'Find start of NAN col:{col}')
                df_list = [df.iloc[:, :col], df.iloc[:, col:]]
                return df_list
            # check the first and last column
            dfs_list = []
            df_1 = df.copy()
            if df.shape[1] > 4:
                for col in [0, -1]:
                    print(f'check column {col} for another table')
                    df_list = check_first_last_col(df_1, col)
                    if len(df_list)>1 and not dfs_list:
                        dfs_list = df_list
                        df_1 = df_list[0]
                    elif len(df_list)>1:
                        dfs_list = dfs_list[1] + df_list
                        df_1 = df_list[0]
            if dfs_list:
                return dfs_list
        
            return [df]
        # print(f'df before separating:\n{df}')
        df_list_1 = separate_table_horizontal(df)
        if len(df_list_1)==2:
            return df_list_1
        # print(f'find repeated for index')
        df_list_0 = find_repeatedness(df, axis=0)
        if len(df_list_0)==2:
            print('separate table vertically (index)')
            return df_list_0
        return [df]
                    
    def cal_none_perc(df):
        num_none_col = df.iloc[0].isna().sum() # give more penalty for missing column names
        num_none_idx = df.iloc[:, 0].isna().sum()
        num_none = df.iloc[1:, 1:].isna().sum().sum()
        num_total = df.shape[0] * df.shape[1]
        n_count_index = df.iloc[:, 0].apply(lambda x: x.count('\n') if x else 0).sum()
        # print(f'n_count_index:{n_count_index}')
        
        if num_none_col >=2 and num_none_idx>=2:
            none_perc = (num_none_col*5 + num_none + num_none_idx*5)/num_total
        elif num_none_col>0 and n_count_index/num_none_col>0.5:
            none_perc = (num_none_col*5 + num_none + num_none_idx*2 + n_count_index*2)/num_total
        else:
            none_perc = (num_none_col*5 + num_none + num_none_idx*2)/num_total
        
        return none_perc
    
    def get_numeric_table(df):
        """ 
        If has enough numeric data, return the df (remove * and unicode), otherwise return empty df
        """
        numeric_perc = df.iloc[:, 1:].map(cal_numeric_values).mean().mean()
        none_perc = df.isna().mean().mean()
        # print(f'numerc_perc:{numeric_perc}')
        if numeric_perc >= 0.2 and none_perc<=0.8:  # small threshold considering tables may not in good structure
            df = df.map(lambda x: re.sub(r"(?<!\n)[ \t\r]+(?!\n)", " ", x)
                .strip() if isinstance(x, str) else x)
            return df
        return pd.DataFrame() 
    
    def examine_table(df, string_thre = 0.5, string_none_thre = 0.15):
        """ 
        Goal: remove invalid values if the whole table is invalid, will return an empty dataframe.
        
        Args:
        - df: dataframe of one table to be checked
        - string_thre: threshold for checking validness of object columns (remove None values in denominator), 0.5 by default
        - string_none_thre: threshold for checking validness of object columns (the whole column), 0.15 by default
        """
        
        def check_first_col(df):
            df_col = df.iloc[:, 0]
            none_val = df_col.isna().sum()
            start_with_num = df_col.str.match(r'^\d').sum() - df_col.str.match(r'^\d{4}$').sum()
            denominator = len(df_col) - none_val
            if denominator == 0:
                denominator = 0.001
            invalid_perc = (start_with_num)/denominator
            if invalid_perc > 0.5:
                return pd.DataFrame()
            else:
                return df
        
        def check_strings(df): 
            # helper function
            def check_punctuation(df_obj):
                df_obj = df_obj.apply(lambda col: col.map(lambda x: x.rstrip('.') if isinstance(x, str) else x))
                invalid_punc_df = pd.DataFrame(0, index=df_obj.index, columns=df_obj.columns)
                def has_non_float_dot(value):
                    if not isinstance(value, str) or '.' not in value:
                        return 0  # Ignore non-strings or values without a dot
                    float_pattern = re.compile(r'^\d+\.\d+$')  
                    if float_pattern.fullmatch(value):  
                        return 0  # Directly return 0 if it's a float
                    # Check if at least one `.` is NOT part of a float
                    for i, char in enumerate(value):
                        if char == '.':
                            # If it's the first or last character, it's clearly not a decimal point
                            if i == 0 or i == len(value) - 1:
                                return value
                            # If the character before OR after is not a digit, it's not a decimal point
                            if not (value[i-1].isdigit() and value[i+1].isdigit()):
                                return value  # Found a non-float dot, return the value
                    return 0
                    
                for col in df_obj.columns:
                    df_col = df_obj[col]
                    values_array = df_col.values
                    index_list = df_col.index
                    punc_list = [has_non_float_dot(v) for v in values_array] 
                    if punc_list:
                        invalid_list = [index_list[i] for i in range(len(punc_list)) if punc_list[i]!=0]
                        invalid_punc_df.loc[invalid_list, col] = 1
                return invalid_punc_df  # invalid_perc_df 
            
            def check_capital_pattern(df_obj):
                df_invalid = pd.DataFrame(0, index=df_obj.index, columns=df_obj.columns)
                df_cap = df_obj.map(lambda x: int(x[0].isupper()) if isinstance(x, str) and x and not x[0].isdigit() else x) # only check string upper/lower
                df_count_pattern = pd.DataFrame([[0]*df_cap.shape[1]], columns=df_cap.columns)
                for col in df_cap.columns:
                    df_cap_col = df_cap[col]
                    # remove irrelevant part
                    df_cap_col = df_cap_col[df_cap_col != -1] 
                    indices = df_cap_col.index.to_list()
                    values = df_cap_col.tolist()
                    # counting
                    valid_count = 0
                    in_sequence = False
                    seq_start_idx = None  
                    temp_count = 0  # to remove the isolated 1
                    seq_indices = []
                    
                    for i in range(len(values)): # iterate over the list
                        idx = indices[i]
                        if df_cap_col[idx] == 1:
                            seq_indices = [idx]
                            in_sequence = True
                            seq_start_idx = idx
                            temp_count = 1  # Start counting sequence

                        elif df_cap_col[idx] == 0:
                            if in_sequence:  # Continue valid sequence
                                seq_indices.append(idx)
                                temp_count += 1
                            # Ensure index continuity
                            if seq_start_idx is not None and idx > seq_start_idx:
                                if indices[i] != indices[i - 1] + 1:  # Non-contiguous index
                                    in_sequence = False  # Invalidate sequence
                                    temp_count = 0
                                    seq_indices = []
                                    
                        else:
                            if in_sequence and df_cap_col[indices[i-1]]==0:
                                temp_count += 1
                        if in_sequence and (i == len(values) - 1 or values[i + 1] == 1):
                            if temp_count > 2:  # Ensure at least two 0 exists
                                valid_count += temp_count  # number of values that follow the pattern
                                df_invalid.loc[seq_indices,col] = 1
                            in_sequence = False
                            temp_count = 0
                            
                    df_count_pattern[col] = valid_count
                
                return df_invalid # df_count_pattern
            
            
            def is_obj_column_valid(df, col):
                # Check if the column is of dtype object and contains less than 10% numeric values or digit strings
                df_col = df[col]
                df_col = df_col.str.replace(r'\s+', ' ', regex=True).str.strip()
                df_col = df_col.str.replace('\xa0', '', regex=True)
                df_col = df_col.astype(str).apply(lambda x: x.replace('\n', '').replace('-', '').strip())
                digit_str_count = df_col.str.match(r'^-?[\d,]+(\.\d+)?$').sum()
                none_value = df_col.isna().sum()
                total_count = len(df_col) - none_value
                if total_count == 0:
                    total_count = 0.01
                # Calculate the percentage of digit strings
                digit_percentage = (digit_str_count / total_count) * 100

                return digit_percentage < 10
            
            obj_cols = df.select_dtypes(include=['object']).columns
            obj_cols = [col for col in obj_cols if is_obj_column_valid(df, col)]
            # print(f'obj_col:{obj_cols}')
            df_obj = df[obj_cols].map(lambda x: re.sub(r'^\W+', '', x) if isinstance(x, str) else x)
            num_entries = df.shape[0]
            none_array = df_obj.isna().sum().values.reshape(1, -1)
            # check capital pattern
            df_pattern_invalid = check_capital_pattern(df_obj)
            # check punctuation
            df_punc_invalid = check_punctuation(df_obj)
            # combine and calculate perc
            df_combined_invalid = df_pattern_invalid + df_punc_invalid
            df_combined_invalid = df_combined_invalid.replace({1: None, 2: None})
            invalid_array = df_combined_invalid.isna().sum().values.reshape(1, -1)
            epsilon = 1e-10  # Small value to avoid division by zero
            denominator = num_entries - none_array
            denominator = np.where(denominator == 0, epsilon, denominator)
            invalid_str_perc_array = (invalid_array)/denominator
            df_invalid_obj = pd.DataFrame(invalid_str_perc_array, columns=obj_cols)
            invalid_perc_array =  (invalid_array)/(num_entries)
            df_invalid_all = pd.DataFrame(invalid_perc_array, columns=obj_cols)
            
            return df_invalid_obj, df_invalid_all
        

        # missing values
        df = df.replace(r'^\s*$', None, regex=True)
        if df.empty:
            return df
        else:
            df_check_col = check_first_col(df)
            if df_check_col.empty:
                return df_check_col
            
            # check string: capital letter and punctuation
            df_invalid_obj, df_invalid_all = check_strings(df)
            # Q: take the idea of forest (voting) for the cols?
            for col in df_invalid_obj.columns[1:]:
                if df_invalid_obj[col][0] >= string_thre or df_invalid_all[col][0] >= string_none_thre:
                    if df[col][:3].str.lower().isin(['unit']).any():
                        continue
                    # print(df_invalid_obj[col][0], df_invalid_all[col][0])
                    # print(df[col])
                    df = df.drop(columns=[col])
                    
                else:
                    # check none_perc 
                    none_perc = df[col].isna().sum()/df.shape[0]
                    if  none_perc >= 0.8:
                        df = df.drop(columns=[col])
            
            # avoid df having too many strings
            lengths = df.map(lambda x: len(str(x)) if pd.notna(x) else 0)
            avg_len = lengths.sum().sum()/(df.map(lambda x: 1 if pd.notna(x) else 0).sum().sum())
            # print(f'avg_len:{avg_len}')
            if avg_len >= 40:
                df = pd.DataFrame()
            if not df.empty and df.iloc[-1, 0] and df.iloc[-1, 0].count('\n')>3:
                df = pd.DataFrame()
            
        return df
    
    flavor = 'hybrid' # by default
    
    # get results for different flavor
    df_dict = {key: [] for key in flavor_list}
    for flavor in flavor_list:
        # print(f'\nfit flavor {flavor}')
        df_list_0, bbox_list_0 = read_tables_camelot(pdf_path, flavor, pages)
        # print(f'df_list_0 for {flavor}: {df_list_0}')
        df_valid_list = []
        bbox_valid_list = []
        for i, df in enumerate(df_list_0):
            # check validness of df
            df = unpack_rows(df)
            df = examine_table(df)
            df = df.dropna(how='all')
            df = df.dropna(axis=1, how='all')
            df = get_numeric_table(df)
            if not df.empty and df.shape[1]>=2 and df.shape[0]>=2:
                df = complement_cols(df)
                df_valid_list.append(df)
                bbox_valid_list.append(bbox_list_0[i])
     
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[int(pages)-1]  
            page_height = page.height    
                
        remove_index_list = remove_redundant_tables({'df':df_valid_list, 'bbox': bbox_valid_list}, 'camelot', page_height)
        df_list = [df for i, df in enumerate(df_valid_list) if i not in remove_index_list]
        bbox_list = [b for i, b in enumerate(bbox_valid_list) if i not in remove_index_list]
        
        # separate tables (should happen after unpacking)
        # print(f'num of tables for  {flavor}: {len(df_list)}')
        if len(df_list) < 4:
            df_list_1 = []
            bbox_list_1 = []
            for i, df in enumerate(df_list):
                # print(f'check separation for {flavor}:\n{df.iloc[:2]}')
                separated_table_list = separate_tables(df)
                separated_table_list = [df_.dropna(how='all', axis=0).reset_index(drop=True) for df_ in separated_table_list]
                # separated_table_list  = [pd.concat([df_.iloc[0], combine_rows(df_.iloc[1:])], axis = 0) for df_ in separated_table_list]
                df_list_1 += separated_table_list
                if len(separated_table_list) == 1:
                    bbox_list_1.append(bbox_list[i])
                else:
                    print(f'Separate df for {flavor}')
                    bbox_list_1 += [bbox_list[i]]*2
        else:
            df_list_1 = df_list
            bbox_list_1 = bbox_list
            
        df_dict[flavor] = {'df': df_list_1, 'bbox': bbox_list_1, 'page': pages}
        
    # drop out if no df identified
    non_empty_flavor = [key for key in df_dict if len(df_dict.get(key).get('df'))>0]
    non_overloaded_flavor = [key for key in df_dict if (len(df_dict.get(key).get('df'))>0) and (len(df_dict.get(key).get('df'))<=4)]
    
    if len(non_empty_flavor)==1:
        flavor = non_empty_flavor[0]
    elif len(non_overloaded_flavor)==1:
        flavor = non_overloaded_flavor[0]
    elif len(non_overloaded_flavor)==0:
        return {}, {}
    else:
        # check the missing value percentage
        # step 1: compare general missing percentage 
        none_perc_dict = {key: [cal_none_perc(df) for df in df_dict[key]['df']] for key in non_empty_flavor}
        overall_none_perc_dict = {key: np.mean(none_perc_dict[key]) for key in non_empty_flavor}
        print(f'original overall_none_perc_dict: {overall_none_perc_dict}')
        # add penalty if all tables has no column names
        overall_none_perc_dict = {key: value for key, value in overall_none_perc_dict.items() if value <0.75}
        if not overall_none_perc_dict:
            print('No table df extracted!')
            return {}, {}
        
        for key in overall_none_perc_dict.keys():
            missing_columns = []
            if df_dict.get(key):
                dfs = df_dict[key]['df']
                valid_index_lst = []
                for i, df in enumerate(dfs):
                    # need to make sure its non-numeric (column)
                    numeric_rows = df.iloc[:, 1:].map(cal_numeric_values).sum(axis=1)/df.shape[1]
                    idx_col = numeric_rows[numeric_rows >= 0.3] if (numeric_rows >= 0.3).any() else None
                    if idx_col is not None and idx_col.index[0]==0:
                        none_row = pd.DataFrame([[np.nan] * df.shape[1]], columns=df.columns)
                        df = pd.concat([none_row, df]).reset_index(drop=True)
                        dfs[i] = df
                        overall_none_perc_dict[key] += 0.3
                    end = idx_col.index[0] if idx_col is not None else 2
                    valid_rows = df.iloc[0:end].isna().sum(axis=1).lt(2)  
                    # print(f'valid_rows:\n{valid_rows}')
                    valid_indices = valid_rows[valid_rows].index
                    valid_indices = [idx for idx in valid_indices if idx_col is not None and idx not in idx_col]
                    valid_index = min(valid_indices) if len(valid_indices)>0 else 0
                    # print(f'valid_index:{valid_index}, {df.iloc[valid_index]}')
                    valid_index_lst.append(valid_index)
                    # print(df.iloc[valid_index, 1:].notna().sum()<=1)
                missing_columns = [i for i, df in enumerate(dfs) if df.iloc[valid_index_lst[i], 1:].notna().sum() <= 1]
                # print(f'missing_col:{missing_columns}')
                if len(missing_columns) == len(dfs) and len(dfs)>1:
                    overall_none_perc_dict[key] += 0.2*len(dfs)
                elif len(missing_columns)==len(dfs) and len(dfs)==1:
                    overall_none_perc_dict[key] += 0.4
        
        print(f'overall_none_perc_dict: {overall_none_perc_dict}')
        min_none_perc = {key: value for key, value in overall_none_perc_dict.items() if value == min(overall_none_perc_dict.values())}
        if len(min_none_perc) == 1:
            flavor = list(min_none_perc.keys())[0]
        # step 2: still cannot decide then check the variance (distribution)
        else:
            var_none_perc_dict = {key: np.var(none_perc_dict[key]) for key in non_empty_flavor}
            min_var = {key: value for key, value in var_none_perc_dict.items() if value == min(var_none_perc_dict.values())}
            if len(min_var) == 1:
                flavor = list(min_var.keys())[0]
                
    print(f'Flavor chosen: {flavor}')      
    final_table_dict = df_dict[flavor]
    
    return final_table_dict, df_dict # df_dict

def cal_numeric_values(value):
    if value is None:
        return False
    value = str(value)
    exclude_chars = {',', '.', '-', '%', ' ', '\n'}
    # exclude date
    if value.isdigit() and (1800 <= int(value) <= 2100):  
        return False
    try:
        parsed_date = pd.to_datetime(value, errors='coerce', dayfirst=True)
        if pd.notna(parsed_date):  # If conversion is successful, it's a date
            return False
    except Exception:
        pass
    # Check if the value contains digits
    if any(char.isdigit() for char in value):
        non_digit_count = sum(1 for char in value if not char.isdigit() and char not in exclude_chars)            
        if non_digit_count <= 3:
            return True
    return False


def is_non_year_numeric(value):
    # Remove commas (thousands separators)
    value = str(value).replace(',', '')
    # Try to convert the value to a float
    try:
        # Convert to float
        num_value = float(value)
        if len(str(int(num_value))) == 4:  # check if the number has 4 digits, likely a year
            return False
        return True
    except ValueError:
        # If it can't be converted to a float, return False
        return False

# combined rows ('\n): by detecting the combination of None with neighbors
def combine_rows(df_2):
    
    def detect_row(df, row):
        def is_punctuation(char):
            # Regex pattern: match if the character is not alphanumeric
            return bool(re.match(r'[^a-zA-Z0-9]', char))
        def is_only_punctuation(row):
            return all(is_punctuation(str(cell)) for cell in row)
        
        # non_cols  = df.iloc[row].isna().tolist()
        num = df.shape[1]
        check_cols = [idx for idx in range(1, df.shape[1])]  # if non_cols[idx]==0] # start with 0 or 1
        df_up = df.iloc[row - 1].fillna('_').tolist()
        df_row = df.iloc[row].fillna('_').tolist()
        df_dw = df.iloc[row+1].fillna('_').tolist()
        
        df_up_row_none = df.iloc[row-1:row+1].fillna('').map(lambda x: 1 if x=='' else 0)
        list_combined_none = df_up_row_none.sum(axis=0).tolist()
        if all(df_up[idx] == '_' for idx in check_cols) and all(df_dw[idx] == '_' for idx in check_cols):
            print(f'Combine three rows, as the row above and below {row} are all missing')
            df_bf = df.iloc[:row-1]
            df_af = df.iloc[row+2:]
            df_md = pd.DataFrame([[df_up[i] + df_row[i] + df_dw[i] for i in range(num)]], columns=df.columns)
            df_md = df_md.apply(lambda col: col.str.replace('_', ' '))
            df_new = pd.concat([df_bf, df_md, df_af], axis=0, ignore_index=True).reset_index(drop=True)
        # elif all(df_up[idx] == '_' for idx in check_cols):
        #     print(f'Combine the row above {row}')
        #     df_bf = df.iloc[:row-1]
        #     df_af = df.iloc[row+1:]
        #     df_md = pd.DataFrame([[df_up[i] + '\n' + df_row[i] for i in range(num)]], columns=df.columns)
        #     df_md = df_md.apply(lambda col: col.str.replace('_', ' ').str.strip())
        #     df_new = pd.concat([df_bf, df_md, df_af], axis=0, ignore_index=True).reset_index(drop=True)
        # add: the whole row only has punctuation but no real stuff
        elif all(is_only_punctuation(str(cell)) for cell in df_row):
            print(f'The whole {row} row is all punctuation, deleted')
            df_bf = df.iloc[:row]
            df_af = df.iloc[row+1:]
            df_new = pd.concat([df_bf, df_af], axis=0, ignore_index=True)   
        # intertwined missing value between rows (two rows for now)
        elif len(list_combined_none) >= 3 and all(x == 1 for x in list_combined_none[1:]): 
            print(f'Interwined with the row above {row}')
            df_bf = df.iloc[:row-1]
            df_md = pd.DataFrame([[df_up[i] + df_row[i] for i in range(num)]], columns=df.columns)
            df_md = df_md.apply(lambda col: col.str.replace('_', ' ').str.strip())
            df_af = df.iloc[row+1:]
            df_new = pd.concat([df_bf, df_md, df_af], axis=0, ignore_index=True).reset_index(drop=True)
        else:
            df_new = df.copy()
        df_new = df_new.reset_index(drop=True)
        
        return df_new
    
    df_3 = df_2.copy()
    row = 0
    while row < df_3.shape[0]-1:
        if row == 0 and all(pd.isna(df_3.iloc[0, 1:])) and pd.to_numeric(df_3.iloc[1, 1:], errors='coerce').notnull().sum()>=1:
            row += 1
            continue
        df_3 = detect_row(df_3, row)
        row += 1
            
    return df_3

# cleaning
def clean_table_df(df):
    # identify title (maybe include the title of the page)
    def identify_title_columns(df):
        if df.shape[0]<3:
            return pd.DataFrame(), '', ''
        title = None
        cols_text = ''
        num_cols = df.shape[1]
        first_row = df.iloc[0].fillna('').tolist()
        second_row = df.iloc[1].fillna('').tolist()
        third_row = df.iloc[2].fillna('').tolist()
        first_row_all = df.iloc[0].astype(str)  # Ensure values are strings
        all_uppercase_start = first_row_all.str.match(r"^[A-Z]")
        combined_2_row = [[first_row[i], second_row[i]] for i in range(num_cols)]
        combined_2_row_c = [(0 if l.count('')==1 else 1) for l in combined_2_row]
        
        first_row_text = '_'.join(first_row)
        first_row_nume = sum([int(is_non_year_numeric(e)) for e in first_row])
        num_lines_1 = first_row_text.count('\n')
        second_row_text = '_'.join(second_row)
        second_row_nume = sum([int(is_non_year_numeric(e)) for e in second_row])
        num_lines_2 = second_row_text.count('\n')  
        third_row_text = '_'.join(third_row)   
        third_row_nume = sum([int(is_non_year_numeric(e)) for e in third_row]) 
        num_lines_3 = third_row_text.count('\n') 
        # numeric_num = (df.iloc[:, 1:].map(pd.to_numeric, errors='coerce').notnull()).sum().sum()
        # numeric_perc = numeric_num/(df.shape[0]*df.shape[1]-df.shape[0])
        # print(f'numeric_perc:{numeric_perc}')
        # Case 2: both first and second row has None, but can be combined into one row -> no title, first two rows merge into cols
        if sum(combined_2_row_c) == 0:
            print('First two rows as cols')
            cols = [str(l[0]) + str(l[1]) for l in combined_2_row]
            df_1 = df.iloc[2:]
            df_1.columns = cols
            
        # Case 1: no missing values in first row -> column names, no title
        elif first_row[1:].count('') == 0 and 2* num_cols > num_lines_1 >=0 and first_row_nume<=2 and df.iloc[0].dtype==object and all(all_uppercase_start):
            print('First row as cols!')
            df_1 = df.iloc[1:]
            cols = first_row
            df_1.columns = cols
        
        # Case 3: two many \n in the first row: first row as cols combined, no title
        elif 2*num_cols > num_lines_1 > max(num_cols-2, 2):
            print('First row as cols, tangled!')
            cols_text = first_row_text.strip('_')
            df_1 = df.iloc[1:]  # df_1.columns = indexRange
        # Case 4: first row not cols, see if the second is cols  (\n count)
        elif 2*num_cols > num_lines_2 >=0 and second_row[1:].count('')==0 and second_row_nume<=2: 
             # first two rows are not titles nor cols, third row has no missing values (assume to be col names)
            print('Second row as cols!')
            title = first_row_text.strip('_')
            cols = second_row
            df_1 = df.iloc[2:]
            df_1.columns = cols
        elif 2*num_cols > num_lines_1 > max(num_cols-2, 2):
            print('Second row as cols, tangled!')
            # print(num_lines_2)
            cols_text = second_row_text.strip('_')
            title = first_row_text.strip('_')
            df_1 = df.iloc[2:]
       
        elif third_row[1:].count('')==0 and  2*num_cols> num_lines_3 >=0 and third_row_nume<=2:
            print('Thrid row as cols')
            title = "|".join([str(l[0]) + '_' + str(l[1]) for l in combined_2_row]).strip('_').strip('|')
            cols = df.iloc[2].tolist()
            df_1 = df.iloc[3:]
            df_1.columns = cols
        else:
        # by default: no change (no column names in first three rows)   
            df_1 = df.copy()
                    
        return  df_1, title, cols_text
    
    # identify column names: when cols_text is not ''
    def format_columns(df_1, cols_text):
        df_2 = df_1.copy()
        # identify how many name are multi-rows (assume at most has two lines)
        col_list = re.split(r'\n+', cols_text)
        col_list = [e.strip() for e in col_list if not all(ch == '_' for ch in e.strip())]
        num = len(col_list)
        num_cols = df_1.shape[1]
        num_multi_lines = num - num_cols
        
        if num_multi_lines > 0:
            single_row_cols = col_list[num_multi_lines:num-num_multi_lines]
            start = col_list[:num_multi_lines]
            end = col_list[-num_multi_lines:]
            multi_line_cols = [start[i] + '_' + end[i] for i in range(num_multi_lines)]
            # problem: no adjust for order (for now just assume multile line is at the end) -> entails some ML to match
            new_cols = single_row_cols + multi_line_cols
        elif num_multi_lines == 0:
            new_cols = col_list
        else:
            # missing cols
            new_cols = col_list + list(range(num_multi_lines))
            # Q: just leave it empty?
            
        df_2.columns = new_cols[:num_cols]
        
        return df_2 
    
    # combined rows ('\n): by detecting the combination of None with neighbors
    def combine_rows(df_2):
        
        def detect_row(df, row):
            def is_punctuation(char):
                # Regex pattern: match if the character is not alphanumeric
                return bool(re.match(r'[^a-zA-Z0-9]', char))
            def is_only_punctuation(row):
                return all(is_punctuation(str(cell)) for cell in row)
            
            # non_cols  = df.iloc[row].isna().tolist()
            num = df.shape[1]
            check_cols = [idx for idx in range(1, df.shape[1])]  # if non_cols[idx]==0] # start with 0 or 1
            df_up = df.iloc[row - 1].fillna('_').tolist()
            df_row = df.iloc[row].fillna('_').tolist()
            df_dw = df.iloc[row+1].fillna('_').tolist()
            
            df_up_row_none = df.iloc[row-1:row+1].fillna('').map(lambda x: 1 if x=='' else 0)
            list_combined_none = df_up_row_none.sum(axis=0).tolist()
            
            if all(df_up[idx] == '_' for idx in check_cols) and all(df_dw[idx] == '_' for idx in check_cols):
                print(f'Combine three rows, as the row above and below {row} are all missing')
                df_bf = df.iloc[:row-1]
                df_af = df.iloc[row+2:]
                df_md = pd.DataFrame([[df_up[i] + df_row[i] + df_dw[i] for i in range(num)]], columns=df.columns)
                df_md = df_md.apply(lambda col: col.str.replace('_', ' '))
                df_new = pd.concat([df_bf, df_md, df_af], axis=0, ignore_index=True).reset_index(drop=True)
            # elif all(df_up[idx] == '_' for idx in check_cols):
            #     print(f'Combine the row above {row}')
            #     df_bf = df.iloc[:row-1]
            #     df_af = df.iloc[row+1:]
            #     df_md = pd.DataFrame([[df_up[i] + '\n' + df_row[i] for i in range(num)]], columns=df.columns)
            #     df_md = df_md.apply(lambda col: col.str.replace('_', ' ').str.strip())
            #     df_new = pd.concat([df_bf, df_md, df_af], axis=0, ignore_index=True).reset_index(drop=True)
            # add: the whole row only has punctuation but no real stuff
            elif all(is_only_punctuation(str(cell)) for cell in df_row):
                print(f'The whole {row} row is all punctuation, deleted')
                df_bf = df.iloc[:row]
                df_af = df.iloc[row+1:]
                df_new = pd.concat([df_bf, df_af], axis=0, ignore_index=True)   
            # intertwined missing value between rows (two rows for now)
            elif len(list_combined_none) >= 3 and all(x == 1 for x in list_combined_none[1:]): 
                if df.iloc[row-1].isna().sum() == df.shape[1]-1 and df.iloc[row-1, 0] is not None and str(df.iloc[row-1, 0]) != '' and df.iloc[row-1, 0][0].isupper():
                    df_new = df.copy()
                else:
                    print(f'Interwined with the row above {row}')
                    df_bf = df.iloc[:row-1]
                    df_md = pd.DataFrame([[df_up[i] + df_row[i] for i in range(num)]], columns=df.columns)
                    df_md = df_md.apply(lambda col: col.str.replace('_', ' ').str.strip())
                    df_af = df.iloc[row+1:]
                    df_new = pd.concat([df_bf, df_md, df_af], axis=0, ignore_index=True).reset_index(drop=True)  
            else:
                df_new = df.copy()
            df_new = df_new.reset_index(drop=True)
            
            return df_new

        df_3 = df_2.copy()
        row = 0
        while row < df_3.shape[0]-1:
            if row == 0 and all(pd.isna(df_3.iloc[0, 1:])) and pd.to_numeric(df_3.iloc[1, 1:], errors='coerce').notnull().sum()>=1:
                row += 1
                continue
            df_3 = detect_row(df_3, row)
            row += 1
                
        return df_3
    
    # identify (segament)
    df_1, title, cols_text = identify_title_columns(df)
    # format columns: maybe wrong order (can include the column name in the later ML for matching)
    if df_1.empty:
        return {'title': '', 'df': df_1}
    if cols_text!='':
        df_2 = format_columns(df_1, cols_text)
    else:
        df_2 = df_1.copy()
    # identify and combine rows
    # print(f'before combining rows:\n{df_2}')
    df_3 = combine_rows(df_2)
    # print(f'after combining rows:\n{df_3}')
    df_dict = {'title': title, 'df': df_3}
    
    return df_dict


def find_table_position(table_df, pdf_file, table_bbox, page_num=0):
    """
    Find the bounding box of a table by identifying the first and last rows 
    within a given bounding box from Camelot, converting coordinates for pdfplumber.

    Args:
    - table_df (pd.DataFrame): The cleaned table (input as a DataFrame).
    - pdf_file (str): Path to the PDF file.
    - table_bbox (tuple): The list of bounding boxes from Camelot (x0, y0, x1, y1).
    - page_num (int): The page number where the table is located (default is 0).

    Returns:
    - tuple: pdfplumber ordinated bbox (top-left corner, bottom-right corner) of the refined table.
    """
    # If table_bbox is a tuple, we can choose the first bounding box
    if table_bbox:
        x0_table, y0_table, x1_table, y1_table = table_bbox
    else:
        print('Empty bbox')
        return None  # Return None if table_bbox is empty

    if table_df.empty:
        return None
    else:
        if table_df.shape[0]< 2 or table_df.shape[1]<2:
            return None
    
    with pdfplumber.open(pdf_file) as pdf:
        page = pdf.pages[page_num]  
        # display_table_box(pdf_path, table_bbox, page_num=page_num, plumber_ordinate=0)
        # Convert Camelot bbox to pdfplumber coordinate system
        page_height = page.height  
        # print(page_height)
        y0_table_plumber = page_height - y1_table
        y1_table_plumber = page_height - y0_table

        # Extract words only within the transformed bbox
        words = [
            word for word in page.extract_words()
            if x0_table <= word['x0'] <= x1_table and y0_table_plumber <= word['top'] <= y1_table_plumber
        ]
        # display_table_box(pdf_file, (x0_table, y0_table_plumber, x1_table, y1_table_plumber), page_num=page_num, plumber_ordinate=1)
        if not words:
            print(f'empty words')
            return None
        
        def get_bbox_from_mode_or_m(bboxes, axis):
            """Calculate the mode of bounding box coordinates for robustness,
            and fall back to the median if no repetition is found.
            """
            if axis == 0:
                a0_col = 'top'
                a1_col = 'bottom'
            elif axis == 1:
                a0_col = 'x0'
                a1_col = 'x1'
                
            a0_vals = [round(bbox[a0_col], 2) for bbox in bboxes]
            a1_vals = [round(bbox[a1_col], 2) for bbox in bboxes]
            
            def get_mode_or_max(vals):
                mode = Counter(vals).most_common(1)
                if mode and mode[0][1] / len(vals) > 0.4:
                    return mode[0][0]
                return max(vals)
            
            def get_mode_or_min(vals):
                mode = Counter(vals).most_common(1)
                if mode and mode[0][1] / len(vals) > 0.4: 
                    return mode[0][0]
                return min(vals)
            
            a0_result = get_mode_or_min(a0_vals)
            a1_result = get_mode_or_max(a1_vals)
            # print(f'a0, a1: {a0_result, a1_result}')

            return (a0_result, a1_result)
        
        
        def locate_axis_boundary(table_df, axis, words):
            """
            axis = 0: row
            axis = 1: column
            """
            if axis==0:
                first_text = table_df.columns.dropna().tolist()
                first_idx = -1
                if all(isinstance(x, int) and x < 100 for x in first_text) or len(table_df.columns.dropna().tolist())<2:
                    first_idx = None
                    for i in range(table_df.shape[0]):  # Iterate over all columns
                        if len(table_df.iloc[i].dropna().tolist()) > 0:
                            first_idx = i
                            break
                    
                if first_idx is None: 
                    return None, None   
                elif first_idx != -1:
                    first_text = table_df.iloc[first_idx].dropna().tolist()
                
                last_idx = None
                for i in range(1, table_df.shape[0] + 1):  # Iterate from the last column
                    if len(table_df.iloc[-i].dropna().tolist()) >= 2:
                        last_idx = -i
                        last_text = table_df.iloc[last_idx].dropna().tolist()
                        break

                if last_idx is None:
                    return None, None   
                
            elif axis==1:
                # print(table_df.shape, table_df.iloc[:2])
                first_idx = None
                for i in range(table_df.shape[1]):  # Iterate over all columns
                    if len(table_df.iloc[:, i].dropna().tolist()) > 0:
                        first_idx = i
                        first_text = table_df.iloc[:, first_idx].dropna().tolist()
                        break
                if first_idx is None:  # If no non-empty column is found
                    return None, None   
                
                last_idx = None
                for i in range(1, table_df.shape[1] + 1):  # Iterate from the last column
                    if len(table_df.iloc[:, -i].dropna().tolist()) >= 2:
                        last_idx = -i
                        # print(f'found last_idx:{last_idx}')
                        last_text = table_df.iloc[:, last_idx].dropna().tolist()
                        # print(f'last_text:{last_text}')
                        break
                if last_idx is None:
                    return None, None   
                

            else:
                print('Wrong axis!')
                return None, None   
            
            first_text = [str(e).split("\n")[0] for e in first_text]
            first_text = [word for e in first_text for word in re.findall(r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?|\w+\b', str(e))] # prevent separate float
            first_text = [e for e in first_text if len(e)>1]
            last_text = [str(e).split("\n")[-1] for e in last_text]  # deal with \n
            last_text = [word for e in last_text for word in re.findall(r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?|\w+\b', str(e))] # re.findall(r'\b\w+\.\w+|\w+\b'
            last_text = [e for e in last_text if len(e)>1]            
            first_bboxes, last_bboxes = [], []
            for word in words:
                if any(str(cell) in word['text'] for cell in first_text):
                    first_bboxes.append(word)
                if any(str(cell) in word['text'] for cell in last_text):
                    last_bboxes.append(word)
            if not first_bboxes or not last_bboxes:
                return None, None           
            
            first_bbox_mode = get_bbox_from_mode_or_m(first_bboxes, axis)
            last_bbox_mode = get_bbox_from_mode_or_m(last_bboxes, axis)
            first_0 = first_bbox_mode[0]
            last_1 = last_bbox_mode[1]
            
            return first_0, last_1
            
        top, bottom = locate_axis_boundary(table_df, 0, words)    
        left, right = locate_axis_boundary(table_df, 1, words)
        
        if any(val is None for val in [top, bottom, left, right]):
            return None
        
        refined_bbox = (left, top, right, bottom)

        return refined_bbox


def display_table_box(pdf_file, table_bbox, page_num=0, plumber_ordinate=1):
    """
    Display the part of the PDF corresponding to the table's bounding box.

    Args:
    - pdf_file (str): Path to the PDF file.
    - table_bbox (tuple): Bounding box of the table in the form of (x0, y0, x1, y1).
    - page_num (int): The page number where the table is located (default is 0).
    """
    # no need to convert, as the table_bbox is the pdfplumber version of adjusted bbox
    table_bbox_list = list(table_bbox)
    with pdfplumber.open(pdf_file) as pdf:
        page = pdf.pages[page_num]
        page_height = page.height  
        if plumber_ordinate!=1:
            table_bbox_list[1] = page_height - table_bbox[3]
            table_bbox_list[3] = page_height - table_bbox[1]
        cropped_page = page.within_bbox(table_bbox_list)
        im = cropped_page.to_image()
        plt.imshow(im.original)
        plt.axis('off')  
        plt.show()
       
def sanitize_sheet_name(name, page_num):
    # Replace invalid characters with underscores and truncate to 31 characters
        name = re.sub(r'[\\/:*?"<>|]', '_', name)  
        limit = 31 - len(str(page_num)) - 1
        name = name[:limit] + '_' + str(page_num)
        return name
     
def save_tables_to_xlsx(df_dict_list, output_path, start_idx):
    df_dict_list = [df_dict for df_dict in df_dict_list if not df_dict['df'].empty]
    print(f'\nFinal num of df: {len(df_dict_list)}')
    # Use ExcelWriter to write multiple sheets
    with pd.ExcelWriter(output_path) as writer:
        for idx, df_dict in enumerate(df_dict_list):
            # Get the title or use "Sheet" + number if no title is provided
            page_num = df_dict['page']
            sheet_idx = start_idx + idx
            sheet_name = df_dict['title'] if df_dict['title'] else f'Sheet{sheet_idx + 1}'
            sheet_name = sanitize_sheet_name(sheet_name, page_num)
            df_dict['df'].to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Data successfully saved to {output_path}")

def get_sheet_dict(df_dict_list, start_idx):
    sheets_dict = {}
    for idx, df_dict in enumerate(df_dict_list):
        page_num = df_dict['page']
        sheet_idx = start_idx + idx
        sheet_name = df_dict['title'] if df_dict['title'] else f'Sheet{sheet_idx + 1}'
        sheet_name = sheet_name.strip() + "_" + str(page_num)
        sheets_dict[sheet_name] = df_dict['df']
    return sheets_dict

# return the order of column names if there is two-line column names after obtaining the cleaned table
def adjust_column_name_order(plumber_bbox, cleaned_table_df, pdf_path, page_num=0):
    if not plumber_bbox:
        return pd.DataFrame()
    cols = cleaned_table_df.columns.tolist()
    if not any(["_" in str(col) for col in cols]):
        return cleaned_table_df
    # has two-line column names
    print('Adjust column name order..')
    adjusted_table_df = cleaned_table_df.copy()
    # use unique words only for each col (compared with other column names)
    col_word_sets = {col: col.replace('_', " ").split() for col in cols}  
    unique_col_words = {}
    for col, word_set in col_word_sets.items():
        # Get words that are in this column but not in any other column
        other_column_words = set(word for c in cols for word in c.replace('_', " ").split() if c!=col)
        unique_words = [word for word in word_set if word not in other_column_words]
        if unique_words:
            unique_col_words[col] = unique_words
        else:
            unique_col_words[col] = word_set
    # collect the position of all col names and order 
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_num]  
        x0_table, y0_table, x1_table, y1_table = plumber_bbox
        # in case x0 is not accurate that lose column name information
        x0_table += -3
        # need clean the page.extract_words ['text'] -- different handling method for upper number
        covered_words = [
            word for word in page.extract_words()
            if x0_table <= word['x0'] <= x1_table 
            and y0_table-10 <= word['doctop'] <= y1_table 
            and abs(word['top'] - y0_table) <= 20  
        ]
        def remove_single_number(input_string):
            input_string = str(input_string)
            result = re.sub(r'(?<=\w)\d(?=\w)', '', input_string)  # Remove single number between letters
            return result
        covered_words = list(map(lambda word: {**word, 'text': remove_single_number(word['text'])}, covered_words))
        words = [
            word for word in covered_words
            if any(word['text'] in word_set for word_set in unique_col_words.values())  
        ]
        # sorted by x0
        words_sorted = sorted(words, key=lambda w: (w['x0'], w['doctop']))
        reordered_columns = []
        used_cols = set() # avoid repeatedness
        used_x0 = set()
        for word in words_sorted:
            best_match = None
            best_match_count = 0
            for col, word_set in unique_col_words.items():
                if col not in used_cols and word['x0'] not in used_x0:
                    match_count = sum(1 for col_word in word_set if col_word in word['text'])
                    if match_count > best_match_count:
                        best_match = col
                        best_match_count = match_count
            if best_match:
                reordered_columns.append(best_match)
                used_cols.add(best_match)
                used_x0.add(word['x0'])
        # Ensure all original columns are included (in case some were missed in extraction)
        missing_cols = [col for col in cols if col not in used_cols]
        reordered_columns.extend(missing_cols)
        adjusted_table_df.columns = reordered_columns
        print(f'adjusted column names: \n{reordered_columns}')
        
    return adjusted_table_df


if __name__ == "__main__":
    # test 
    pdf_path = "./data/SR/2023_longi-sustainability-report.pdf"
    # pdf_path = "./data/table/Totalenergies_2024 (dragged) 2.pdf"
    # pdf_path = './data/SR/LGES_2020.pdf'
    # pdf_path = './data/table/Totalenergies_2024 (dragged) 104.pdf'
    sheets_dict, df_dict_list = detect_extract_tables(pdf_path, save_tables=True)
    # print(sheets_dict)
